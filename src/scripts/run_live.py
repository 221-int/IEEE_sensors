"""run_live — 실시간 깜빡임 파이프라인 (스트림/웹캠 -> EAR -> STD -> 카운트).

계측: 프레임 fps + FaceLandmarker 처리시간(ms)을 화면에 표시하고, --log 로
프레임별 CSV 를 남긴다(엣지 실현성/논문 수치용).

프레임 소스:
    --source url     (기본) MJPEG 서버에서 수신.  --url http://HOST:8000/stream.mjpg
    --source webcam  로컬 카메라 직접.             --camera 0

실행 (src/ 에서):
    python -m scripts.run_live --source webcam --show
    python -m scripts.run_live --source url --url http://192.168.0.20:8000/stream.mjpg --show
    python -m scripts.run_live --source url --seconds 60 --log run.csv
"""
import argparse
import csv
import time

from eyeblink import config
from eyeblink import landmarks as L
from eyeblink.pipeline import BlinkPipeline
from eyeblink.robust import get_preprocessor, SuperResolution
from eyeblink.frontend import EarFrontend, SrGate
from eyeblink.profiling import FpsMeter, StageTimer


def frame_source(args):
    """소스에 따라 BGR 프레임 제너레이터 반환."""
    if args.source == "webcam":
        import cv2
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise SystemExit(f"Cannot open camera (--camera {args.camera}). "
                             "Try another index (0/1) or check camera permission.")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
        finally:
            cap.release()
    else:
        from eyeblink.streaming.client import mjpeg_frames
        yield from mjpeg_frames(args.url)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["url", "webcam"], default="url")
    ap.add_argument("--url",
                    default=f"http://127.0.0.1:{config.STREAM_PORT}{config.STREAM_PATH}")
    ap.add_argument("--camera", type=int, default=config.CAM_INDEX)
    ap.add_argument("--preprocess", choices=["none", "sr", "lowlight"], default="none")
    ap.add_argument("--sr-model", choices=["fsrcnn", "espcn"], default=config.SR_MODEL,
                    help="conditional SR model (needs models/<MODEL>_x<scale>.pb)")
    ap.add_argument("--sr-scale", type=int, default=config.SR_SCALE, help="SR upscale factor")
    ap.add_argument("--sr-w-eye", type=float, default=config.SR_W_EYE_MIN,
                    help="gate: turn SR on when eye width(px) < this")
    ap.add_argument("--show", action="store_true", help="show OpenCV window")
    ap.add_argument("--log", default=None, help="per-frame CSV path")
    ap.add_argument("--seconds", type=float, default=0.0, help="auto-stop after N s (0=inf)")
    a = ap.parse_args()

    if not L.mediapipe_available():
        raise SystemExit("mediapipe is required:  pip install mediapipe")
    import cv2
    import mediapipe as mp

    # SR 은 프론트엔드의 조건부 얼굴 ROI two-pass 로 처리(전체 프레임 아님).
    # lowlight 만 전체 프레임 전처리로 유지, none 은 무동작.
    pre = get_preprocessor("lowlight") if a.preprocess == "lowlight" \
        else get_preprocessor("none")
    sr = SuperResolution(model=a.sr_model, scale=a.sr_scale) if a.preprocess == "sr" else None
    try:
        landmarker = L.build_landmarker()           # 모델 파일 없으면 안내 후 종료
    except (FileNotFoundError, RuntimeError) as e:
        raise SystemExit(str(e))
    gate = SrGate(on_below=a.sr_w_eye, enable=(sr is not None))
    frontend = EarFrontend(landmarker, sr=sr, gate=gate)   # pass1(VIDEO) + 조건부 pass2(IMAGE)
    pipe = BlinkPipeline()
    fps = FpsMeter(window=30)
    timer = StageTimer()

    writer = fh = None
    if a.log:
        fh = open(a.log, "w", newline="")
        writer = csv.writer(fh)
        writer.writerow(["time", "cap_fps", "landmark_ms", "sr_ms", "w_eye", "used_sr",
                         "ear", "calibrating", "state", "blink", "total_blinks", "bpm"])

    print("[live] Calib: keep eyes OPEN first -> when prompted, CLOSE eyes tight. (q: quit)")
    t_start = time.perf_counter()
    try:
        for frame in frame_source(a):
            if a.preprocess == "lowlight":
                frame = pre(frame)                   # 전체 프레임 저조도 보정(선택)
            fps.tick()
            t = time.time()
            r = frontend.process(frame, int(t * 1000))   # pass1(VIDEO)[+조건부 SR pass2]
            timer.add("landmark", r.lm_ms)
            if r.used_sr:
                timer.add("sr", r.sr_ms)
            lm_ms = r.lm_ms

            if not r.has_face:
                if a.show and _show(cv2, frame, f"no face  fps={fps.fps:.1f}"):
                    break
            else:
                st = pipe.update(t, r.ear)
                blink = 0
                if st["calibrating"]:
                    hint = "OPEN eyes" if st["phase"] == "open" else "CLOSE eyes tight"
                    label = f"CALIB [{hint}] {int(st['progress']*100)}%  fps={fps.fps:.1f}"
                    m = {"total_blinks": 0, "blink_rate_bpm": 0.0}
                else:
                    m = st["metrics"]
                    sr_tag = " SR" if r.used_sr else ""
                    label = (f"{st['state']} blinks={m['total_blinks']} "
                             f"bpm={m['blink_rate_bpm']:.1f} fps={fps.fps:.1f} "
                             f"lm={lm_ms:.0f}ms we={r.w_eye:.0f}{sr_tag}")
                    if st["blink"] is not None:
                        blink = 1
                        print(f"blink #{m['total_blinks']} dur={st['blink'].duration*1000:.0f}ms "
                              f"completeness={st['blink'].completeness:.2f}")
                if writer:
                    writer.writerow([f"{t:.4f}", f"{fps.fps:.2f}", f"{lm_ms:.2f}",
                                     f"{r.sr_ms:.2f}", f"{r.w_eye:.1f}", int(r.used_sr),
                                     f"{r.ear:.4f}", int(st['calibrating']),
                                     st.get('state', ''), blink,
                                     m['total_blinks'], f"{m['blink_rate_bpm']:.2f}"])
                if a.show and _show(cv2, frame, label):
                    break

            if a.seconds and (time.perf_counter() - t_start) >= a.seconds:
                break
    except KeyboardInterrupt:
        pass
    except ConnectionError as e:                      # MJPEG 접속/수신 실패
        print(f"[live] {e}")
    finally:
        if fh:
            fh.close()
        if a.show:
            cv2.destroyAllWindows()

    s = timer.summary().get("landmark")
    if s:
        print(f"\n[profile] landmark(pass1[+pass2]): mean={s['mean_ms']:.1f}ms "
              f"p95={s['p95_ms']:.1f}ms -> max ~{s['max_fps']:.1f} fps  (frames measured {s['n']})")
    s2 = timer.summary().get("sr")
    if s2:
        print(f"[profile] sr upsample (SR-on frames only): mean={s2['mean_ms']:.1f}ms "
              f"p95={s2['p95_ms']:.1f}ms  n={s2['n']}")
    print(f"[profile] capture fps(recent)={fps.fps:.1f}")


def _show(cv2, frame, label):
    cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 220, 0), 2)
    cv2.imshow("eyeblink", frame)
    return (cv2.waitKey(1) & 0xFF) == ord("q")


if __name__ == "__main__":
    main()
