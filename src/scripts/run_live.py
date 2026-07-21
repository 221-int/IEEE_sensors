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
from eyeblink.robust import get_preprocessor
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
    ap.add_argument("--show", action="store_true", help="OpenCV 창으로 표시")
    ap.add_argument("--log", default=None, help="프레임별 CSV 경로")
    ap.add_argument("--seconds", type=float, default=0.0, help="N초 후 자동 종료(0=무한)")
    a = ap.parse_args()

    if not L.mediapipe_available():
        raise SystemExit("mediapipe is required:  pip install mediapipe")
    import cv2
    import mediapipe as mp

    pre = get_preprocessor(a.preprocess)
    try:
        landmarker = L.build_landmarker()           # 모델 파일 없으면 안내 후 종료
    except (FileNotFoundError, RuntimeError) as e:
        raise SystemExit(str(e))
    pipe = BlinkPipeline()
    fps = FpsMeter(window=30)
    timer = StageTimer()

    writer = fh = None
    if a.log:
        fh = open(a.log, "w", newline="")
        writer = csv.writer(fh)
        writer.writerow(["time", "cap_fps", "landmark_ms", "ear",
                         "calibrating", "state", "blink", "total_blinks", "bpm"])

    print("[live] Calib: keep eyes OPEN first -> when prompted, CLOSE eyes tight. (q: quit)")
    t_start = time.perf_counter()
    try:
        for frame in frame_source(a):
            frame = pre(frame)                       # 강건화 훅(기본 무동작)
            fps.tick()
            t = time.time()
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with timer.timed("landmark"):
                res = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(t * 1000))
            lm_ms = timer._s["landmark"][-1]

            if not res.face_landmarks:
                if a.show and _show(cv2, frame, f"no face  fps={fps.fps:.1f}"):
                    break
            else:
                ear = L.frame_ear(res.face_landmarks[0], w, h)
                st = pipe.update(t, ear)
                blink = 0
                if st["calibrating"]:
                    hint = "OPEN eyes" if st["phase"] == "open" else "CLOSE eyes tight"
                    label = f"CALIB [{hint}] {int(st['progress']*100)}%  fps={fps.fps:.1f}"
                    m = {"total_blinks": 0, "blink_rate_bpm": 0.0}
                else:
                    m = st["metrics"]
                    label = (f"{st['state']} blinks={m['total_blinks']} "
                             f"bpm={m['blink_rate_bpm']:.1f} fps={fps.fps:.1f} lm={lm_ms:.0f}ms")
                    if st["blink"] is not None:
                        blink = 1
                        print(f"blink #{m['total_blinks']} dur={st['blink'].duration*1000:.0f}ms "
                              f"completeness={st['blink'].completeness:.2f}")
                if writer:
                    writer.writerow([f"{t:.4f}", f"{fps.fps:.2f}", f"{lm_ms:.2f}",
                                     f"{ear:.4f}", int(st['calibrating']),
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
        print(f"\n[profile] landmark: mean={s['mean_ms']:.1f}ms p95={s['p95_ms']:.1f}ms "
              f"-> max ~{s['max_fps']:.1f} fps  (frames measured {s['n']})")
    print(f"[profile] capture fps(recent)={fps.fps:.1f}")


def _show(cv2, frame, label):
    cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 220, 0), 2)
    cv2.imshow("eyeblink", frame)
    return (cv2.waitKey(1) & 0xFF) == ord("q")


if __name__ == "__main__":
    main()
