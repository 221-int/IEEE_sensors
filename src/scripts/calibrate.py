"""calibrate — 피험자별 EAR 캘리브레이션 실험 (박사님 프로토콜).

절차(피험자별, 자동 진행):
  1) "눈 편하게 뜨고" OPEN_SEC 초   -> 개안 EAR 수집
  2) "눈 감으세요"   CLOSED_SEC 초  -> 폐안 EAR 수집
  3) 개인 임계값(baseline/floor/STD) 확정
  4) TRIALS 회: "지금 깜빡이세요" 지시 후 창 안에서 STD 가 정확히 1회로 세는지 검증
  결과 CSV(results/):
    calib_<subject>_frames.csv   프레임별 EAR(phase 포함) — 통계 분석용
    calib_<subject>_trials.csv   시행별 detected/correct
    calib_summary.csv            피험자 요약(임계값·EAR 통계·정확도) 누적

실행 (src/ 에서):
  python -m scripts.calibrate --subject hanul --source webcam --show
  python -m scripts.calibrate --subject hanul --source url --url http://192.168.0.18:8000/stream.mjpg --show
"""
import argparse
import os
import time

from eyeblink import config
from eyeblink import landmarks as L
from eyeblink.experiment import (CalibrationRecorder, write_frames_csv,
                                 write_trials_csv, append_summary_csv)
from scripts.run_live import frame_source


def _overlay(cv2, frame, text, color=(0, 220, 0)):
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imshow("calibrate", frame)


def _run_window(gen, cv2, mp, landmarker, seconds, prompt, show, on_ear=None):
    """seconds 동안 프레임에서 EAR 추출. on_ear(t, ear) 가 있으면 호출."""
    t0 = time.time()
    while True:
        t = time.time()
        remain = seconds - (t - t0)
        if remain <= 0:
            break
        frame = next(gen, None)
        if frame is None:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(t * 1000))
        got_face = bool(res.face_landmarks)
        if got_face and on_ear is not None:
            on_ear(t, L.frame_ear(res.face_landmarks[0], w, h))
        if show:
            tag = prompt if got_face else prompt + "  (no face)"
            _overlay(cv2, frame, f"{tag}  {remain:.1f}s",
                     (0, 220, 0) if got_face else (0, 0, 255))
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                raise KeyboardInterrupt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True, help="피험자 이름/ID (예: hanul)")
    ap.add_argument("--source", choices=["url", "webcam"], default="url")
    ap.add_argument("--url",
                    default=f"http://127.0.0.1:{config.STREAM_PORT}{config.STREAM_PATH}")
    ap.add_argument("--camera", type=int, default=config.CAM_INDEX)
    ap.add_argument("--open-sec", type=float, default=config.CALIB_OPEN_SEC)
    ap.add_argument("--closed-sec", type=float, default=config.CALIB_CLOSED_SEC)
    ap.add_argument("--trials", type=int, default=config.CALIB_TRIALS)
    ap.add_argument("--trial-sec", type=float, default=config.CALIB_TRIAL_SEC)
    ap.add_argument("--outdir", default=config.RESULTS_DIR)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    if not L.mediapipe_available():
        raise SystemExit("mediapipe 가 필요합니다:  pip install mediapipe")
    import cv2
    import mediapipe as mp
    try:
        landmarker = L.build_landmarker()
    except (FileNotFoundError, RuntimeError) as e:
        raise SystemExit(str(e))

    rec = CalibrationRecorder(a.subject)
    gen = frame_source(a)
    ema = {"v": None}

    def feed_smoothed(t, ear):
        ema["v"] = ear if ema["v"] is None else \
            config.EMA_ALPHA * ear + (1 - config.EMA_ALPHA) * ema["v"]
        if rec.feed_trial(t, ema["v"]):
            print(f"  -> STD 카운트!")

    print(f"[calib] 피험자 = {a.subject}")
    try:
        # 1) 개안
        _run_window(gen, cv2, mp, landmarker, 3.0, "READY: 곧 개안 측정", a.show)
        print("[1/3] 눈 편하게 뜨고 계세요...")
        _run_window(gen, cv2, mp, landmarker, a.open_sec, "OPEN eyes (relax)", a.show, rec.add_open)

        # 2) 폐안
        _run_window(gen, cv2, mp, landmarker, 3.0, "READY: 곧 눈 감기", a.show)
        print("[2/3] 눈 감아주세요...")
        _run_window(gen, cv2, mp, landmarker, a.closed_sec, "CLOSE eyes", a.show, rec.add_closed)

        # 3) 임계값 확정
        c = rec.finalize_thresholds()
        print(f"[calib] baseline={c['baseline']:.3f}  floor={c['closed_floor']:.3f}  "
              f"t_open={c['t_open']:.3f}  t_closed={c['t_closed']:.3f}")
        if c["baseline"] is None or c["closed_floor"] is None:
            raise SystemExit("캘리브 실패: 얼굴/눈이 충분히 잡히지 않았어요. 조명/거리 조정 후 재시도.")

        # 4) 깜빡임 검증 시행
        print(f"[3/3] 이제 지시에 맞춰 '한 번씩' 깜빡이세요. 총 {a.trials}회.")
        for i in range(1, a.trials + 1):
            _run_window(gen, cv2, mp, landmarker, 1.5, f"Trial {i}/{a.trials}: READY", a.show)
            print(f"  Trial {i}: 지금 깜빡이세요!")
            ema["v"] = None
            rec.begin_trial(i)
            _run_window(gen, cv2, mp, landmarker, a.trial_sec,
                        f"Trial {i}/{a.trials}: BLINK NOW", a.show, feed_smoothed)
            detected, correct = rec.end_trial(expected=1)
            print(f"  Trial {i}: detected={detected}  {'OK' if correct else 'X'}")
    except KeyboardInterrupt:
        print("\n[calib] 중단됨.")
    finally:
        if a.show:
            cv2.destroyAllWindows()

    # 저장
    os.makedirs(a.outdir, exist_ok=True)
    fp = os.path.join(a.outdir, f"calib_{a.subject}_frames.csv")
    tp = os.path.join(a.outdir, f"calib_{a.subject}_trials.csv")
    sp = os.path.join(a.outdir, "calib_summary.csv")
    write_frames_csv(rec, fp)
    write_trials_csv(rec, tp)
    append_summary_csv(rec, sp)

    tr = rec.summary_trials()
    print(f"\n[calib] 완료. 정확도 {tr['correct']}/{tr['trials']} "
          f"({tr['accuracy']*100:.0f}%)")
    print(f"  프레임: {fp}\n  시행:   {tp}\n  요약:   {sp}")


if __name__ == "__main__":
    main()
