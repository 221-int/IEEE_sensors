"""fps_sweep — 입력 영상을 여러 목표 fps로 다운샘플해 STD 검출을 비교.

목적(박사님): 프레임레이트를 60/30/15fps 로 낮춰도 인식 정확도/실시간성이
유지되는지 -> 최소 사양(예: 15fps 로도 충분한가) 규명.

방법:
  1) 원본 영상의 모든 프레임에서 EAR 을 한 번만 추출(랜드마커 throughput 도 측정).
  2) EAR 분포 백분위수로 자동 캘리브(개안 baseline / 폐안 floor).
  3) 목표 fps 마다 EAR 시퀀스를 데시메이션 -> BlinkSTD 로 깜빡임 수 집계.
  4) 목표별 (유효 fps, 깜빡임 수, 정답 대비 오차) 표 출력.

실행 (src/ 에서):
    python -m scripts.fps_sweep --video clip.mp4 --targets 60 30 15
    python -m scripts.fps_sweep --video clip.mp4 --targets 60 30 15 --truth 42
"""
import argparse

import numpy as np

from eyeblink import config
from eyeblink import landmarks as L
from eyeblink.std import BlinkSTD
from eyeblink.profiling import StageTimer


def extract_ears(video):
    """영상의 모든 프레임 -> (times, ears, native_fps, landmarker ms 요약)."""
    import cv2
    if not L.mediapipe_available():
        raise SystemExit("mediapipe 가 필요합니다:  pip install mediapipe")

    cap = cv2.VideoCapture(video)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    landmarker = L.build_landmarker()
    timer = StageTimer()
    times, ears = [], []
    i = 0
    import mediapipe as mp
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with timer.timed("landmark"):
            res = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                int(i * 1000 / native_fps))
        if res.face_landmarks:
            ears.append(L.frame_ear(res.face_landmarks[0], w, h))
            times.append(i / native_fps)
        i += 1
    cap.release()
    return np.array(times), np.array(ears), native_fps, timer.summary()


def auto_calib(ears, open_pct=85, closed_pct=3):
    baseline = float(np.percentile(ears, open_pct))
    floor = float(np.percentile(ears, closed_pct))
    return baseline, floor


def count_blinks(times, ears, baseline, floor):
    std = BlinkSTD(baseline, floor)
    n = 0
    for t, e in zip(times, ears):
        if std.update(float(t), float(e)):
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="입력 영상 파일(mp4 등)")
    ap.add_argument("--targets", type=int, nargs="+", default=[60, 30, 15])
    ap.add_argument("--truth", type=int, default=None, help="정답 깜빡임 수(있으면 오차 계산)")
    a = ap.parse_args()

    times, ears, native, lm = extract_ears(a.video)
    print(f"[sweep] frames_with_face={len(ears)}  native_fps={native:.1f}")
    if lm.get("landmark"):
        d = lm["landmark"]
        print(f"[sweep] landmarker: mean={d['mean_ms']:.1f}ms p95={d['p95_ms']:.1f}ms "
              f"-> max ~{d['max_fps']:.1f} fps")
    baseline, floor = auto_calib(ears)
    print(f"[sweep] auto-calib: baseline={baseline:.3f} floor={floor:.3f}\n")

    print(f"{'target_fps':>10}{'eff_fps':>9}{'frames':>8}{'blinks':>8}{'err':>7}")
    for tgt in a.targets:
        step = max(1, round(native / tgt))
        eff = native / step
        t2, e2 = times[::step], ears[::step]
        n = count_blinks(t2, e2, baseline, floor)
        err = "" if a.truth is None else f"{n - a.truth:+d}"
        print(f"{tgt:>10}{eff:>9.1f}{len(e2):>8}{n:>8}{err:>7}")


if __name__ == "__main__":
    main()
