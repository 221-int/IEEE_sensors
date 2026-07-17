"""
benchmark.py — PC / 엣지(라즈베리파이 5) 파이프라인 벤치마크.

목적: 동일한 입력 영상을 PC와 엣지에서 각각 돌려, 단계별 지연시간·처리량·
자원 사용을 측정하고 비교한다. 분류 출력 자체는 순수 NumPy 결정론 연산이라
같은 입력이면 두 기기에서 동일하므로, 여기서는 *실시간 거동*을 측정한다.

측정 항목:
  - 단계별 latency: landmark 검출 / EAR 계산 / 세그먼터 / 분류기 (mean·p95·max)
  - end-to-end 처리량(fps)과 지속 가능한 최대 fps
  - CPU%, 메모리(RSS), SoC 온도(라즈베리파이의 thermal_zone)

실행:
  python benchmark.py --mock                 # 카메라/MediaPipe 없이 harness 점검
  python benchmark.py --video clip.mp4        # 동일 영상으로 PC/엣지 비교(권장)
  python benchmark.py --video clip.mp4 --stride 2   # fps 절반으로 다운샘플
  python benchmark.py                         # 웹캠 실시간

권장: PC와 라즈베리파이에서 *같은 영상 파일*로 --video를 돌려 결과표를 비교.
"""
import argparse
import os
import time

import numpy as np

import config
from blink_segmenter import BlinkSegmenter
from model import BlinkClassifier

try:
    import psutil
    _PS = True
except Exception:
    _PS = False


def read_soc_temp():
    """라즈베리파이 등에서 SoC 온도(℃). 없으면 None."""
    path = "/sys/class/thermal/thermal_zone0/temp"
    try:
        return int(open(path).read().strip()) / 1000.0
    except Exception:
        return None


def summarize(name, arr):
    if not arr:
        print(f"{name:22s} (해당 없음)")
        return
    a = np.asarray(arr)
    print(f"{name:22s} mean={a.mean():7.3f}ms  p95={np.percentile(a,95):7.3f}ms  "
          f"max={a.max():7.3f}ms  n={len(a)}")


# ── mock: 카메라/MediaPipe 없이 세그먼터+분류기만 계측 (샌드박스 점검용) ──────
def run_mock(n_frames=6000, fps=60.0):
    rng = np.random.default_rng(0)
    seg = BlinkSegmenter(baseline=0.30, threshold=0.225)
    clf = BlinkClassifier()
    ts_seg, ts_cls = [], []
    blinks = 0
    t, dt = 0.0, 1.0 / fps
    for i in range(n_frames):
        phase = i % 90                      # 약 1.5초마다 한 번 깜빡임
        if 40 <= phase <= 48:
            ear = 0.30 - 0.20 * np.exp(-((phase - 44) ** 2) / (2 * 2.0 ** 2))
        else:
            ear = 0.30 + rng.normal(0, 0.003)
        t0 = time.perf_counter(); ev = seg.update(t, ear)
        ts_seg.append((time.perf_counter() - t0) * 1e3)
        if ev is not None:
            blinks += 1
            t0 = time.perf_counter(); clf.predict(ev.features[None, :])
            ts_cls.append((time.perf_counter() - t0) * 1e3)
        t += dt
    print(f"[mock] frames={n_frames}  blinks={blinks}")
    summarize("segmenter.update", ts_seg)
    summarize("classifier.predict", ts_cls)
    print("(mock는 MediaPipe 앞단을 제외한 경량부만 계측 — harness 동작 확인용)")


# ── real: 영상/카메라 입력으로 전체 파이프라인 계측 ─────────────────────────
def run_real(video, stride, max_frames):
    import cv2
    import mediapipe as mp
    import detector as D

    is_cam = video in (None, "0", 0)
    cap = cv2.VideoCapture(0 if is_cam else video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    landmarker = D.build_landmarker()
    calib = D.Calibrator()
    seg = None
    clf = BlinkClassifier.load() if os.path.exists(config.MODEL_PATH) else BlinkClassifier()

    ts_lm, ts_feat, ts_seg, ts_cls = [], [], [], []
    n = 0; blinks = 0; idx = -1
    if _PS:
        proc = psutil.Process(os.getpid()); proc.cpu_percent(None)
    t_start = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if stride > 1 and idx % stride != 0:
            continue
        n += 1
        ts_ms = int(time.time() * 1000) if is_cam else int(idx * 1000.0 / src_fps)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        t0 = time.perf_counter(); res = landmarker.detect_for_video(img, ts_ms)
        ts_lm.append((time.perf_counter() - t0) * 1e3)
        if not res.face_landmarks:
            continue
        lms = res.face_landmarks[0]

        t0 = time.perf_counter(); ear = D.frame_ear(lms, w, h)
        ts_feat.append((time.perf_counter() - t0) * 1e3)

        if not calib.done:
            calib.update(ear); continue
        if seg is None:
            seg = BlinkSegmenter(calib.baseline, calib.threshold)

        now = ts_ms / 1000.0
        t0 = time.perf_counter(); ev = seg.update(now, ear)
        ts_seg.append((time.perf_counter() - t0) * 1e3)
        if ev is not None:
            blinks += 1
            t0 = time.perf_counter(); clf.predict(ev.features[None, :])
            ts_cls.append((time.perf_counter() - t0) * 1e3)

        if max_frames and n >= max_frames:
            break

    elapsed = time.perf_counter() - t_start
    cap.release()

    print(f"\n[real] processed={n} frames  blinks={blinks}  wall={elapsed:.2f}s  "
          f"throughput={n/max(elapsed,1e-6):.1f} fps"
          + (f"  (input {src_fps:.0f}fps, stride {stride})" if not is_cam else ""))
    summarize("landmark.detect", ts_lm)
    summarize("frame_ear", ts_feat)
    summarize("segmenter.update", ts_seg)
    summarize("classifier.predict", ts_cls)
    e2e = sum(np.mean(x) if x else 0.0 for x in (ts_lm, ts_feat, ts_seg))
    print(f"{'end-to-end/frame':22s} mean~{e2e:7.3f}ms  -> 지속가능 최대 ~{1000/max(e2e,1e-6):.1f} fps")
    if _PS:
        print(f"CPU={proc.cpu_percent(None):.0f}%   RSS={proc.memory_info().rss/1e6:.0f} MB")
    temp = read_soc_temp()
    if temp is not None:
        print(f"SoC temp={temp:.1f} C")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="영상 파일 경로 (생략/0 이면 웹캠)")
    ap.add_argument("--mock", action="store_true",
                    help="카메라/MediaPipe 없이 경량부만 계측 (점검용)")
    ap.add_argument("--stride", type=int, default=1,
                    help="N프레임마다 1개 처리 (fps 다운샘플)")
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()
    if a.mock:
        run_mock()
    else:
        run_real(a.video, a.stride, a.max_frames)


if __name__ == "__main__":
    main()
