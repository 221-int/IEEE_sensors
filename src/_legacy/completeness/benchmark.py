"""benchmark — 엣지 성능 하니스 (PC 베이스라인 → Pi 에서 동일 실행).

측정:
  [정합성/성능]  LOSO 정확도 · 모델 params · 모델 파일 크기(KB)   ← Pi 에서 동일해야 함
  [지연]         분류기 추론 latency, 특징 추출 latency, 프레임당 세그먼터 latency
  [처리량]       비-MediaPipe 연산의 fps 상한(추정)
  [메모리]       최대 상주 메모리(ru_maxrss)
  [랜드마크]     MediaPipe 스테이지 latency — mediapipe+task 있으면 측정, 없으면
                 "Pi 에서 측정" 로 표시 (엣지의 지배적 비용, 실현성 관문)

PC 에서 베이스라인을 찍고, 라즈베리파이에서 같은 명령을 돌려 비교표를 만든다.

실행 (src/ 에서):
    python -m scripts.benchmark
    python -m scripts.benchmark --target-fps 30
"""
import argparse
import os
import platform
import resource
import time

import numpy as np

from eyeblink import config
from eyeblink import features as F
from eyeblink.classifier import BlinkClassifier
from eyeblink.segmenter import BlinkSegmenter
from eyeblink.synth import make_synthetic
from scripts import train as T


def _percentile_ms(fn, repeats, warmup=50):
    for _ in range(warmup):
        fn()
    ts = np.empty(repeats)
    for i in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts[i] = (time.perf_counter() - t0) * 1e3
    return float(ts.mean()), float(np.percentile(ts, 95))


def _fake_blink_series(n=12):
    """완전 깜빡임 하나에 해당하는 (times, ear_ratio) 합성 시계열."""
    t = np.linspace(0.0, 0.30, n)
    e = 1.0 - 0.9 * np.exp(-((t - 0.15) ** 2) / (2 * 0.05 ** 2))
    return t, e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-fps", type=float, default=30.0,
                    help="프레임당 예산 계산용 목표 fps")
    ap.add_argument("--repeats", type=int, default=3000)
    args = ap.parse_args()

    # ── 데이터·모델 준비 ────────────────────────────────────────────────────
    if not os.path.exists(config.BLINK_CSV):
        T.save_rows(make_synthetic())
    try:
        X, y, subj = T.load_csv()
    except ValueError:
        T.save_rows(make_synthetic()); X, y, subj = T.load_csv()

    # ── 정합성/성능 (Pi 에서 동일해야 하는 값) ───────────────────────────────
    cm = T.loso(X, y, subj)
    acc, _ = T.report(cm, "LOSO (Pi 와 동일해야 함)")
    clf = BlinkClassifier().fit(X, y)
    clf.save()
    size_kb = os.path.getsize(config.MODEL_PATH) / 1024

    # ── 지연 측정 ────────────────────────────────────────────────────────────
    x1 = X[:1].astype(np.float64)
    infer_mean, infer_p95 = _percentile_ms(lambda: clf.predict(x1), args.repeats)

    tb, eb = _fake_blink_series()
    feat_mean, feat_p95 = _percentile_ms(lambda: F.extract(tb, eb, 0.1), args.repeats)

    seg = BlinkSegmenter(baseline=0.3, threshold=0.225, closed_ratio=0.1)
    frame_mean, frame_p95 = _percentile_ms(
        lambda: seg.update(time.perf_counter(), 0.28), args.repeats)

    max_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    # ── MediaPipe 랜드마크 스테이지 (있으면 측정) ────────────────────────────
    from eyeblink import landmarks as L
    mp_note = None
    if L.mediapipe_available() and os.path.exists(config.TASK_PATH):
        try:
            lm = L.build_landmarker()
            import mediapipe as mp
            img = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=np.zeros((480, 640, 3), np.uint8))
            m_mean, m_p95 = _percentile_ms(
                lambda: lm.detect_for_video(img, int(time.time() * 1000)),
                repeats=200, warmup=10)
            mp_note = f"mean={m_mean:.2f} ms  p95={m_p95:.2f} ms"
        except Exception as e:            # pragma: no cover
            mp_note = f"측정 실패: {e}"
    else:
        mp_note = "미측정 (mediapipe/face_landmarker.task 없음) — 라즈베리파이에서 측정 필요"

    # ── 프레임 예산 ──────────────────────────────────────────────────────────
    budget_ms = 1000.0 / args.target_fps
    our_frame_ms = frame_mean            # EAR·세그먼터 등 비-랜드마크 프레임 비용

    # ── 요약 출력 + 저장 ─────────────────────────────────────────────────────
    lines = [
        "=== EDGE BENCHMARK ===",
        f"플랫폼            : {platform.platform()}  ({platform.machine()})",
        f"[정합성] LOSO acc : {acc*100:.1f}%   (Pi 에서 동일해야 함)",
        f"[모델]  params    : {clf.param_count()}   file={size_kb:.1f} KB",
        f"[지연]  분류 추론  : mean={infer_mean:.4f} ms  p95={infer_p95:.4f} ms  (/blink)",
        f"[지연]  특징 추출  : mean={feat_mean:.4f} ms  p95={feat_p95:.4f} ms  (/blink)",
        f"[지연]  프레임 처리: mean={frame_mean:.4f} ms  p95={frame_p95:.4f} ms  (/frame, 랜드마크 제외)",
        f"[처리량] 비-랜드마크 fps 상한(추정): {1000.0/max(our_frame_ms,1e-6):,.0f} fps",
        f"[메모리] 최대 상주 : {max_rss_mb:.1f} MB",
        f"[랜드마크] MediaPipe: {mp_note}",
        f"[예산]  목표 {args.target_fps:.0f}fps → 프레임당 {budget_ms:.2f} ms; "
        f"우리 연산 {our_frame_ms:.3f} ms 사용 → 랜드마크에 남는 예산 ≈ {budget_ms-our_frame_ms:.2f} ms",
    ]
    out = "\n".join(lines)
    print("\n" + out)
    path = os.path.join(config.RESULTS_DIR, "edge_benchmark.txt")
    with open(path, "w") as f:
        f.write(out + "\n")
    print(f"\n결과 저장 -> {path}")


if __name__ == "__main__":
    main()
