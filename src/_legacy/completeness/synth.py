"""synth — 합성 깜빡임 특징 생성 (카메라 없이 파이프라인 점검용).

문헌 가정을 인코딩: 완전 깜빡임은 완성도가 높고(≈1) 더 깊게/길게 감기며,
불완전 깜빡임은 완성도가 낮고(눈을 끝까지 안 감음) 얕고 빠르다.
※ 합성 수치이므로 결과 자체는 의미 없음 — 동작 확인·지연 측정용.
"""
import numpy as np

from . import config


def make_synthetic(n_subjects=8, per_class=40, seed=0):
    """(subject_id, label, feat) 리스트 반환. label 0=complete, 1=incomplete."""
    rng = np.random.default_rng(seed)
    rows = []
    idx_comp = config.FEATURE_NAMES.index("completeness")
    for sj in range(n_subjects):
        soff = rng.normal(0, 0.03, size=config.FEATURE_LEN)
        for label in (0, 1):                       # 0 complete, 1 incomplete
            for _ in range(per_class):
                if label == 0:                     # complete: 완성도↑, 길고 깊음
                    comp = rng.normal(0.95, 0.05)
                    dur = rng.normal(0.32, 0.05)
                    vel = rng.normal(6.0, 1.2)
                else:                               # incomplete: 완성도↓, 얕고 빠름
                    comp = rng.normal(0.55, 0.10)
                    dur = rng.normal(0.20, 0.04)
                    vel = rng.normal(8.5, 1.6)
                comp = float(np.clip(comp, 0.05, 1.2))
                dur = max(dur, 0.05)
                min_ratio = float(np.clip(1.0 - comp, 0.0, 0.95))
                dc, do = dur * 0.45, dur * 0.55
                f = np.array([
                    dur, dc, dur * 0.2, do,
                    comp, min_ratio,
                    vel, vel * 0.8,
                    dc / do, 1.25,
                    comp * dur,
                ], np.float32) + soff.astype(np.float32)
                f[idx_comp] = comp                 # 완성도는 오프셋 노이즈 제외
                rows.append((f"S{sj:02d}", label, f))
    return rows
