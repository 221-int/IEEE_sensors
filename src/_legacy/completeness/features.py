"""features — 깜빡임별 운동학 특징 + 완성도(completeness).

각 깜빡임을 (시간, ear_ratio) 시계열로 받아 config.FEATURE_NAMES 순서의
특징 벡터를 만든다. 여기서 ear_ratio = EAR / 개안 baseline.

★ 핵심(비교 논문과의 차별점): 깜빡임을 이진 이벤트로 세지 않고, 각 깜빡임의
'완성도'를 폐안 floor 기준으로 정규화해 측정한다.
    completeness = (1 - min_ratio) / (1 - closed_ratio)
    - 1.0 근처  : 완전 깜빡임(눈을 끝까지 감음)
    - 낮을수록  : 불완전 깜빡임(덜 감음) — 안구건조에 더 직접 연관
"""
import numpy as np

from . import config


def compute_completeness(min_ratio, closed_ratio):
    """폐안 floor 로 정규화한 완성도. closed_ratio=0 이면 진폭과 동일."""
    denom = max(1.0 - float(closed_ratio), 1e-6)
    return float(np.clip((1.0 - float(min_ratio)) / denom, 0.0, 1.5))


def is_incomplete(completeness, thresh=config.COMPLETE_THRESHOLD):
    return bool(completeness < thresh)


def extract(times, ear_ratio, closed_ratio=0.0):
    """깜빡임별 특징 벡터(FEATURE_LEN,) 계산.

    Args:
        times:       프레임 타임스탬프(초), 단조 증가 1-D.
        ear_ratio:   같은 프레임들의 EAR / baseline. onset 직전~offset 직후.
        closed_ratio: 폐안 floor / baseline (캘리브에서). 완성도 정규화에 사용.
    """
    times = np.asarray(times, dtype=np.float64)
    e = np.asarray(ear_ratio, dtype=np.float64)
    n = len(e)
    if n < 3:
        return np.zeros(config.FEATURE_LEN, dtype=np.float32)

    dt = np.diff(times)
    dt[dt <= 0] = 1e-3
    vel = np.diff(e) / dt

    i_min = int(np.argmin(e))
    min_ratio = float(e[i_min])
    completeness = compute_completeness(min_ratio, closed_ratio)

    dur_total = float(times[-1] - times[0])
    dur_closing = float(times[i_min] - times[0])
    dur_opening = float(times[-1] - times[i_min])

    amp = max(1.0 - min_ratio, 1e-6)
    valley_mask = e <= (min_ratio + 0.10 * amp)
    dur_valley = float(np.sum(dt[valley_mask[1:]])) if n > 1 else 0.0

    closing_vel = vel[:i_min] if i_min > 0 else vel[:1]
    opening_vel = vel[i_min:] if i_min < len(vel) else vel[-1:]
    peak_close_vel = float(np.abs(np.min(closing_vel))) if len(closing_vel) else 0.0
    peak_open_vel = float(np.max(opening_vel)) if len(opening_vel) else 0.0

    asym_dur = dur_closing / max(dur_opening, 1e-3)
    asym_vel = peak_close_vel / max(peak_open_vel, 1e-3)
    auc_deficit = float(np.sum((1.0 - e[1:]) * dt))

    feat = np.array([
        dur_total, dur_closing, dur_valley, dur_opening,
        completeness, min_ratio,
        peak_close_vel, peak_open_vel,
        asym_dur, asym_vel, auc_deficit,
    ], dtype=np.float32)
    return feat
