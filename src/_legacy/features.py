"""
features.py — 깜빡임별 운동학(kinematics) 특징 벡터.

이 프로젝트의 핵심 기여: 깜빡임을 이진 이벤트로 보지 않고, 각 깜빡임을
*운동학*으로 기술한다. 의도(voluntary)와 무의식(spontaneous) 깜빡임은
운동학이 다르다:

  - 의도 깜빡임: 더 길고, 더 완전히 감기고, 더 느리게 닫힘
  - 무의식 깜빡임: 더 짧고, 불완전한 경우가 많고, 더 빠름

깜빡임은 (시간, ear_ratio) 샘플의 시계열로 표현한다. 여기서
ear_ratio = EAR / 눈뜸 baseline (즉 1.0 ~ 뜸, ~0 ~ 완전히 감김).
"""
import numpy as np

import config


def extract(times, ear_ratio):
    """깜빡임별 특징 벡터 계산.

    Args:
        times:      프레임 타임스탬프(초), 단조 증가하는 1-D 배열.
        ear_ratio:  같은 프레임들의 EAR / baseline 값. onset 직전부터
                    offset 직후까지 한 번의 깜빡임을 포함.

    Returns:
        config.FEATURE_NAMES 순서의 (FEATURE_LEN,) np.ndarray.
    """
    times = np.asarray(times, dtype=np.float64)
    e = np.asarray(ear_ratio, dtype=np.float64)
    n = len(e)
    if n < 3:
        return np.zeros(config.FEATURE_LEN, dtype=np.float32)

    dt = np.diff(times)
    dt[dt <= 0] = 1e-3
    vel = np.diff(e) / dt                      # 스텝별 d(ear_ratio)/dt

    i_min = int(np.argmin(e))
    min_ratio = float(e[i_min])
    amplitude = 1.0 - min_ratio                # 눈뜸 baseline 대비 완성도

    dur_total = float(times[-1] - times[0])

    # Onset = 최저점까지의 닫힘 구간; offset = 그 이후 열림 구간.
    dur_closing = float(times[i_min] - times[0])
    dur_opening = float(times[-1] - times[i_min])

    # Valley = 최저점의 10% 이내 프레임("감은 채 유지"되는 평탄 구간).
    valley_mask = e <= (min_ratio + 0.10 * max(amplitude, 1e-6))
    dur_valley = float(np.sum(dt[valley_mask[1:]])) if n > 1 else 0.0

    closing_vel = vel[:i_min] if i_min > 0 else vel[:1]
    opening_vel = vel[i_min:] if i_min < len(vel) else vel[-1:]
    peak_close_vel = float(np.abs(np.min(closing_vel))) if len(closing_vel) else 0.0
    peak_open_vel = float(np.max(opening_vel)) if len(opening_vel) else 0.0

    asym_dur = dur_closing / max(dur_opening, 1e-3)
    asym_vel = peak_close_vel / max(peak_open_vel, 1e-3)

    # 눈뜸 baseline(1.0)과 EAR 곡선 사이의 면적(깜빡임 구간 전체).
    auc_deficit = float(np.sum((1.0 - e[1:]) * dt))

    feat = np.array([
        dur_total, dur_closing, dur_valley, dur_opening,
        amplitude, min_ratio,
        peak_close_vel, peak_open_vel,
        asym_dur, asym_vel, auc_deficit,
    ], dtype=np.float32)
    return feat
