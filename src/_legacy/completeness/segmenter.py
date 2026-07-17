"""segmenter — EAR 스트림에서 깜빡임 이벤트를 검출하고 특징 벡터를 낸다.

스트리밍 상태머신: OPEN -> CLOSING -> CLOSED(valley) -> OPENING -> OPEN
임계값 아래로 BLINK_MIN_FRAMES 이상 내려갔다가 다시 위로 올라오면 깜빡임을
확정한다. 이벤트 구간(앞뒤 마진 포함)의 (time, ear_ratio)를 features.extract 로
넘긴다. 배치가 아니라 프레임 스트림에서 바로 동작(실시간).

사용:
    seg = BlinkSegmenter(baseline, threshold, closed_ratio)
    for t, ear in stream:
        ev = seg.update(t, ear)     # BlinkEvent 또는 None
"""
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from . import config
from . import features as F


@dataclass
class BlinkEvent:
    t_start: float
    t_end: float
    times: np.ndarray
    ear_ratio: np.ndarray
    features: np.ndarray = field(default=None)
    completeness: float = 0.0
    incomplete: bool = False
    label: int = -1                 # 라벨 수집 중에 설정

    @property
    def duration(self):
        return self.t_end - self.t_start


class BlinkSegmenter:
    def __init__(self, baseline, threshold, closed_ratio=0.0,
                 margin=4, min_frames=config.BLINK_MIN_FRAMES,
                 max_frames=config.BLINK_MAX_FRAMES):
        self.baseline = baseline
        self.threshold = threshold          # 절대 EAR 임계값
        self.closed_ratio = closed_ratio    # 폐안 floor / baseline
        self.margin = margin
        self.min_frames = min_frames
        self.max_frames = max_frames
        self._ring_t = deque(maxlen=margin)
        self._ring_e = deque(maxlen=margin)
        self._in_blink = False
        self._buf_t, self._buf_e = [], []
        self._closed_count = 0

    def update(self, t, ear):
        """(타임스탬프, EAR) 하나를 넣는다. BlinkEvent 또는 None."""
        ratio = ear / max(self.baseline, 1e-6)
        closed = ear < self.threshold

        if not self._in_blink:
            if closed:
                self._in_blink = True
                self._buf_t = list(self._ring_t) + [t]
                self._buf_e = list(self._ring_e) + [ratio]
                self._closed_count = 1
            else:
                self._ring_t.append(t)
                self._ring_e.append(ratio)
            return None

        self._buf_t.append(t)
        self._buf_e.append(ratio)
        if closed:
            self._closed_count += 1
            if self._closed_count > self.max_frames:   # 깜빡임 아님; 리셋
                self._reset()
            return None

        if self._closed_count >= self.min_frames:
            ev = self._finalize()
            self._reset(seed_t=t, seed_e=ratio)
            return ev
        self._reset(seed_t=t, seed_e=ratio)
        return None

    def _finalize(self):
        times = np.asarray(self._buf_t, dtype=np.float64)
        ear = np.asarray(self._buf_e, dtype=np.float64)
        feat = F.extract(times, ear, self.closed_ratio)
        completeness = float(feat[config.FEATURE_NAMES.index("completeness")])
        ev = BlinkEvent(t_start=float(times[0]), t_end=float(times[-1]),
                        times=times, ear_ratio=ear, features=feat,
                        completeness=completeness,
                        incomplete=F.is_incomplete(completeness))
        return ev

    def _reset(self, seed_t=None, seed_e=None):
        self._in_blink = False
        self._buf_t, self._buf_e = [], []
        self._closed_count = 0
        self._ring_t.clear()
        self._ring_e.clear()
        if seed_t is not None:
            self._ring_t.append(seed_t)
            self._ring_e.append(seed_e)
