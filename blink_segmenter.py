"""
blink_segmenter.py — EAR 샘플 스트림에서 깜빡임 이벤트를 검출하고
깜빡임별 운동학 특징 벡터를 내보낸다.

스트리밍 상태 머신 (이전 프로젝트의 4단계 아이디어 재사용):
    OPEN -> CLOSING -> CLOSED(valley) -> OPENING -> OPEN
임계값 아래로 BLINK_MIN_FRAMES 이상 내려갔다가 다시 위로 올라오면 깜빡임
이벤트를 확정한다. 이벤트 구간(앞뒤 약간의 마진 포함)의 원시
(time, ear_ratio) 샘플을 features.extract 로 넘긴다.

사용 (라이브):
    seg = BlinkSegmenter(baseline, threshold)
    for t, ear in stream:
        ev = seg.update(t, ear)        # BlinkEvent 또는 None 반환
        if ev: ...                     # ev.features 가 분류 준비됨
"""
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import config
import features as F


@dataclass
class BlinkEvent:
    t_start: float
    t_end: float
    times: np.ndarray
    ear_ratio: np.ndarray
    features: np.ndarray = field(default=None)
    label: int = -1            # 라벨링 수집 중에 설정됨

    @property
    def duration(self):
        return self.t_end - self.t_start


class BlinkSegmenter:
    def __init__(self, baseline, threshold,
                 margin=4, min_frames=config.BLINK_MIN_FRAMES,
                 max_frames=config.BLINK_MAX_FRAMES):
        self.baseline = baseline
        self.threshold = threshold        # 절대 EAR 임계값
        self.margin = margin              # 감김 구간 앞뒤로 보관할 프레임 수
        self.min_frames = min_frames
        self.max_frames = max_frames
        self._ring_t = deque(maxlen=margin)
        self._ring_e = deque(maxlen=margin)
        self._in_blink = False
        self._buf_t = []
        self._buf_e = []
        self._closed_count = 0

    def update(self, t, ear):
        """(타임스탬프, EAR) 샘플 하나를 넣는다. BlinkEvent 또는 None 반환."""
        ratio = ear / max(self.baseline, 1e-6)
        closed = ear < self.threshold

        if not self._in_blink:
            if closed:
                # 시작: 깔끔한 운동학을 위해 onset 직전 마진으로 시드
                self._in_blink = True
                self._buf_t = list(self._ring_t) + [t]
                self._buf_e = list(self._ring_e) + [ratio]
                self._closed_count = 1
            else:
                self._ring_t.append(t)
                self._ring_e.append(ratio)
            return None

        # 깜빡임 진행 중
        self._buf_t.append(t)
        self._buf_e.append(ratio)
        if closed:
            self._closed_count += 1
            if self._closed_count > self.max_frames:    # 깜빡임 아님; 리셋
                self._reset()
            return None

        # 눈이 다시 떠짐 -> 감김이 충분히 길었으면 확정
        if self._closed_count >= self.min_frames:
            ev = self._finalize()
            self._reset(seed_t=t, seed_e=ratio)
            return ev
        self._reset(seed_t=t, seed_e=ratio)
        return None

    def _finalize(self):
        times = np.asarray(self._buf_t, dtype=np.float64)
        ear = np.asarray(self._buf_e, dtype=np.float64)
        ev = BlinkEvent(t_start=float(times[0]), t_end=float(times[-1]),
                        times=times, ear_ratio=ear)
        ev.features = F.extract(times, ear)
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
