"""metrics — 깜빡임 카운트 + 분당 깜빡임률(BPM) 롤링 집계.

STD 가 낸 BlinkEvent 를 record() 로 넣으면 누적 카운트와 최근 창(window_sec)
기준 BPM 을 준다. 완성도(불완전 깜빡임 비율)는 선택 지표로 함께 제공.
"""
from collections import deque

from . import config


class BlinkCounter:
    def __init__(self, window_sec=config.METRIC_WINDOW_SEC):
        self.window_sec = window_sec
        self._events = deque()       # (t_end, completeness, incomplete)
        self.total = 0
        self.total_incomplete = 0
        self._t0 = None

    def record(self, event):
        if self._t0 is None:
            self._t0 = event.t_start
        self.total += 1
        inc = bool(getattr(event, "incomplete", False))
        self.total_incomplete += int(inc)
        self._events.append((event.t_end, event.completeness, inc))
        self._evict(event.t_end)

    def _evict(self, now):
        while self._events and now - self._events[0][0] > self.window_sec:
            self._events.popleft()

    def metrics(self, now):
        """현재 시각 기준 지표 dict."""
        self._evict(now)
        n = len(self._events)
        span_min = self.window_sec / 60.0
        elapsed_min = ((now - self._t0) / 60.0) if self._t0 else 0.0
        return {
            "blink_rate_bpm": (n / span_min) if span_min > 0 else 0.0,   # 최근 창 기준
            "cumulative_bpm": (self.total / elapsed_min) if elapsed_min > 0 else 0.0,
            "window_blinks": n,
            "total_blinks": self.total,
            "incomplete_blink_rate": (self.total_incomplete / self.total) if self.total else 0.0,
        }
