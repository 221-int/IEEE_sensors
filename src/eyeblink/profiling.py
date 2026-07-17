"""profiling — fps / 처리시간 계측 유틸 (경량, 표준 라이브러리만).

  FpsMeter   : 최근 창(window) 프레임 기준 롤링 fps.
  StageTimer : 단계별(landmark/ear/std) 소요시간(ms) 누적 -> mean/p95 요약.

엣지 실현성(예: FaceLandmarker 가 Pi5 에서 몇 fps 나오나) 확인·논문 수치용.
"""
import time
from collections import deque


class FpsMeter:
    def __init__(self, window=30):
        self._t = deque(maxlen=window)

    def tick(self, now=None):
        self._t.append(time.perf_counter() if now is None else now)

    @property
    def fps(self):
        if len(self._t) < 2:
            return 0.0
        span = self._t[-1] - self._t[0]
        return (len(self._t) - 1) / span if span > 0 else 0.0


class StageTimer:
    """단계별 소요시간(ms) 누적 통계."""

    def __init__(self):
        self._s = {}   # name -> list[ms]

    def add(self, name, ms):
        self._s.setdefault(name, []).append(ms)

    def timed(self, name):
        return _Timed(self, name)

    def summary(self):
        out = {}
        for k, v in self._s.items():
            if not v:
                continue
            sv = sorted(v)
            out[k] = {
                "mean_ms": sum(v) / len(v),
                "p95_ms": sv[min(len(sv) - 1, int(0.95 * len(sv)))],
                "max_fps": (1000.0 / (sum(v) / len(v))) if sum(v) > 0 else 0.0,
                "n": len(v),
            }
        return out


class _Timed:
    def __init__(self, timer, name):
        self.timer = timer
        self.name = name

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.timer.add(self.name, (time.perf_counter() - self._t0) * 1e3)
        return False
