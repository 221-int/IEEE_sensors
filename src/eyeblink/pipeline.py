"""pipeline — 실시간 깜빡임 파이프라인 (EAR 레벨 글루).

프레임에서 EAR 을 뽑는 부분(랜드마크)은 상위 스크립트(scripts/run_live)가 맡고,
이 모듈은 EAR 스트림을 받아:
    개인 캘리브(개안/폐안) -> EMA 평활 -> STD 4단계 판정 -> 카운트/BPM 집계
를 수행한다. MediaPipe/OpenCV 없이 단위 테스트 가능.

사용:
    pipe = BlinkPipeline()
    for t, ear in stream:
        st = pipe.update(t, ear)
        if st["calibrating"]:
            ...  # st["phase"] ("open"/"closed"), st["progress"]
        else:
            ...  # st["state"], st["blink"](이번 프레임 확정 이벤트|None), st["metrics"]
"""
from . import config
from .landmarks import TwoPhaseCalibrator
from .std import BlinkSTD
from .metrics import BlinkCounter


class BlinkPipeline:
    def __init__(self, calibrator=None, window_sec=config.METRIC_WINDOW_SEC,
                 ema_alpha=config.EMA_ALPHA):
        self.calib = calibrator or TwoPhaseCalibrator()
        self.ema_alpha = ema_alpha
        self.counter = BlinkCounter(window_sec)
        self.std = None
        self._ema = None

    @property
    def calibrating(self):
        return not self.calib.done

    def _smooth(self, ear):
        if self._ema is None:
            self._ema = ear
        else:
            self._ema = self.ema_alpha * ear + (1 - self.ema_alpha) * self._ema
        return self._ema

    def update(self, t, ear):
        # 1) 캘리브 단계(개안 baseline -> 폐안 floor)
        if not self.calib.done:
            prog = self.calib.update(ear)
            return {"calibrating": True, "phase": self.calib.phase,
                    "progress": prog, "ear": ear}

        # 2) 캘리브 완료 -> STD 초기화(개안/폐안 임계값 확정)
        if self.std is None:
            self.std = BlinkSTD(self.calib.baseline, self.calib.closed_floor)

        # 3) EMA 평활 -> STD 판정 -> 카운트
        s = self._smooth(ear)
        ev = self.std.update(t, s)
        if ev is not None:
            self.counter.record(ev)
        return {"calibrating": False, "state": self.std.state.value,
                "ear": ear, "ear_smooth": s, "blink": ev,
                "metrics": self.counter.metrics(t)}
