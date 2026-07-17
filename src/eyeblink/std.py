"""std — 상태 천이도(State Transition Diagram) 기반 깜빡임 판정.

4단계 상태:  A(OPEN) -> B(CLOSING) -> C(CLOSED) -> D(OPENING) -> A
한 번의 완전한 순환(A->B->C->D->A)이 일어났을 때만 깜빡임 1회로 판정한다.

  - 1초 이내 폐안은 정상 깜빡임으로 카운트(재개안 시 1회).
  - 실눈/부분 감김(C 에 못 미치고 되돌아옴)은 B->A 로 폐기 -> 오검출 방지.
  - STD_MAX_CLOSED_SEC 초과 폐안(졸음/내려봄)은 실격 -> 재개안해도 미카운트.
  - 경계 채터링은 개안/폐안 두 임계값(히스테리시스)으로 억제.

임계값은 개인 캘리브의 개안 baseline 과 폐안 floor 로부터 밴드 비율로 정한다.
MediaPipe/OpenCV 불필요 -> 정지 EAR 시퀀스로 단위 테스트 가능.
"""
from dataclasses import dataclass
from enum import Enum

from . import config


class State(Enum):
    OPEN = "A"       # 완전 개안
    CLOSING = "B"    # 감기는 중
    CLOSED = "C"     # 완전 폐안
    OPENING = "D"    # 뜨는 중


@dataclass
class BlinkEvent:
    t_start: float          # CLOSING 진입 시각
    t_end: float            # OPEN 복귀(1회 확정) 시각
    min_ear: float          # 순환 중 최소 EAR
    completeness: float     # (baseline-min_ear)/(baseline-closed_floor), [0,1]

    @property
    def duration(self):
        return self.t_end - self.t_start

    @property
    def incomplete(self):
        return self.completeness < config.COMPLETE_THRESHOLD


class BlinkSTD:
    """개안/폐안 임계값 기반 STD 상태머신. update(t, ear) 로 프레임 공급."""

    def __init__(self, baseline, closed_floor=0.0,
                 close_frac=config.STD_CLOSE_FRAC,
                 open_frac=config.STD_OPEN_FRAC,
                 max_closed_sec=config.STD_MAX_CLOSED_SEC):
        band = max(baseline - closed_floor, 1e-6)
        self.baseline = baseline
        self.closed_floor = closed_floor
        self.t_closed = closed_floor + close_frac * band   # 이 아래 = 폐안
        self.t_open = closed_floor + open_frac * band       # 이 위   = 개안
        self.max_closed_sec = max_closed_sec
        self.state = State.OPEN
        self._reset_cycle()

    def _reset_cycle(self):
        self._t_start = None
        self._min_ear = None
        self._t_closed_enter = None
        self._too_long = False           # 장시간 폐안(졸음) 실격 플래그

    def _track(self, t, ear):
        if self._t_start is None:
            self._t_start = t
        self._min_ear = ear if self._min_ear is None else min(self._min_ear, ear)

    def update(self, t, ear):
        """프레임(타임스탬프, EAR) 하나 처리. 1회 확정 시 BlinkEvent, 아니면 None."""
        s = self.state
        if s is State.OPEN:
            if ear < self.t_open:
                self.state = State.CLOSING
                self._reset_cycle()
                self._track(t, ear)

        elif s is State.CLOSING:
            self._track(t, ear)
            if ear < self.t_closed:
                self.state = State.CLOSED
                self._t_closed_enter = t
            elif ear >= self.t_open:
                self.state = State.OPEN          # 폐안 못 미치고 복귀 -> 폐기
                self._reset_cycle()

        elif s is State.CLOSED:
            self._track(t, ear)
            if ear > self.t_closed:
                self.state = State.OPENING
            elif (self._t_closed_enter is not None
                  and t - self._t_closed_enter > self.max_closed_sec):
                self._too_long = True            # 졸음/내려봄 -> 실격(폐안 유지)

        elif s is State.OPENING:
            self._track(t, ear)
            if ear >= self.t_open:
                too_long = self._too_long
                ev = None if too_long else self._finalize(t)
                self.state = State.OPEN
                self._reset_cycle()
                return ev
            elif ear < self.t_closed:
                self.state = State.CLOSED         # 되감김(채터링)
        return None

    def _finalize(self, t_end):
        band = max(self.baseline - self.closed_floor, 1e-6)
        completeness = (self.baseline - self._min_ear) / band
        completeness = float(min(max(completeness, 0.0), 1.0))
        return BlinkEvent(t_start=self._t_start, t_end=t_end,
                          min_ear=self._min_ear, completeness=completeness)
