"""Pi 5 PMIC(`vcgencmd pmic_read_adc`) 전력 측정 유틸.

⚠️ **rail 이름/출력 포맷은 아직 실제 Pi5에서 검증되지 않았다.**
    아래 RAIL 목록·정규식은 jfikar/RPi5-power (커뮤니티 자료,
    https://github.com/jfikar/RPi5-power) 에서 관찰된 이름·포맷을 근거로
    작성했다. 실제 착수 시 `vcgencmd pmic_read_adc` 원문 1회와 반드시
    대조할 것 — 특히:
      1) rail 이름 철자가 실제와 같은지
      2) "<NAME> current(<idx>)=<value>A" / "<NAME> volt(<idx>)=<value>V"
         포맷(공백·소수점 자릿수)이 실제와 같은지
    다르면 `PMIC_LINE_RE` 정규식만 고치면 된다 — 이 파일의 나머지
    (전력 계산, 적분, idle/active 분리)는 파싱 결과 dict 에만 의존하므로
    영향받지 않는다.
"""

from __future__ import annotations

import re
import subprocess
import threading
import time

import numpy as np

# ------------------------------------------------------------------ rail 정의
# 전류(A) 12개 — 이 12개 쌍의 V*I 합이 순간 전력(W)이다.
CURRENT_RAILS: list[str] = [
    "3V7_WL_SW", "3V3_SYS", "1V8_SYS", "DDR_VDD2", "DDR_VDDQ",
    "1V1_SYS", "0V8_SW", "VDD_CORE", "3V3_DAC", "3V3_ADC",
    "0V8_AON", "HDMI",
]

# 전압만 있고 대응 전류가 없어 전력 합산에서 제외되는 rail.
#   EXT5V_V — 입력 전압 (5V 레일 자체의 전류가 아니라 상위 12개 rail 전류의
#             입력단 전압이라 이중계산 방지 위해 제외)
#   BATT_V  — RTC 코인셀 전압, 전류계 없음
VOLT_ONLY_RAILS: list[str] = ["EXT5V_V", "BATT_V"]

# "<NAME> current(<idx>)=<value>A" 또는 "<NAME> volt(<idx>)=<value>V"
# NAME 자체에 _A/_V 접미사가 이미 붙어 나온다 (jfikar/RPi5-power 관찰 포맷).
PMIC_LINE_RE = re.compile(
    r"([A-Za-z0-9_]+)\s+(?:current|volt)\(\d+\)\s*=\s*([-+]?[0-9]*\.?[0-9]+)\s*[AV]?",
    re.IGNORECASE,
)


class PmicParseError(ValueError):
    pass


def parse_pmic_adc(raw: str) -> dict[str, float]:
    """`vcgencmd pmic_read_adc` 원문 텍스트 -> {rail_name: value} dict.

    rail_name 은 이미 _A/_V 접미사를 포함한다 (예: "VDD_CORE_A": 1.30).
    한 줄도 못 읽으면 PmicParseError.
    """
    values: dict[str, float] = {}
    for line in raw.splitlines():
        m = PMIC_LINE_RE.search(line)
        if not m:
            continue
        name, val = m.group(1), m.group(2)
        try:
            values[name] = float(val)
        except ValueError:
            continue
    if not values:
        raise PmicParseError(
            f"pmic_read_adc 출력에서 한 줄도 파싱하지 못했다 (포맷 불일치 가능). "
            f"raw[:200]={raw[:200]!r}"
        )
    return values


def compute_instant_power_w(values: dict[str, float]) -> tuple[float, list[str]]:
    """12개 rail의 P = Σ(Vᵢ×Iᵢ) 계산. (전력 W, 누락된 rail 이름 목록) 반환.

    누락 rail은 그냥 합산에서 빠지고 이름만 보고된다 — 한두 rail 파싱
    실패로 전체 샘플을 버리면 5~10Hz 폴링에서 데이터 손실이 커진다.
    """
    total = 0.0
    missing: list[str] = []
    for rail in CURRENT_RAILS:
        a_key, v_key = f"{rail}_A", f"{rail}_V"
        if a_key in values and v_key in values:
            total += values[a_key] * values[v_key]
        else:
            missing.append(rail)
    return total, missing


def integrate_energy_j(samples: list[tuple[float, float]]) -> float:
    """(timestamp_s, watt) 샘플을 실제 타임스탬프 간격 기준 사다리꼴 적분 -> J.

    `1초 × N` 근사 금지 지침에 따라 항상 실측 dt를 쓴다.
    """
    if len(samples) < 2:
        return 0.0
    ts = np.asarray([s[0] for s in samples], dtype=np.float64)
    ws = np.asarray([s[1] for s in samples], dtype=np.float64)
    order = np.argsort(ts)
    ts, ws = ts[order], ws[order]
    # numpy>=2.0 에서 np.trapz 가 제거되고 np.trapezoid 로 이름이 바뀌었다.
    # 로컬(데스크톱)과 Pi 양쪽에서 numpy 버전이 다를 수 있어 둘 다 지원한다.
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz_fn(ws, ts))


def mean_power_in_window(
    samples: list[tuple[float, float]], t_start: float, t_end: float
) -> float | None:
    vals = [w for t, w in samples if t_start <= t <= t_end]
    return float(np.mean(vals)) if vals else None


def battery_life_hours(
    power_w: float | None,
    capacity_ah: float = 5.0,
    cell_v: float = 3.7,
    eta: float = 0.87,
) -> float | None:
    """battery_wh = capacity_ah * cell_v * eta ; hours = battery_wh / power_w.

    capacity_ah/cell_v 는 **예시값**이다 (요청자가 임의로 넣은 5000mAh/3.7V) —
    실제 타깃 기기 배터리 스펙이 아니므로 이 자체를 논문 수치로 쓰지 말 것.
    eta 는 배터리 셀 -> 5V 승압 컨버터 효율 가정치(0.85~0.90 범위 중 기본값
    0.87을 씀). 실제 컨버터 데이터시트가 있으면 그 값으로 교체할 것.
    """
    if not power_w or power_w <= 0:
        return None
    battery_wh = capacity_ah * cell_v * eta
    return battery_wh / power_w


class PowerSampler(threading.Thread):
    """`vcgencmd pmic_read_adc` 를 별도 스레드 · 고정 주기(기본 7.5Hz)로 폴링.

    기존 온도/CPU%/RSS `Sampler`(1Hz)와 절대 같은 스레드에 넣지 않는다 —
    폴링 주기가 다른 두 계측을 섞으면 `vcgencmd` 호출 자체의 부하가 측정에
    끼어들 때 어느 쪽 때문인지 구분이 안 된다. 폴링 주기는 4개 모드
    실행에서 반드시 동일하게 유지해야 한다 (CLI --power-hz 고정값 사용).
    """

    def __init__(self, hz: float = 7.5, cmd_fn=None):
        super().__init__(daemon=True)
        if hz <= 0:
            raise ValueError("hz must be > 0")
        self.interval = 1.0 / hz
        self.hz = hz
        self.stop_evt = threading.Event()
        self.samples: list[tuple[float, float]] = []  # (perf_counter, watt)
        self.parse_errors = 0
        self.missing_rails: set[str] = set()
        self.first_raw_sample: str | None = None
        self._cmd_fn = cmd_fn or self._read_pmic_adc_raw

    @staticmethod
    def _read_pmic_adc_raw() -> str | None:
        try:
            return subprocess.check_output(
                "vcgencmd pmic_read_adc",
                shell=True,
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        except Exception:
            return None

    def _sample_once(self) -> None:
        raw = self._cmd_fn()
        ts = time.perf_counter()
        if raw is None:
            self.parse_errors += 1
            return
        if self.first_raw_sample is None:
            self.first_raw_sample = raw
        try:
            values = parse_pmic_adc(raw)
            watts, missing = compute_instant_power_w(values)
        except PmicParseError:
            self.parse_errors += 1
            return
        if missing:
            self.missing_rails.update(missing)
        self.samples.append((ts, watts))

    def run(self) -> None:
        # 시작하자마자 1개는 즉시 뽑아 idle 구간이 짧아도 데이터가 있게 한다.
        self._sample_once()
        while not self.stop_evt.wait(self.interval):
            self._sample_once()

    def stop(self) -> None:
        self.stop_evt.set()
        self.join(timeout=3)