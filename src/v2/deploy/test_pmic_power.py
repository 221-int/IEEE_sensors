"""pmic_power.py 단위테스트 — 실물 Pi5 없이 고정 fixture 텍스트로 검증한다.

⚠️ fixture 텍스트(FIXTURE_RAW / FIXTURE_VALUES)는 jfikar/RPi5-power 커뮤니티
자료에서 관찰된 포맷을 근거로 만든 **가정치**이며, 실제 `vcgencmd
pmic_read_adc` 출력과 대조되지 않았다. 착수 시 실제 Pi5 원문 1회와 이
fixture의 포맷(줄 구조, rail 이름 철자)을 대조할 것.

실행:
    python -m pytest src/v2/deploy/test_pmic_power.py -v
    또는
    python -m unittest src.v2.deploy.test_pmic_power -v
"""

from __future__ import annotations

import unittest

from src.v2.deploy.pmic_power import (
    CURRENT_RAILS,
    PmicParseError,
    battery_life_hours,
    compute_instant_power_w,
    integrate_energy_j,
    mean_power_in_window,
    parse_pmic_adc,
)

# 12개 전류 rail + 12개 전압 rail + EXT5V_V/BATT_V(전류 없음) = 26줄.
# 값은 임의로 지어낸 그럴듯한 수치 (전형적 idle-근처 Pi5 전력대).
FIXTURE_VALUES: dict[str, float] = {
    "0V8_AON_A": 0.0304, "0V8_AON_V": 0.8016,
    "0V8_SW_A": 0.1300, "0V8_SW_V": 0.7986,
    "1V1_SYS_A": 0.1791, "1V1_SYS_V": 1.1000,
    "1V8_SYS_A": 0.1195, "1V8_SYS_V": 1.8009,
    "3V3_ADC_A": 0.0018, "3V3_ADC_V": 3.3113,
    "3V3_DAC_A": 0.0009, "3V3_DAC_V": 3.3113,
    "3V3_SYS_A": 0.2557, "3V3_SYS_V": 3.3199,
    "3V7_WL_SW_A": 0.0002, "3V7_WL_SW_V": 3.6994,
    "DDR_VDD2_A": 0.2695, "DDR_VDD2_V": 1.1000,
    "DDR_VDDQ_A": 0.4090, "DDR_VDDQ_V": 0.5498,
    "HDMI_A": 0.0003, "HDMI_V": 5.0490,
    "VDD_CORE_A": 1.2969, "VDD_CORE_V": 0.8500,
    "EXT5V_V": 5.1000,
    "BATT_V": 3.0000,
}


def _fixture_text() -> str:
    lines = []
    idx = 0
    for name, val in FIXTURE_VALUES.items():
        kind = "volt" if name.endswith("_V") else "current"
        unit = "V" if name.endswith("_V") else "A"
        lines.append(f"{name} {kind}({idx})={val:.4f}{unit}")
        idx += 1
    return "\n".join(lines) + "\n"


FIXTURE_RAW = _fixture_text()


class TestParsePmicAdc(unittest.TestCase):
    def test_parses_all_26_lines(self):
        values = parse_pmic_adc(FIXTURE_RAW)
        self.assertEqual(len(values), 26)
        for name, expected in FIXTURE_VALUES.items():
            self.assertAlmostEqual(values[name], expected, places=4)

    def test_all_12_current_rails_present(self):
        values = parse_pmic_adc(FIXTURE_RAW)
        for rail in CURRENT_RAILS:
            self.assertIn(f"{rail}_A", values)
            self.assertIn(f"{rail}_V", values)

    def test_empty_input_raises(self):
        with self.assertRaises(PmicParseError):
            parse_pmic_adc("")

    def test_garbage_input_raises(self):
        with self.assertRaises(PmicParseError):
            parse_pmic_adc("this is not pmic output at all\nno numbers here\n")

    def test_extra_whitespace_tolerated(self):
        raw = "VDD_CORE_A   current(0)  =  1.2969 A\nVDD_CORE_V current(1)=0.8500V\n"
        values = parse_pmic_adc(raw)
        self.assertAlmostEqual(values["VDD_CORE_A"], 1.2969, places=4)


class TestComputeInstantPowerW(unittest.TestCase):
    def test_matches_manual_sum(self):
        values = parse_pmic_adc(FIXTURE_RAW)
        watts, missing = compute_instant_power_w(values)
        expected = sum(
            FIXTURE_VALUES[f"{r}_A"] * FIXTURE_VALUES[f"{r}_V"] for r in CURRENT_RAILS
        )
        self.assertEqual(missing, [])
        self.assertAlmostEqual(watts, expected, places=6)
        # 대략적인 sanity range (Pi5 idle~light-load 전형적인 대역)
        self.assertGreater(watts, 0.5)
        self.assertLess(watts, 15.0)

    def test_excludes_volt_only_rails(self):
        """EXT5V_V/BATT_V 는 대응 전류가 없으므로 값이 커도 합산에 영향 없어야 함."""
        values = parse_pmic_adc(FIXTURE_RAW)
        watts_before, _ = compute_instant_power_w(values)
        values2 = dict(values)
        values2["EXT5V_V"] = 999.0
        values2["BATT_V"] = 999.0
        watts_after, _ = compute_instant_power_w(values2)
        self.assertAlmostEqual(watts_before, watts_after, places=9)

    def test_missing_rail_reported_not_crashed(self):
        values = parse_pmic_adc(FIXTURE_RAW)
        del values["HDMI_A"]  # HDMI rail 전류 누락 시뮬레이션
        watts, missing = compute_instant_power_w(values)
        self.assertIn("HDMI", missing)
        expected = sum(
            FIXTURE_VALUES[f"{r}_A"] * FIXTURE_VALUES[f"{r}_V"]
            for r in CURRENT_RAILS
            if r != "HDMI"
        )
        self.assertAlmostEqual(watts, expected, places=6)


class TestIntegrateEnergyJ(unittest.TestCase):
    def test_constant_power_matches_p_times_t(self):
        # 5W 로 10초 -> 50 J
        samples = [(0.0, 5.0), (5.0, 5.0), (10.0, 5.0)]
        self.assertAlmostEqual(integrate_energy_j(samples), 50.0, places=6)

    def test_ramp_matches_trapezoid_formula(self):
        # 0 -> 10s 동안 0W -> 10W 선형 증가: 삼각형 넓이 = 0.5*10*10 = 50 J
        samples = [(0.0, 0.0), (10.0, 10.0)]
        self.assertAlmostEqual(integrate_energy_j(samples), 50.0, places=6)

    def test_single_or_empty_sample_returns_zero(self):
        self.assertEqual(integrate_energy_j([]), 0.0)
        self.assertEqual(integrate_energy_j([(0.0, 5.0)]), 0.0)

    def test_unordered_timestamps_handled(self):
        samples = [(10.0, 5.0), (0.0, 5.0), (5.0, 5.0)]
        self.assertAlmostEqual(integrate_energy_j(samples), 50.0, places=6)


class TestMeanPowerInWindow(unittest.TestCase):
    def test_filters_to_window(self):
        samples = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (5.0, 100.0)]
        m = mean_power_in_window(samples, 0.0, 2.0)
        self.assertAlmostEqual(m, 2.0, places=6)

    def test_empty_window_returns_none(self):
        samples = [(0.0, 1.0)]
        self.assertIsNone(mean_power_in_window(samples, 10.0, 20.0))


class TestBatteryLifeHours(unittest.TestCase):
    def test_known_values(self):
        # 5Ah * 3.7V * 0.87 = 16.095 Wh ; @1W -> 16.095h
        h = battery_life_hours(1.0, capacity_ah=5.0, cell_v=3.7, eta=0.87)
        self.assertAlmostEqual(h, 16.095, places=3)

    def test_zero_or_none_power_returns_none(self):
        self.assertIsNone(battery_life_hours(0.0))
        self.assertIsNone(battery_life_hours(None))
        self.assertIsNone(battery_life_hours(-1.0))


if __name__ == "__main__":
    unittest.main()