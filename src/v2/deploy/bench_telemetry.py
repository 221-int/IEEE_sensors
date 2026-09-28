"""Pi 5 telemetry for paced runs: thermal/throttle/clock/fan and raw PMIC logging.

Commands use argument lists (no shell). Off-Pi every reader returns None, and
the result QC marks the run as not reportable.

PMIC power is the sum of V*I over the 12 current rails of `pmic_power.py`.
This rail-sum definition is not verified against board input power; results
call it `pmic_rail_sum_w` and never "board power".
"""
from __future__ import annotations

import glob
import gzip
import json
import os
import subprocess
import threading
import time

import numpy as np

from src.v2.deploy.pmic_power import (
    PmicParseError,
    PowerSampler,
    compute_instant_power_w,
    parse_pmic_adc,
)

NOW_BITS = {0: "under-voltage", 1: "arm-freq-capped", 2: "throttled", 3: "soft-temp-limit"}
STICKY_BITS = {16: "under-voltage occurred", 17: "arm-freq-capped occurred",
               18: "throttling occurred", 19: "soft-temp-limit occurred"}


def run_cmd(args: list[str], timeout: float = 5.0) -> str | None:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout,
                              check=True).stdout.strip()
    except Exception:
        return None


def read_file(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def read_temp_c() -> float | None:
    out = run_cmd(["vcgencmd", "measure_temp"])            # temp=53.6'C
    try:
        return float(out.split("=")[1].split("'")[0]) if out else None
    except (IndexError, ValueError):
        return None


def read_throttled() -> int | None:
    out = run_cmd(["vcgencmd", "get_throttled"])           # throttled=0x0
    try:
        return int(out.split("=")[1], 16) if out else None
    except (IndexError, ValueError):
        return None


def read_arm_hz() -> int | None:
    out = run_cmd(["vcgencmd", "measure_clock", "arm"])    # frequency(0)=2400020480
    try:
        return int(out.split("=")[1]) if out else None
    except (IndexError, ValueError):
        return None


def decode_throttle(v: int | None) -> dict:
    if v is None:
        return {"raw": None, "now": [], "sticky": []}
    return {"raw": hex(v), "now": [n for b, n in NOW_BITS.items() if v & (1 << b)],
            "sticky": [n for b, n in STICKY_BITS.items() if v & (1 << b)]}


def read_governors() -> dict[str, str | None]:
    out = {}
    for p in sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor")):
        out[p.split("/")[5]] = read_file(p)
    return out


def read_fan() -> dict:
    """Pi 5 active cooler: hwmon fan1_input (rpm) / pwm1, if exposed."""
    rpm = pwm = None
    for d in glob.glob("/sys/class/hwmon/hwmon*"):
        if os.path.exists(os.path.join(d, "fan1_input")):
            v = read_file(os.path.join(d, "fan1_input"))
            rpm = int(v) if v and v.isdigit() else None
            v = read_file(os.path.join(d, "pwm1"))
            pwm = int(v) if v and v.isdigit() else None
            break
    return {"rpm": rpm, "pwm": pwm}


def read_proc_rss_mb() -> float | None:
    s = read_file("/proc/self/status")
    if s:
        for line in s.splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return None


def platform_snapshot() -> dict:
    return {
        "temp_c": read_temp_c(), "throttled": decode_throttle(read_throttled()),
        "arm_hz": read_arm_hz(), "governors": read_governors(), "fan": read_fan(),
        "model": read_file("/proc/device-tree/model"),
        "cmdline_has_vcgencmd": run_cmd(["vcgencmd", "version"]) is not None,
    }


class ThermalSampler(threading.Thread):
    """1 Hz temperature / throttle word / ARM clock / fan / RSS."""

    def __init__(self, hz: float = 1.0, readers: dict | None = None):
        super().__init__(daemon=True)
        self.interval = 1.0 / hz
        self.stop_evt = threading.Event()
        self.samples: list[dict] = []
        r = readers or {}
        self._temp = r.get("temp", read_temp_c)
        self._thr = r.get("throttled", read_throttled)
        self._arm = r.get("arm", read_arm_hz)
        self._fan = r.get("fan", read_fan)
        self._rss = r.get("rss", read_proc_rss_mb)

    def sample_once(self) -> None:
        self.samples.append({"t": time.perf_counter(), "temp_c": self._temp(),
                             "throttled": self._thr(), "arm_hz": self._arm(),
                             "fan": self._fan(), "rss_mb": self._rss()})

    def run(self) -> None:
        self.sample_once()
        while not self.stop_evt.wait(self.interval):
            self.sample_once()

    def stop(self) -> None:
        self.stop_evt.set()
        self.join(timeout=3)

    def window(self, t0: float, t1: float) -> dict:
        s = [x for x in self.samples if t0 <= x["t"] <= t1]
        temps = [x["temp_c"] for x in s if x["temp_c"] is not None]
        arm = [x["arm_hz"] for x in s if x["arm_hz"] is not None]
        rpm = [x["fan"]["rpm"] for x in s if x["fan"] and x["fan"].get("rpm") is not None]
        words = [x["throttled"] for x in s if x["throttled"] is not None]
        now_flags = sorted({f for w in words for f in decode_throttle(w)["now"]})
        return {
            "n": len(s),
            "temp_c": {"first": temps[0], "last": temps[-1], "max": max(temps),
                       "mean": float(np.mean(temps))} if temps else None,
            "arm_hz": {"min": min(arm), "max": max(arm)} if arm else None,
            "fan_rpm": {"min": min(rpm), "max": max(rpm), "mean": float(np.mean(rpm))} if rpm else None,
            "throttle_now_flags": now_flags,
            "rss_mb_max": max((x["rss_mb"] for x in s if x["rss_mb"] is not None), default=None),
        }


def _pmic_raw_cmd() -> str | None:
    out = run_cmd(["vcgencmd", "pmic_read_adc"])
    return out if out else None


class LoggingPowerSampler(PowerSampler):
    """PowerSampler that also keeps every raw PMIC text for later rail audits."""

    def __init__(self, hz: float = 7.5, cmd_fn=None):
        super().__init__(hz=hz, cmd_fn=cmd_fn or _pmic_raw_cmd)
        self.raw: list[tuple[float, float, str | None]] = []    # (perf, wall, text)
        self.ext5v: list[tuple[float, float]] = []

    def _sample_once(self) -> None:
        raw = self._cmd_fn()
        ts, wall = time.perf_counter(), time.time()
        self.raw.append((ts, wall, raw))
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
        if "EXT5V_V" in values:
            self.ext5v.append((ts, values["EXT5V_V"]))

    def write_raw(self, path: str) -> None:
        with gzip.open(path, "wt", encoding="utf-8") as f:
            for ts, wall, raw in self.raw:
                f.write(json.dumps({"t_perf": ts, "t_wall": wall, "raw": raw}) + "\n")

    def window(self, t0: float, t1: float) -> dict:
        s = [(t, w) for t, w in self.samples if t0 <= t <= t1]
        dur = t1 - t0
        if not s:
            return {"n": 0, "duration_s": dur}
        ts = np.array([t for t, _ in s])
        ws = np.array([w for _, w in s])
        gaps = np.diff(np.concatenate([[t0], ts, [t1]]))
        ext = [v for t, v in self.ext5v if t0 <= t <= t1]
        return {"n": int(ws.size), "duration_s": dur, "mean_w": float(ws.mean()),
                "sd_w": float(ws.std(ddof=1)) if ws.size > 1 else 0.0,
                "effective_hz": ws.size / dur if dur > 0 else None,
                "max_gap_s": float(gaps.max()),
                "energy_j_mean_x_duration": float(ws.mean() * dur),
                "ext5v_v_mean": float(np.mean(ext)) if ext else None}
