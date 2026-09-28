"""Pi environment record before paced measurements (read-only; changes nothing).

    python -m src.v2.deploy.pi_env_check --out results/pi_paced/env_check.json

Records package versions, board model, governors, clocks, temperature,
throttle word, fan, and three raw `vcgencmd pmic_read_adc` texts with their
parsed rail sums, so the rail list can be audited against the real output.
Exit code 1 if a precondition of the protocol is not met.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

from src.v2.deploy import bench_telemetry as BT
from src.v2.deploy.pmic_power import CURRENT_RAILS, PmicParseError, compute_instant_power_w, parse_pmic_adc


def versions() -> dict:
    out = {"python": platform.python_version(), "executable": sys.executable}
    for mod in ("numpy", "onnxruntime", "cv2", "mediapipe", "psutil"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception as e:
            out[mod] = f"unavailable: {e.__class__.__name__}"
    return out


def pmic_audit(n: int = 3) -> list[dict]:
    rows = []
    for _ in range(n):
        raw = BT.run_cmd(["vcgencmd", "pmic_read_adc"])
        row = {"raw": raw}
        if raw:
            try:
                vals = parse_pmic_adc(raw)
                w, missing = compute_instant_power_w(vals)
                extra = sorted({k.rsplit("_", 1)[0] for k in vals} - set(CURRENT_RAILS) - {"EXT5V", "BATT"})
                row.update(rail_sum_w=w, missing_rails=missing, unlisted_rails=extra,
                           ext5v_v=vals.get("EXT5V_V"), n_values=len(vals))
            except PmicParseError as e:
                row["error"] = str(e)
        rows.append(row)
        time.sleep(0.5)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/pi_paced/env_check.json"))
    a = ap.parse_args(argv)
    snap = BT.platform_snapshot()
    rec = {"created": datetime.now().astimezone().isoformat(), "machine": platform.machine(),
           "platform": platform.platform(), "versions": versions(), "snapshot": snap,
           "os_release": BT.read_file("/etc/os-release"),
           "kernel_cmdline": BT.read_file("/proc/cmdline"),
           "vcgencmd_version": BT.run_cmd(["vcgencmd", "version"]),
           "config_arm_freq": BT.run_cmd(["vcgencmd", "get_config", "arm_freq"]),
           "pmic_audit": pmic_audit()}
    problems = []
    if rec["machine"] != "aarch64" or not snap["cmdline_has_vcgencmd"]:
        problems.append("not a Pi (aarch64 + vcgencmd)")
    if not snap["governors"] or any(v != "performance" for v in snap["governors"].values()):
        problems.append(f"governor: {snap['governors']}")
    if snap["throttled"]["raw"] != "0x0":
        problems.append(f"throttled={snap['throttled']['raw']}")
    for i, row in enumerate(rec["pmic_audit"]):
        if "rail_sum_w" not in row or row.get("missing_rails"):
            problems.append(f"pmic sample {i}: {row.get('error') or row.get('missing_rails') or 'no output'}")
    expected = {"numpy": "1.26.4", "onnxruntime": "1.19.2", "mediapipe": "0.10.18", "cv2": "4.11.0"}
    rec["version_differences_from_requirements_pi"] = {k: rec["versions"][k] for k, v in expected.items()
                                                       if rec["versions"][k] != v}
    rec["problems"] = problems
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf8")
    print(json.dumps({k: rec[k] for k in ("machine", "versions", "problems",
                                          "version_differences_from_requirements_pi")}, indent=1))
    print("->", a.out)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
