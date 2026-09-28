"""QC and aggregation for paced Pi runs (`run_paced.py` output).

    python -m src.v2.deploy.check_paced_results --session-dir results/pi_paced/S1 \
        --protocol protocol.json

Per run: hard failures make a run non-reportable (off-Pi, governor, gates,
throttling, PMIC gaps/parse errors, clip mismatch, protocol mismatch). Warnings
do not exclude a run but must be reported. Aggregation uses reportable runs only,
lists every excluded run with its reason, and never drops a run for its value.
Differences are paired within randomised blocks against a reference mode.
The spread across a handful of repeats is descriptive, not a population CI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

REF_MODE = "ours_vpres"


def _get(d, path, default=None):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d or d[k] is None:
            return default
        d = d[k]
    return d


def qc_run(r: dict, protocol: dict | None = None, protocol_sha: str | None = None) -> dict:
    hard, warn = [], []
    cfg = r.get("config", {})
    if r.get("schema") != "paced_run_v1":
        hard.append(f"unexpected schema {r.get('schema')}")
    if _get(r, "env.machine") != "aarch64" or not _get(r, "platform_start.cmdline_has_vcgencmd", False):
        hard.append("not a Raspberry Pi run (no aarch64/vcgencmd): desktop dry run only")
    for when in ("platform_start", "platform_end"):
        gov = _get(r, f"{when}.governors", {}) or {}
        if not gov or any(v != "performance" for v in gov.values()):
            hard.append(f"CPU governor not 'performance' at {when}: {gov}")
    if not _get(r, "gates.bundle.passed", False):
        hard.append(f"bundle/device verification failed: {_get(r, 'gates.bundle.errors')}")
    if not _get(r, "gates.equivalence.authorized", False):
        hard.append("train/serve equivalence gate not authorized")
    if cfg.get("intra_threads") != 2 or not cfg.get("no_spin"):
        hard.append("ORT must use intra_threads=2 and no spinning")
    # throttling: NOW bits in any sample, or sticky bits appearing during the run
    for ph, w in (r.get("thermal") or {}).items():
        if w.get("throttle_now_flags"):
            hard.append(f"throttle flags during {ph}: {w['throttle_now_flags']}")
    s0 = set(_get(r, "platform_start.throttled.sticky", []) or [])
    s1 = set(_get(r, "platform_end.throttled.sticky", []) or [])
    if s1 - s0:
        hard.append(f"new sticky throttle bits during run: {sorted(s1 - s0)}")
    if s0:
        warn.append(f"sticky throttle bits already set at start (reboot for a clean state): {sorted(s0)}")
    if not r.get("thermal_samples"):
        hard.append("no thermal samples")
    # power sampling
    pw = r.get("power", {})
    if pw.get("parse_errors"):
        hard.append(f"PMIC read/parse errors: {pw['parse_errors']}")
    if pw.get("missing_rails"):
        hard.append(f"PMIC rails missing: {pw['missing_rails']}")
    hz = cfg.get("power_hz") or 0
    for ph in ("idle_pre", "active", "idle_post"):
        w = _get(pw, f"windows.{ph}") or {}
        if not w.get("n"):
            hard.append(f"no PMIC samples in {ph}")
            continue
        if hz and w.get("effective_hz", 0) < 0.5 * hz:
            hard.append(f"PMIC effective rate {w['effective_hz']:.2f} Hz in {ph} < half of {hz}")
        if hz and w.get("max_gap_s", 0) > max(1.0, 3.0 / hz):
            hard.append(f"PMIC gap {w['max_gap_s']:.2f} s in {ph}")
    drift, pre = pw.get("idle_drift_w"), pw.get("idle_pre_w")
    if drift is not None and pre and abs(drift) > max(0.05, 0.03 * pre):
        warn.append(f"idle baseline drift {drift:+.3f} W (post - pre)")
    # input identity
    clip = r.get("clip", {})
    if clip.get("expected_sha256") and clip.get("sha256") != clip.get("expected_sha256"):
        hard.append("clip hash mismatch")
    if r.get("loop_mismatches"):
        hard.append(f"clip pass length mismatch: {r['loop_mismatches']}")
    fps_meta = clip.get("fps_meta")
    if fps_meta and abs(fps_meta - cfg.get("fps", 30)) > 0.5:
        warn.append(f"clip metadata fps {fps_meta} differs from paced fps {cfg.get('fps')}")
    pc = r.get("pacing", {})
    if pc.get("policy") == "freerun":
        warn.append("freerun diagnostic: no 30 fps deadline; do not report as real-time")
    if pc.get("n_not_reached"):
        hard.append(f"{pc['n_not_reached']} timeline frames never reached")
    # protocol conformance
    if protocol is not None:
        if _get(r, "protocol.sha256") != protocol_sha:
            hard.append("run protocol hash differs from the checked protocol")
        expected = dict(protocol.get("run", {}))
        pset = _get(r, "protocol.set", "primary")
        if pset not in protocol.get("sets", {}):
            hard.append(f"unknown protocol set {pset}")
        else:
            expected.update(protocol["sets"][pset].get("run_overrides", {}))
            if r.get("mode") not in protocol["sets"][pset]["modes"]:
                hard.append(f"mode {r.get('mode')} not in protocol set {pset}")
        for k, v in expected.items():
            if cfg.get(k) != v and not (isinstance(v, float) and abs((cfg.get(k) or 0) - v) < 1e-6):
                hard.append(f"config {k}={cfg.get(k)} differs from protocol {v}")
    elif not r.get("protocol"):
        warn.append("run was not tied to a protocol file")
    miss = pc.get("n_deadline_miss")
    p99 = _get(pc, "response_ms.p99")
    budget = pc.get("budget_ms")
    return {
        "hard_fail": hard, "warnings": warn, "reportable": not hard,
        "workload_identical": pc.get("n_dropped") == 0 and not pc.get("n_not_reached"),
        "realtime_pass": (pc.get("policy") != "freerun" and pc.get("n_dropped") == 0 and miss == 0),
        "response_p99_within_budget": (p99 is not None and budget is not None and p99 <= budget),
    }


METRICS = {
    "response_p50_ms": "pacing.response_ms.p50", "response_p99_ms": "pacing.response_ms.p99",
    "response_max_ms": "pacing.response_ms.max",
    "processing_p50_ms": "pacing.processing_ms.p50", "processing_p99_ms": "pacing.processing_ms.p99",
    "schedule_delay_p99_ms": "pacing.schedule_delay_ms.p99", "dispatch_p99_ms": "pacing.dispatch_ms.p99",
    "deadline_miss": "pacing.n_deadline_miss", "dropped": "pacing.n_dropped",
    "busy_fraction": "pacing.busy_fraction",
    "detect_p50_ms": "stages_ms.detect.p50", "crop_p50_ms": "stages_ms.crop.p50",
    "encode_p50_ms": "stages_ms.encode.p50", "head_p50_ms": "stages_ms.head.p50",
    "read_p50_ms": "stages_ms.read.p50",
    "idle_pre_w": "power.idle_pre_w", "active_w": "power.active_w", "delta_w": "power.delta_w",
    "idle_drift_w": "power.idle_drift_w",
    "total_mj_per_timeline_frame": "power.total_mj_per_timeline_frame",
    "incremental_mj_per_timeline_frame": "power.incremental_mj_per_timeline_frame",
    "pmic_active_hz": "power.windows.active.effective_hz",
    "active_temp_max_c": "thermal.active.temp_c.max", "active_temp_first_c": "thermal.active.temp_c.first",
    "active_fan_rpm_mean": "thermal.active.fan_rpm.mean",
    "active_process_cpu_s": "phases.active.process_cpu_s",
    "face_rate": "outcomes.face_rate", "decisions": "outcomes.decision", "onsets": "outcomes.onset",
}
PAIRED = ("delta_w", "incremental_mj_per_timeline_frame", "active_w", "response_p99_ms", "processing_p50_ms")


def describe(vals):
    a = np.asarray([v for v in vals if v is not None], float)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()), "sd": float(a.std(ddof=1)) if a.size > 1 else None,
            "min": float(a.min()), "max": float(a.max()), "values": a.tolist()}


def aggregate(session_dir: Path, protocol_path: Path | None = None, ref_mode: str = REF_MODE) -> dict:
    protocol = sha = None
    if protocol_path:
        protocol = json.loads(protocol_path.read_text(encoding="utf8"))
        sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    runs, excluded = [], []
    for p in sorted(session_dir.glob("*/run.json")):
        r = json.loads(p.read_text(encoding="utf8"))
        q = qc_run(r, protocol, sha)
        row = {"run_id": r.get("run_id"), "mode": r.get("mode"), "block": r.get("block"),
               "order_index": r.get("order_index"), "path": str(p), "qc": q,
               "metrics": {k: _get(r, v) for k, v in METRICS.items()}}
        (runs if q["reportable"] else excluded).append(row)
    modes = sorted({x["mode"] for x in runs})
    per_mode = {}
    for m in modes:
        rows = [x for x in runs if x["mode"] == m]
        per_mode[m] = {"n_runs": len(rows), "run_ids": [x["run_id"] for x in rows],
                       "realtime_pass_all": all(x["qc"]["realtime_pass"] for x in rows),
                       "workload_identical_all": all(x["qc"]["workload_identical"] for x in rows),
                       "warnings": sorted({w for x in rows for w in x["qc"]["warnings"]}),
                       "metrics": {k: describe([x["metrics"][k] for x in rows]) for k in METRICS}}
    paired = {}
    ref = {x["block"]: x for x in runs if x["mode"] == ref_mode}
    for m in modes:
        if m == ref_mode:
            continue
        diffs = {k: [] for k in PAIRED}
        blocks = []
        for x in runs:
            if x["mode"] == m and x["block"] in ref:
                blocks.append(x["block"])
                for k in PAIRED:
                    a, b = x["metrics"][k], ref[x["block"]]["metrics"][k]
                    diffs[k].append(None if a is None or b is None else a - b)
        paired[f"{m} - {ref_mode}"] = {"blocks": blocks, **{k: describe(v) for k, v in diffs.items()}}
    faces = [x["metrics"]["face_rate"] for x in runs if x["metrics"]["face_rate"] is not None]
    return {
        "session_dir": str(session_dir), "protocol": str(protocol_path) if protocol_path else None,
        "protocol_sha256": sha, "n_runs_reportable": len(runs), "n_runs_excluded": len(excluded),
        "excluded": [{"run_id": x["run_id"], "mode": x["mode"], "reasons": x["qc"]["hard_fail"]} for x in excluded],
        "per_mode": per_mode, "paired_within_block": paired,
        "input_consistency": {"face_rate_range_all_runs": [min(faces), max(faces)] if faces else None},
        "_scope": ("Descriptive over repeated runs on one device and one clip; power is the PMIC rail sum, "
                   "not board input power; freerun and paced results are never pooled."),
    }


def to_markdown(s: dict) -> str:
    lines = [f"# Paced run summary — {s['session_dir']}", "",
             f"Reportable runs: {s['n_runs_reportable']}, excluded: {s['n_runs_excluded']}", ""]
    keys = ["response_p99_ms", "processing_p50_ms", "deadline_miss", "dropped", "detect_p50_ms",
            "encode_p50_ms", "idle_pre_w", "active_w", "delta_w", "incremental_mj_per_timeline_frame",
            "active_temp_max_c"]
    lines.append("| mode | n | " + " | ".join(keys) + " |")
    lines.append("|---|---:|" + "---:|" * len(keys))
    for m, d in s["per_mode"].items():
        cells = []
        for k in keys:
            x = d["metrics"][k]
            cells.append("—" if not x.get("n") else
                         f"{x['mean']:.4g}" + (f" ± {x['sd']:.2g}" if x.get("sd") is not None else ""))
        lines.append(f"| {m} | {d['n_runs']} | " + " | ".join(cells) + " |")
    lines += ["", "mean ± SD across runs (descriptive). Power = PMIC rail sum, not board input power.", ""]
    for name, d in s["paired_within_block"].items():
        x = d["delta_w"]
        if x.get("n"):
            lines.append(f"- paired {name}: delta_w diff mean {x['mean']:+.4f} W (n={x['n']} blocks)")
    for e in s["excluded"]:
        lines.append(f"- excluded {e['run_id']} ({e['mode']}): {'; '.join(e['reasons'])}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-dir", required=True, type=Path)
    ap.add_argument("--protocol", type=Path, default=None)
    ap.add_argument("--ref-mode", default=REF_MODE)
    ap.add_argument("--out", type=Path, default=None, help="default <session-dir>/session_summary.json")
    a = ap.parse_args(argv)
    s = aggregate(a.session_dir, a.protocol, a.ref_mode)
    out = a.out or a.session_dir / "session_summary.json"
    out.write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf8")
    out.with_suffix(".md").write_text(to_markdown(s), encoding="utf8")
    print(to_markdown(s))
    print("->", out)
    return 0 if s["n_runs_excluded"] == 0 and s["n_runs_reportable"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
