"""Fixed-protocol paced measurement session on the Pi (2026-09-28).

    python -m src.v2.deploy.paced_session --protocol protocol.json \
        --bundle-root bundle --session-id S1 [--set primary|secondary|freerun]

1. Preflight (refuses to start unless --exploratory): Pi platform, governor
   'performance' on every CPU, throttled=0x0, device-verified bundle, clip hash
   and decoded frame count, equivalence gate.
2. Session reference temperature: mean of `reference_idle_s` of idle sampling.
3. Plan: every block holds each mode once, in a seeded random permutation;
   the plan is written before the first run and never reshuffled.
4. Per run: cooldown of at least `cooldown_min_s`, then wait until the SoC
   temperature is within `cooldown_temp_margin_c` of the reference (at most
   `cooldown_max_extra_s` more; recorded either way); then one `run_paced`
   subprocess. Existing complete run.json files are kept (resume).
5. `check_paced_results` QC + summary.

`--exploratory` allows overrides for desktop dry runs; such sessions are
labelled exploratory and QC keeps them non-reportable (no protocol match / not Pi).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from src.v2.deploy import bench_telemetry as BT
from src.v2.deploy.run_paced import RUN_KEYS, sha256_file


def log(fp, msg: str) -> None:
    line = f"{datetime.now().astimezone().isoformat()} {msg}"
    print(line, flush=True)
    fp.write(line + "\n")
    fp.flush()


def decode_pass(path: str) -> dict:
    """Count decoded frames and digest their bytes once per session."""
    import cv2
    cap = cv2.VideoCapture(path)
    h, n, shape = hashlib.sha256(), 0, None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        h.update(np.ascontiguousarray(f).tobytes())
        shape = list(f.shape)
        n += 1
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return {"frames": n, "shape": shape, "decoded_sha256": h.hexdigest(), "fps_meta": fps,
            "opencv": cv2.__version__}


def make_plan(modes: list[str], repeats: int, seed: int, prefix: str) -> list[dict]:
    rng = random.Random(seed)
    plan = []
    for b in range(1, repeats + 1):
        order = list(modes)
        rng.shuffle(order)
        for m in order:
            plan.append({"block": b, "order_index": len(plan), "mode": m,
                         "run_id": f"{prefix}b{b:02d}_{m}"})
    return plan


def wait_cooldown(fp, ref_temp, min_s, margin, max_extra, read_temp=BT.read_temp_c) -> dict:
    t0 = time.monotonic()
    log(fp, f"cooldown: minimum {min_s:.0f}s")
    time.sleep(min_s)
    target = None if ref_temp is None else ref_temp + margin
    met = target is None
    temp = read_temp()
    while target is not None and not met and time.monotonic() - t0 < min_s + max_extra:
        if temp is not None and temp <= target:
            met = True
            break
        time.sleep(5)
        temp = read_temp()
    if target is not None and temp is not None and temp <= target:
        met = True
    rec = {"waited_s": time.monotonic() - t0, "target_c": target, "temp_at_start_c": temp,
           "target_met": met}
    log(fp, f"cooldown done: {rec}")
    return rec


def preflight(protocol: dict, bundle_root: Path, clip: str, verification: Path, eq: Path) -> dict:
    snap = BT.platform_snapshot()
    problems = []
    if platform.machine() != "aarch64" or not snap["cmdline_has_vcgencmd"]:
        problems.append("not a Raspberry Pi (aarch64 + vcgencmd)")
    gov = snap["governors"]
    if not gov or any(v != "performance" for v in gov.values()):
        problems.append(f"governor not performance: {gov}")
    thr = snap["throttled"]
    if thr["raw"] != "0x0":
        problems.append(f"throttled={thr['raw']} {thr['now'] + thr['sticky']} (reboot / fix power & cooling)")
    exp = protocol["input"]
    sha = sha256_file(clip)
    if sha != exp["sha256"]:
        problems.append("clip sha256 differs from protocol")
    dec = decode_pass(clip)
    if dec["frames"] != exp["frames_per_pass"]:
        problems.append(f"decoded frames {dec['frames']} != protocol {exp['frames_per_pass']}")
    dec["matches_pc_reference"] = dec["decoded_sha256"] == exp.get("decoded_sha256_pc_reference")
    try:
        v = json.loads(verification.read_text(encoding="utf8"))
        import onnxruntime as ort
        if not (v.get("passed") and v.get("machine") == platform.machine()
                and v.get("onnxruntime") == ort.__version__):
            problems.append("bundle verification is not a pass on this device/ORT")
    except Exception as e:
        problems.append(f"bundle verification unreadable: {e}")
    try:
        if not json.loads(eq.read_text(encoding="utf8")).get("pi_measurement_authorized"):
            problems.append("equivalence gate not authorized")
    except Exception as e:
        problems.append(f"equivalence gate unreadable: {e}")
    return {"platform": snap, "clip_sha256": sha, "decode": dec, "problems": problems}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True, type=Path)
    ap.add_argument("--bundle-root", required=True, type=Path)
    ap.add_argument("--bundle-verification", type=Path, default=None)
    ap.add_argument("--equivalence-gate", type=Path, default=Path("results/v2/check_equivalence.json"))
    ap.add_argument("--source", default=None, help="default: protocol input.clip")
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--out-root", type=Path, default=Path("results/pi_paced"))
    ap.add_argument("--set", default="primary", choices=["primary", "secondary", "freerun"])
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--exploratory", action="store_true",
                    help="desktop/dry run: skip preflight refusal, allow overrides; never reportable")
    ap.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE",
                    help="exploratory only, e.g. active_s=10 repeats=1 cooldown_min_s=0")
    a = ap.parse_args(argv)

    protocol = json.loads(a.protocol.read_text(encoding="utf8"))
    run_cfg = dict(protocol["run"])
    sess = dict(protocol["session"])
    sets = protocol["sets"][a.set]
    run_cfg.update(sets.get("run_overrides", {}))
    if a.override and not a.exploratory:
        ap.error("--override needs --exploratory")
    for kv in a.override:
        k, v = kv.split("=", 1)
        v = json.loads(v)
        (run_cfg if k in run_cfg else sess)[k] = v
    clip = a.source or protocol["input"]["clip"]
    ver = a.bundle_verification or a.bundle_root / "pi_model_verification.json"
    out_dir = a.out_root / a.session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = open(out_dir / "session.log", "a", encoding="utf8")
    log(fp, f"session {a.session_id} set={a.set} exploratory={a.exploratory} protocol={a.protocol}")

    plan_path = out_dir / "session_plan.json"
    if plan_path.exists():
        plan_doc = json.loads(plan_path.read_text(encoding="utf8"))
        log(fp, "resuming existing plan (not reshuffled)")
    else:
        pre = preflight(protocol, a.bundle_root, clip, ver, a.equivalence_gate)
        for p in pre["problems"]:
            log(fp, f"PREFLIGHT: {p}")
        if pre["problems"] and not a.exploratory:
            log(fp, "preflight failed; nothing measured")
            return 2
        reps = sess["repeats"] if any(o.startswith("repeats=") for o in a.override) else sets.get("repeats", sess["repeats"])
        plan = make_plan(sets["modes"], int(reps), int(sess["order_seed"]), sets.get("run_prefix", ""))
        plan_doc = {"session_id": a.session_id, "set": a.set, "exploratory": a.exploratory,
                    "overrides": a.override, "protocol": str(a.protocol),
                    "protocol_sha256": sha256_file(a.protocol), "protocol_id": protocol["protocol_id"],
                    "run_config": run_cfg, "session_config": sess, "clip": clip,
                    "created": datetime.now().astimezone().isoformat(), "preflight": pre, "plan": plan}
        plan_path.write_text(json.dumps(plan_doc, ensure_ascii=False, indent=1), encoding="utf8")
    if a.plan_only:
        for p in plan_doc["plan"]:
            print(p)
        return 0

    ref_temp = plan_doc.get("reference_temp_c")
    if "reference_temp_c" not in plan_doc:
        secs = float(sess["reference_idle_s"])
        log(fp, f"reference idle {secs:.0f}s for session temperature")
        temps = []
        t_end = time.monotonic() + secs
        while time.monotonic() < t_end:
            t = BT.read_temp_c()
            if t is not None:
                temps.append(t)
            time.sleep(1.0)
        ref_temp = float(np.mean(temps)) if temps else None
        plan_doc["reference_temp_c"] = ref_temp
        plan_path.write_text(json.dumps(plan_doc, ensure_ascii=False, indent=1), encoding="utf8")
        log(fp, f"reference temp {ref_temp}")

    records_path = out_dir / "session_runs.jsonl"
    first = True
    for item in plan_doc["plan"]:
        if (out_dir / item["run_id"] / "run.json").exists():
            log(fp, f"skip existing {item['run_id']}")
            continue
        cool = None
        if not first:
            cool = wait_cooldown(fp, ref_temp, float(sess["cooldown_min_s"]),
                                 float(sess["cooldown_temp_margin_c"]), float(sess["cooldown_max_extra_s"]))
        first = False
        cmd = [a.python, "-m", "src.v2.deploy.run_paced", "--mode", item["mode"],
               "--bundle-root", str(a.bundle_root), "--bundle-verification", str(ver),
               "--equivalence-gate", str(a.equivalence_gate), "--source", clip,
               "--expected-clip-sha256", protocol["input"]["sha256"],
               "--expected-clip-frames", str(protocol["input"]["frames_per_pass"]),
               "--protocol", str(a.protocol), "--protocol-set", a.set, "--out-dir", str(out_dir), "--run-id", item["run_id"],
               "--session-id", a.session_id, "--block", str(item["block"]),
               "--order-index", str(item["order_index"])]
        for k in RUN_KEYS:
            if k in ("refine_landmarks", "head_stride"):
                continue
            v = run_cfg[k]
            flag = "--" + k.replace("_", "-")
            if k == "no_spin":
                cmd.append("--no-spin" if v else "--allow-spin")
            else:
                cmd += [flag, str(v)]
        log(fp, f"run {item['run_id']} (block {item['block']}, #{item['order_index']}): {' '.join(cmd)}")
        t0 = time.monotonic()
        rc = subprocess.run(cmd).returncode
        rec = {**item, "exit_code": rc, "seconds": time.monotonic() - t0, "cooldown": cool,
               "finished": datetime.now().astimezone().isoformat()}
        with open(records_path, "a", encoding="utf8") as f:
            f.write(json.dumps(rec) + "\n")
        log(fp, f"run {item['run_id']} exit {rc}")

    from src.v2.deploy.check_paced_results import aggregate, to_markdown
    s = aggregate(out_dir, a.protocol)
    (out_dir / "session_summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf8")
    (out_dir / "session_summary.md").write_text(to_markdown(s), encoding="utf8")
    log(fp, f"summary: reportable {s['n_runs_reportable']}, excluded {s['n_runs_excluded']}")
    fp.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
