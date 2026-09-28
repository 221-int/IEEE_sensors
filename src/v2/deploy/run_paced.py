"""One paced 30 fps latency/power run on a fixed clip (2026-09-28).

Phases (all in one process, models already loaded):

    warmup     paced replay of the first `warmup_s` of the clip; not analysed
    settle     sleep, not analysed (lets the warmup transient decay)
    idle_pre   sleep, PMIC baseline (models loaded, no frame reads)
    active     paced replay of exactly round(active_s * fps) timeline frames,
               starting at clip frame 0; the clip loops with a ring reset at
               each seam; lasts until the last frame slot has elapsed
    idle_post  sleep, drift check of the baseline

Every phase boundary is printed and logged with monotonic and wall-clock
times so an external power meter log can be aligned afterwards.

Model identity comes only from a graph-hash-bound contract in the transferred
bundle; mode names say which model ran (vdrop/gap are never reported as
"ours"). `ear_rule` is the weight-free EAR drop ratio; it has no validated
threshold and yields scores only.

    python -m src.v2.deploy.run_paced --mode ours_vpres \
        --bundle-root bundle --source clips/eb9_640x480.avi \
        --protocol protocol.json --out-dir results/pi_paced/S1 --run-id b1_ours_vpres

Use `paced_session.py` for the fixed protocol; call this directly only for
single diagnostics. Desktop runs are dry runs and QC marks them non-reportable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.v2.dataset import crop as C
from src.v2.deploy import inference_policy as IP
from src.v2.deploy.pacing import FrameScheduler, LoopingVideoSource, MonotonicClock, summarize_ms

MODE_SPECS = {                       # mode -> (sub-folder under model root, contract variant)
    "ours_vpres": ("", "vpres"),
    "image_cnn_head": ("image_cnn_head", "image_head"),
    "vdrop": ("vdrop", "vdrop"),
    "gap": ("gap", "gap"),
    "ear_rule": None,
}
MODEL_ROOT = "onnx/fold0_seed0"
RUN_KEYS = ("fps", "pacing", "budget_ms", "warmup_s", "settle_s", "idle_pre_s", "active_s",
            "idle_post_s", "power_hz", "thermal_hz", "intra_threads", "no_spin",
            "refine_landmarks", "head_stride")
SCHEMA = "paced_run_v1"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ear_drop_ratio(buf) -> float:
    """Same formula as run_video.ear_drop_ratio / train_encoder EAR feature."""
    v = np.asarray(buf, np.float64)
    ok = np.isfinite(v)
    if not ok.any():
        return 0.0
    edge = (v[np.argmax(ok)] + v[len(v) - 1 - np.argmax(ok[::-1])]) / 2.0
    lo = np.nanmin(np.where(ok, v, np.inf))
    if not np.isfinite(edge) or edge == 0:
        return 0.0
    return float(np.nan_to_num((edge - lo) / edge, nan=0.0, posinf=0.0, neginf=0.0))


# ---------------------------------------------------------------------- gates
def check_bundle(bundle_root: Path, verification: Path, mode: str) -> dict:
    """Bundle integrity + device verification for the model this run uses."""
    import onnxruntime as ort
    out = {"bundle_root": str(bundle_root), "verification_file": str(verification), "errors": []}
    try:
        manifest = json.loads((bundle_root / "manifest.json").read_text(encoding="utf8"))
        out["manifest_sha256"] = sha256_file(bundle_root / "manifest.json")
    except Exception as e:
        out["errors"].append(f"manifest unreadable: {e}")
        out["passed"] = False
        return out
    ver = None
    try:
        ver = json.loads(verification.read_text(encoding="utf8"))
    except Exception as e:
        out["errors"].append(f"device verification unreadable: {e}")
    if ver is not None:
        out.update(verified_machine=ver.get("machine"), verified_ort=ver.get("onnxruntime"),
                   verified_platform=ver.get("platform"), verified_created=ver.get("created"))
        if not ver.get("passed"):
            out["errors"].append("device verification did not pass")
        if ver.get("machine") != platform.machine():
            out["errors"].append(f"verification machine {ver.get('machine')} != {platform.machine()}")
        if ver.get("onnxruntime") != ort.__version__:
            out["errors"].append(f"verification ORT {ver.get('onnxruntime')} != {ort.__version__}")
    spec = MODE_SPECS[mode]
    if spec is not None:
        contract_rel = "/".join(p for p in (MODEL_ROOT, spec[0], "contract.json") if p)
        out["contract"] = contract_rel
        if ver is not None and contract_rel not in {r.get("contract") for r in ver.get("reports", [])}:
            out["errors"].append(f"{contract_rel} not covered by device verification")
        raw = json.loads((bundle_root / contract_rel).read_text(encoding="utf8"))
        folder = "/".join(contract_rel.split("/")[:-1])
        for rel in [contract_rel] + [f"{folder}/{g}" for g in raw["graph_files"].values()]:
            if manifest.get("sha256", {}).get(rel) != sha256_file(bundle_root / rel):
                out["errors"].append(f"manifest hash mismatch: {rel}")
    out["passed"] = not out["errors"]
    return out


def check_equivalence_gate(path: Path) -> dict:
    try:
        eq = json.loads(path.read_text(encoding="utf8"))
    except Exception as e:
        return {"path": str(path), "authorized": False, "error": str(e)}
    return {"path": str(path), "authorized": bool(eq.get("pi_measurement_authorized")),
            "sha256": sha256_file(path)}


# ---------------------------------------------------------------------- model
def make_session(path: str, intra: int, no_spin: bool):
    import onnxruntime as ort
    so = ort.SessionOptions()
    if intra:
        so.intra_op_num_threads = intra
    so.inter_op_num_threads = 1
    if no_spin:
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        so.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def load_model(bundle_root: Path, mode: str, intra: int, no_spin: bool):
    spec = MODE_SPECS[mode]
    if spec is None:
        return None, None, None
    folder = bundle_root / MODEL_ROOT / spec[0] if spec[0] else bundle_root / MODEL_ROOT
    raw = json.loads((folder / "contract.json").read_text(encoding="utf8"))
    enc_p, head_p = folder / raw["graph_files"]["encoder"], folder / raw["graph_files"]["head"]
    contract = IP.load_contract(folder, [enc_p, head_p])      # hashes, threshold, policy
    if contract["variant"] != spec[1]:
        raise ValueError(f"mode {mode} expects variant {spec[1]}, contract has {contract['variant']}")
    return (make_session(str(enc_p), intra, no_spin), make_session(str(head_p), intra, no_spin),
            contract)


class Pipeline:
    """Frame -> mesh -> crop -> encoder -> 19-slot ring -> head, as run_video.py."""

    def __init__(self, mode, frontend, enc=None, head=None, contract=None, now=time.perf_counter):
        self.mode, self.fe, self.enc, self.head, self.contract, self.now = mode, frontend, enc, head, contract, now
        self.cnn = MODE_SPECS[mode] is not None
        if self.cnn and (enc is None or head is None or contract is None):
            raise ValueError("CNN modes need both graphs and a verified contract")
        self.threshold = contract["threshold_probability"] if contract else None
        self.reset()

    def reset(self) -> None:
        self.ring: deque = deque(maxlen=IP.EVENT_LEN)
        self.ear_ring: deque = deque(maxlen=IP.EVENT_LEN)
        self.was_positive = False

    def missing(self, count: int = 1) -> None:
        IP.advance_missing(self.ring, count)
        for _ in range(min(count, IP.EVENT_LEN)):
            self.ear_ring.append(np.nan)
        if count:
            self.was_positive = False

    def process(self, frame) -> dict:
        r = {"face": False, "crop_ok": False, "decision": False, "prob": np.nan,
             "onset": False, "detect": np.nan, "crop": np.nan, "encode": np.nan, "head": np.nan}
        t0 = self.now()
        mesh = self.fe.mesh(frame)
        t1 = self.now()
        r["detect"] = t1 - t0
        p = None
        if mesh is None:
            self.ring.append(None)
            self.ear_ring.append(np.nan)
        elif self.cnn:
            r["face"] = True
            g, _ = self.fe.crop_from_mesh(frame, mesh)
            x = C.to_input_tensor(g) if g is not None else None
            t2 = self.now()
            r["crop"] = t2 - t1
            if x is None:
                self.ring.append(None)
            else:
                r["crop_ok"] = True
                z = self.enc.run(None, {"crop": x.astype(np.float32)})[0][0]
                t3 = self.now()
                r["encode"] = t3 - t2
                self.ring.append(z)
                if IP.ring_ready(self.ring):
                    d = len(z)
                    zz = np.stack([v if v is not None else np.zeros(d, np.float32)
                                   for v in self.ring])[None].astype(np.float32)
                    mk = np.array([[0.0 if v is None else 1.0 for v in self.ring]], np.float32)
                    p = float(self.head.run(None, {"vectors": zz, "mask": mk})[0].reshape(-1)[0])
                    r["head"] = self.now() - t3
        else:
            r["face"] = r["crop_ok"] = True
            e = C.ear_both(mesh)
            self.ear_ring.append(e["mean"] if "mean" in e else float(np.mean(list(e.values()))))
            t2 = self.now()
            r["crop"] = t2 - t1
            if len(self.ear_ring) == IP.EVENT_LEN and IP.eligible_mask(np.isfinite(self.ear_ring)):
                p = ear_drop_ratio(list(self.ear_ring))
                r["head"] = self.now() - t2
        if p is not None:
            r["decision"], r["prob"] = True, p
        if self.threshold is not None:
            self.was_positive, r["onset"] = IP.candidate_onset(p, self.threshold, self.was_positive)
        return r


# ---------------------------------------------------------------------- run
FRAME_FIELDS = ("read", "skip", "detect", "crop", "encode", "head", "prob")
FRAME_FLAGS = ("face", "crop_ok", "decision", "onset", "seam")


def run_timeline(sched: FrameScheduler, pipe: Pipeline, source, now) -> dict:
    n = sched.n
    rec = {k: np.full(n, np.nan) for k in FRAME_FIELDS}
    rec.update({k: np.zeros(n, bool) for k in FRAME_FLAGS})
    rec["loop"] = np.full(n, -1, np.int32)
    rec["pos"] = np.full(n, -1, np.int32)
    sched.begin()
    while (slot := sched.acquire()) is not None:
        ts = now()
        for _ in slot.skipped:
            if source.grab()["seam"]:
                pipe.reset()
            pipe.missing(1)
        t0 = now()
        frame, info = source.read()
        t1 = now()
        if info["seam"]:
            pipe.reset()
        r = pipe.process(frame)
        sched.complete(slot)
        i = slot.index
        rec["skip"][i], rec["read"][i] = t0 - ts, t1 - t0
        for k in ("detect", "crop", "encode", "head", "prob"):
            rec[k][i] = r[k]
        for k in ("face", "crop_ok", "decision", "onset"):
            rec[k][i] = r[k]
        rec["seam"][i], rec["loop"][i], rec["pos"][i] = info["seam"], info["loop"], info["pos"]
    sched.close()
    return rec


def outcome_summary(rec: dict, processed: np.ndarray) -> dict:
    p = processed
    out = {k: int(rec[k][p].sum()) for k in ("face", "crop_ok", "decision", "onset")}
    out["n_processed"] = int(p.sum())
    out["face_rate"] = out["face"] / out["n_processed"] if out["n_processed"] else None
    out["n_seams"] = int(rec["seam"].sum())
    out["seam_timeline_indices"] = np.flatnonzero(rec["seam"]).tolist()
    return out


def stage_summary(rec: dict, processed: np.ndarray) -> dict:
    return {k: summarize_ms(rec[k][processed] * 1e3) for k in ("read", "skip", "detect", "crop", "encode", "head")}


class PhaseLog:
    def __init__(self, now):
        self.now, self.entries = now, []

    def mark(self, phase: str, event: str) -> float:
        t, wall = self.now(), time.time()
        ct = os.times()
        self.entries.append({"phase": phase, "event": event, "t_perf": t, "t_wall": wall,
                             "wall_iso": datetime.fromtimestamp(wall, timezone.utc).isoformat(),
                             "cpu_user_s": ct.user, "cpu_sys_s": ct.system,
                             "children_user_s": ct.children_user, "children_sys_s": ct.children_system})
        print(f"[phase] {phase:<9} {event:<5} wall={self.entries[-1]['wall_iso']}", flush=True)
        return t

    def window(self, phase: str):
        s = [e for e in self.entries if e["phase"] == phase]
        if len(s) != 2:
            return None
        a, b = s
        return {"t0": a["t_perf"], "t1": b["t_perf"], "wall0": a["wall_iso"], "wall1": b["wall_iso"],
                "duration_s": b["t_perf"] - a["t_perf"],
                "process_cpu_s": (b["cpu_user_s"] + b["cpu_sys_s"]) - (a["cpu_user_s"] + a["cpu_sys_s"]),
                "children_cpu_s": (b["children_user_s"] + b["children_sys_s"])
                                  - (a["children_user_s"] + a["children_sys_s"])}


def execute(cfg: dict, pipe: Pipeline, source, clock, thermal, power, now=None) -> dict:
    """Run all phases; returns raw material for the result JSON (no gates here)."""
    now = now or clock.now
    log = PhaseLog(now)
    thermal.start()
    power.start()

    def idle(phase, seconds):
        log.mark(phase, "start")
        clock.sleep_until(now() + seconds)
        log.mark(phase, "end")

    warm = None
    n_warm = int(round(cfg["warmup_s"] * cfg["fps"]))
    if n_warm > 0:
        log.mark("warmup", "start")
        sw = FrameScheduler(cfg["fps"], n_warm, cfg["pacing"], clock, cfg["budget_ms"] / 1e3)
        run_timeline(sw, pipe, source, now)
        log.mark("warmup", "end")
        warm = sw.summary()
    source.restart()
    pipe.reset()
    idle("settle", cfg["settle_s"])
    idle("idle_pre", cfg["idle_pre_s"])

    n_act = int(round(cfg["active_s"] * cfg["fps"]))
    sched = FrameScheduler(cfg["fps"], n_act, cfg["pacing"], clock, cfg["budget_ms"] / 1e3)
    log.mark("active", "start")
    rec = run_timeline(sched, pipe, source, now)
    log.mark("active", "end")
    idle("idle_post", cfg["idle_post_s"])
    thermal.stop()
    power.stop()
    pf = sched.per_frame()
    return {"log": log, "warmup_pacing": warm, "pacing": sched.summary(), "rec": rec, "per_frame": pf,
            "stages_ms": stage_summary(rec, pf["processed"]),
            "outcomes": outcome_summary(rec, pf["processed"]),
            "loop_lengths": list(source.loop_lengths), "loop_mismatches": source.length_mismatches()}


def power_summary(power, log: PhaseLog, n_timeline: int, n_processed: int) -> dict:
    w = {ph: (power.window(log.window(ph)["t0"], log.window(ph)["t1"]) if log.window(ph) else None)
         for ph in ("warmup", "idle_pre", "active", "idle_post")}
    pre, act, post = (w[k] and w[k].get("mean_w") for k in ("idle_pre", "active", "idle_post"))
    out = {"quantity": "pmic_rail_sum_w (12 current rails V*I; not verified as board input power)",
           "windows": w, "idle_pre_w": pre, "active_w": act, "idle_post_w": post,
           "parse_errors": power.parse_errors, "missing_rails": sorted(power.missing_rails),
           "n_raw_samples": len(power.raw), "first_raw_sample": power.first_raw_sample}
    if pre is not None and act is not None:
        T = w["active"]["duration_s"]
        delta = act - pre
        out.update(
            delta_w=delta,
            delta_w_vs_pre_post_mean=(act - (pre + post) / 2.0) if post is not None else None,
            idle_drift_w=(post - pre) if post is not None else None,
            active_energy_total_j=act * T, active_energy_incremental_j=delta * T,
            total_mj_per_timeline_frame=act * T / n_timeline * 1e3,
            incremental_mj_per_timeline_frame=delta * T / n_timeline * 1e3,
            total_mj_per_processed_frame=(act * T / n_processed * 1e3) if n_processed else None,
            incremental_mj_per_processed_frame=(delta * T / n_processed * 1e3) if n_processed else None,
            energy_method="window mean power x window duration; incremental = (active - idle_pre) x duration")
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", required=True, choices=sorted(MODE_SPECS))
    ap.add_argument("--bundle-root", required=True, type=Path)
    ap.add_argument("--bundle-verification", type=Path, default=None,
                    help="default: <bundle-root>/pi_model_verification.json (made on this device)")
    ap.add_argument("--equivalence-gate", type=Path, default=Path("results/v2/check_equivalence.json"))
    ap.add_argument("--source", required=True)
    ap.add_argument("--expected-clip-sha256", default=None)
    ap.add_argument("--expected-clip-frames", type=int, default=None)
    ap.add_argument("--protocol", type=Path, default=None)
    ap.add_argument("--protocol-set", default="primary", help="protocol 'sets' entry this run belongs to")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--session-id", default=None)
    ap.add_argument("--block", type=int, default=None)
    ap.add_argument("--order-index", type=int, default=None)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--pacing", default="drop", choices=["drop", "queue", "freerun"])
    ap.add_argument("--budget-ms", type=float, default=None, help="default one frame period")
    ap.add_argument("--warmup-s", type=float, default=10.0)
    ap.add_argument("--settle-s", type=float, default=5.0)
    ap.add_argument("--idle-pre-s", type=float, default=60.0)
    ap.add_argument("--active-s", type=float, default=300.0)
    ap.add_argument("--idle-post-s", type=float, default=30.0)
    ap.add_argument("--power-hz", type=float, default=7.5)
    ap.add_argument("--thermal-hz", type=float, default=1.0)
    ap.add_argument("--intra-threads", type=int, default=2)
    # explicit pair: argparse.BooleanOptionalAction would read "--no-spin" itself as False
    ap.add_argument("--no-spin", dest="no_spin", action="store_true", default=True)
    ap.add_argument("--allow-spin", dest="no_spin", action="store_false",
                    help="diagnostic only; protocol runs disable ORT spinning")
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.budget_ms is None:
        args.budget_ms = 1000.0 / args.fps
    cfg = {k: getattr(args, k, None) for k in RUN_KEYS}
    cfg.update(refine_landmarks=False, head_stride=1)
    out_dir = args.out_dir / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now().astimezone().isoformat()

    from src.v2.common import repro
    from src.v2.deploy import bench_telemetry as BT
    from src.v2.deploy.check_paced_results import qc_run

    ver = args.bundle_verification or args.bundle_root / "pi_model_verification.json"
    gates = {"bundle": check_bundle(args.bundle_root, ver, args.mode),
             "equivalence": check_equivalence_gate(args.equivalence_gate)}
    clip = {"path": str(args.source), "sha256": sha256_file(args.source),
            "expected_sha256": args.expected_clip_sha256, "expected_frames_per_pass": args.expected_clip_frames}
    if not gates["bundle"]["passed"] or not gates["equivalence"]["authorized"]:
        print("gate failed; not measuring:", json.dumps(gates, indent=1), file=sys.stderr)
        return 2
    if args.expected_clip_sha256 and clip["sha256"] != args.expected_clip_sha256:
        print(f"clip hash mismatch: {clip['sha256']}", file=sys.stderr)
        return 2

    import cv2
    import onnxruntime as ort
    from src.v2.deploy.frontend import EyeFrontend
    cap = cv2.VideoCapture(args.source)
    clip.update(fps_meta=cap.get(cv2.CAP_PROP_FPS), frame_count_meta=cap.get(cv2.CAP_PROP_FRAME_COUNT),
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()

    t_load = time.perf_counter()
    enc, head, contract = load_model(args.bundle_root, args.mode, args.intra_threads, args.no_spin)
    fe = EyeFrontend(refine_landmarks=False)
    fe._ensure_mesh()
    load_s = time.perf_counter() - t_load
    clock = MonotonicClock()
    pipe = Pipeline(args.mode, fe, enc, head, contract, now=clock.now)
    source = LoopingVideoSource(args.source, args.expected_clip_frames)
    snap_start = BT.platform_snapshot()
    thermal = BT.ThermalSampler(args.thermal_hz)
    power = BT.LoggingPowerSampler(args.power_hz)
    try:
        res = execute(cfg, pipe, source, clock, thermal, power)
    finally:
        source.release()
        fe.close()
    snap_end = BT.platform_snapshot()
    log = res["log"]
    n_tl, n_proc = res["pacing"]["n_timeline_frames"], res["pacing"]["n_processed"]

    np.savez_compressed(out_dir / "frames.npz",
                        **{k: v for k, v in res["per_frame"].items()},
                        **{f"stage_{k}": v for k, v in res["rec"].items()})
    power.write_raw(str(out_dir / "pmic_raw.jsonl.gz"))
    protocol = None
    if args.protocol:
        protocol = {"path": str(args.protocol), "sha256": sha256_file(args.protocol),
                    "id": json.loads(args.protocol.read_text(encoding="utf8")).get("protocol_id"),
                    "set": args.protocol_set}
    result = {
        "schema": SCHEMA, "run_id": args.run_id, "session_id": args.session_id,
        "block": args.block, "order_index": args.order_index, "started": started,
        "finished": datetime.now().astimezone().isoformat(),
        "mode": args.mode, "config": cfg, "protocol": protocol,
        "model": None if contract is None else {
            "variant": contract["variant"], "fold": contract["fold"], "seed": contract["seed"],
            "study": contract.get("study"), "graph_sha256": contract["graph_sha256"],
            "threshold_probability": contract["threshold_probability"]},
        "weights": "rule (no weights)" if contract is None else "trained (graph hash verified)",
        "inference_policy": IP.metadata(contract),
        "gates": gates, "clip": clip,
        "loop_lengths": res["loop_lengths"], "loop_mismatches": res["loop_mismatches"],
        "env": {**repro.env_fingerprint(), "onnxruntime": ort.__version__, "opencv": cv2.__version__,
                "machine": platform.machine(), "python_executable": sys.executable},
        "model_load_s": load_s,
        "phases": {ph: log.window(ph) for ph in ("warmup", "settle", "idle_pre", "active", "idle_post")},
        "phase_log": log.entries,
        "warmup_pacing": res["warmup_pacing"], "pacing": res["pacing"],
        "stages_ms": res["stages_ms"], "outcomes": res["outcomes"],
        "power": power_summary(power, log, n_tl, n_proc),
        "thermal": {ph: thermal.window(log.window(ph)["t0"], log.window(ph)["t1"])
                    for ph in ("warmup", "idle_pre", "active", "idle_post") if log.window(ph)},
        "thermal_samples": thermal.samples,
        "platform_start": snap_start, "platform_end": snap_end,
        "files": {"frames": "frames.npz", "pmic_raw": "pmic_raw.jsonl.gz"},
        "_notes": ["Latency = paced 30 Hz replay of a fixed clip; not free-run throughput.",
                   "Power = PMIC rail sum; not validated as whole-board input power."],
    }
    result["qc"] = qc_run(result)
    tmp = out_dir / "run.json.tmp"
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=float), encoding="utf8")
    os.replace(tmp, out_dir / "run.json")
    pc = res["pacing"]
    print(f"{args.mode} {args.run_id}: processed {n_proc}/{n_tl}, dropped {pc['n_dropped']}, "
          f"miss {pc.get('n_deadline_miss')}, response p99 {pc.get('response_ms', {}).get('p99')}, "
          f"reportable={result['qc']['reportable']}")
    for w in result["qc"]["hard_fail"] + result["qc"]["warnings"]:
        print("  !", w)
    print("  ->", out_dir / "run.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
