"""
run_video.py

Per-stage latency benchmark on a REAL video, for the Pi5 edge measurement.

This is the entry point the paper table is built from. It streams an actual
clip frame-by-frame (never random tensors -- crop cost depends on where the
face is) and times every stage separately:

    read    frame acquisition / decode
    detect  MediaPipe FaceMesh -> landmarks
    crop    eye corners -> canonical 64x160 crop -> float32 tensor
    infer   ONNX graph (encoder [+ MLP])
    e2e     sum of the above

Reports p50 / p95 / p99 (not the mean -- reviewers care about the worst case
against the 33.3 ms budget at 30 fps), plus sustained fps, CPU%, peak RSS, SoC
temperature and the Raspberry Pi throttle flags sampled throughout the run.

Two modes, same harness, so the numbers are directly comparable:
    --mode ours   read -> detect -> crop -> ONNX  (vector pipeline)
    --mode ear    read -> detect -> EAR           (classic baseline)

Run (Pi, 5 minutes, both modes):
    python -m src.deploy.run_video --video clip.avi --mode ours --minutes 5 \
        --csv results/pi_ours.csv --json results/pi_ours.json
    python -m src.deploy.run_video --video clip.avi --mode ear  --minutes 5 \
        --csv results/pi_ear.csv  --json results/pi_ear.json

Quick smoke test (a few hundred frames):
    python -m src.deploy.run_video --video clip.avi --mode ours --max-frames 300

Notes
-----
- The first --warmup frames are excluded from the statistics (cache / lazy init).
- --minutes loops the clip until the wall-clock target is reached. Pi5 throttles
  under sustained load, so a 30-second number and a 10-minute number differ --
  that difference is itself a result.
- By default `infer` is ONE stage (pipeline.onnx fuses encoder + MLP). To split
  it into the two rows the paper table wants, export an mlp.onnx and pass
  --encoder/--mlp; the script then times them separately.
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time

import numpy as np

try:
    import psutil
except ImportError:
    psutil = None


# ------------------------------- helpers -------------------------------
def _run(cmd):
    """Run a shell command, return stripped stdout or None."""
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                             timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def read_temp_c():
    """SoC temperature in Celsius, or None."""
    out = _run("vcgencmd measure_temp")
    if out and "=" in out:
        try:
            return float(out.split("=")[1].rstrip("'C"))
        except ValueError:
            pass
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return None


def read_throttled():
    """Raspberry Pi throttle bitmask as a hex string, or None.

    bit 0 under-voltage now      bit 16 under-voltage occurred
    bit 1 arm freq capped now    bit 17 arm freq capped occurred
    bit 2 currently throttled    bit 18 throttling occurred
    bit 3 soft temp limit now    bit 19 soft temp limit occurred
    """
    out = _run("vcgencmd get_throttled")
    if out and "=" in out:
        return out.split("=")[1]
    return None


def decode_throttled(hexstr):
    if not hexstr:
        return []
    try:
        v = int(hexstr, 16)
    except ValueError:
        return []
    flags = [
        (0, "under-voltage NOW"), (1, "arm-freq-capped NOW"),
        (2, "throttled NOW"), (3, "soft-temp-limit NOW"),
        (16, "under-voltage occurred"), (17, "arm-freq-capped occurred"),
        (18, "THROTTLING OCCURRED"), (19, "soft-temp-limit occurred"),
    ]
    return [name for bit, name in flags if v & (1 << bit)]


class Monitor(threading.Thread):
    """Background sampler: temperature, throttle flags, CPU%, RSS."""

    def __init__(self, period=2.0):
        super().__init__(daemon=True)
        self.period = period
        self.stop_evt = threading.Event()
        self.temps = []
        self.cpu = []
        self.rss = []
        self.throttle_seen = set()
        self.throttle_raw = []
        self.proc = psutil.Process(os.getpid()) if psutil else None
        if self.proc:
            self.proc.cpu_percent(None)  # prime

    def sample(self):
        t = read_temp_c()
        if t is not None:
            self.temps.append(t)
        th = read_throttled()
        if th:
            self.throttle_raw.append(th)
            for f in decode_throttled(th):
                self.throttle_seen.add(f)
        if self.proc:
            # cpu_percent(None) measures since the previous call, so the wait
            # below must happen BEFORE the first sample or it always reads 0.
            self.cpu.append(self.proc.cpu_percent(None))
            self.rss.append(self.proc.memory_info().rss / 1e6)

    def run(self):
        while not self.stop_evt.is_set():
            self.stop_evt.wait(self.period)
            self.sample()

    def stop(self):
        self.stop_evt.set()
        self.join(timeout=3)
        self.sample()  # final reading, so short runs still get one


def pct(vals, q):
    if not vals:
        return float("nan")
    return float(np.percentile(np.asarray(vals), q))


def env_info(sess_list, args):
    """Everything §5 says to record about the measurement environment."""
    import cv2

    info = {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "input_hw": [args.h, args.w],
        "mode": args.mode,
        "video": os.path.basename(args.video),
    }
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    info["os"] = line.split("=", 1)[1].strip().strip('"')
    except Exception:
        pass
    try:
        import mediapipe
        info["mediapipe"] = mediapipe.__version__
    except Exception:
        pass
    if sess_list:
        import onnxruntime as ort
        info["onnxruntime"] = ort.__version__
        info["ort_providers"] = sess_list[0].get_providers()
        info["ort_intra_threads"] = args.intra_threads or "default(all cores)"
        info["ort_spinning"] = "disabled" if args.no_spin else "default(enabled)"
    # cooling / hardware
    info["cpu_count"] = os.cpu_count()
    if psutil:
        info["mem_total_gb"] = round(psutil.virtual_memory().total / 1e9, 2)
    gov = _run("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if gov:
        info["cpu_governor"] = gov
    model = _run("cat /proc/device-tree/model")
    if model:
        info["board"] = model.replace("\x00", "")
    fan = _run("cat /sys/class/thermal/cooling_device0/type")
    if fan:
        info["cooling_device0"] = fan
    # OpenCV SIMD -- crop is the bottleneck, so this matters
    try:
        for line in cv2.getBuildInformation().splitlines():
            if "Baseline:" in line and "NEON" in line:
                info["opencv_simd_baseline"] = line.split(":", 1)[1].strip()
    except Exception:
        pass
    return info


# ------------------------------- main -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="clip to stream (e.g. an eyeblink8 .avi)")
    ap.add_argument("--mode", choices=["ours", "ear"], default="ours")
    ap.add_argument("--encoder", default="models/onnx/pipeline.onnx",
                    help="ONNX graph for the infer stage (default: fused pipeline)")
    ap.add_argument("--mlp", default=None,
                    help="optional separate mlp.onnx -> times encode and mlp as two stages")
    ap.add_argument("--minutes", type=float, default=0.0,
                    help="loop the clip until this many minutes elapse (0 = one pass)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=30, help="frames excluded from stats")
    ap.add_argument("--intra-threads", type=int, default=0,
                    help="onnxruntime intra-op threads (0 = default/all cores)")
    ap.add_argument("--no-spin", action="store_true",
                    help="disable onnxruntime thread spin-wait. ORT threads keep "
                         "spinning after a run finishes and steal CPU from the "
                         "next frame's decode/detect; this makes the stages "
                         "independent so the EAR comparison stays honest.")
    ap.add_argument("--refine-landmarks", action="store_true",
                    help="enable MediaPipe iris sub-model (slower; off by default)")
    ap.add_argument("--h", type=int, default=64)
    ap.add_argument("--w", type=int, default=160)
    ap.add_argument("--budget-ms", type=float, default=33.3, help="frame budget (30 fps)")
    ap.add_argument("--csv", default=None, help="per-frame timings")
    ap.add_argument("--json", default=None, help="summary + environment info")
    ap.add_argument("--monitor-period", type=float, default=2.0)
    args = ap.parse_args()

    import cv2
    from src.dataset.eye_preprocess import (
        crop_both_eyes_from_corners, eye_corners_from_landmarks, to_input_tensor)

    if not os.path.exists(args.video):
        sys.exit(f"video not found: {args.video}")

    # ---------------- set up the stages ----------------
    from src.deploy.frontend import EyeFrontend
    fe = EyeFrontend(refine_landmarks=args.refine_landmarks)

    sessions = []
    stage_names = ["read", "detect", "crop"]
    if args.mode == "ours":
        import onnxruntime as ort
        so = ort.SessionOptions()
        if args.intra_threads:
            so.intra_op_num_threads = args.intra_threads
        if args.no_spin:
            so.add_session_config_entry("session.intra_op.allow_spinning", "0")
            so.add_session_config_entry("session.inter_op.allow_spinning", "0")
        enc_sess = ort.InferenceSession(args.encoder, so,
                                        providers=["CPUExecutionProvider"])
        enc_in = enc_sess.get_inputs()[0].name
        sessions.append(enc_sess)
        mlp_sess = mlp_in = None
        if args.mlp:
            mlp_sess = ort.InferenceSession(args.mlp, so,
                                            providers=["CPUExecutionProvider"])
            mlp_in = mlp_sess.get_inputs()[0].name
            sessions.append(mlp_sess)
            stage_names += ["encode", "mlp"]
        else:
            stage_names += ["infer"]
    else:
        from src.dataset.capture_eye_dataset import compute_ear
        stage_names += ["ear"]
        # the crop stage is not part of the EAR path; it is timed as 0 and
        # excluded from e2e below
        stage_names.remove("crop")

    t = {k: [] for k in stage_names}
    e2e = []
    outputs = []
    per_frame = []

    print(f"mode={args.mode}  video={os.path.basename(args.video)}  "
          f"warmup={args.warmup}  "
          + (f"target={args.minutes} min" if args.minutes else "single pass"))
    print("running...", flush=True)

    mon = Monitor(args.monitor_period)
    mon.start()
    temp_start = read_temp_c()
    throttle_start = read_throttled()

    cap = cv2.VideoCapture(args.video)
    n = 0            # frames processed
    n_kept = 0       # frames in the statistics
    n_miss = 0       # frames with no face detected
    loops = 0
    t_wall0 = time.perf_counter()

    try:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            t1 = time.perf_counter()
            if not ok:
                cap.release()
                elapsed_min = (time.perf_counter() - t_wall0) / 60.0
                if args.minutes and elapsed_min < args.minutes:
                    cap = cv2.VideoCapture(args.video)   # loop the clip
                    loops += 1
                    continue
                break

            # -- detect
            lm = fe.landmarks(frame)
            t2 = time.perf_counter()
            if lm is None:
                n_miss += 1
                n += 1
                if _should_stop(args, n, t_wall0):
                    break
                continue

            hh, ww = frame.shape[:2]
            if args.mode == "ours":
                # -- crop (corners + canonical crop + tensor)
                le_a, le_b, re_a, re_b = eye_corners_from_landmarks(lm, ww, hh)
                crop = crop_both_eyes_from_corners(frame, le_a, le_b, re_a, re_b,
                                                   out_h=args.h, out_w=args.w)
                if crop is None:
                    n_miss += 1
                    n += 1
                    if _should_stop(args, n, t_wall0):
                        break
                    continue
                x = to_input_tensor(crop)
                t3 = time.perf_counter()

                # -- infer
                if mlp_sess is None:
                    y = enc_sess.run(None, {enc_in: x})[0]
                    t4 = time.perf_counter()
                    stage_times = [t1 - t0, t2 - t1, t3 - t2, t4 - t3]
                    out_val = float(np.ravel(y)[0])
                else:
                    v = enc_sess.run(None, {enc_in: x})[0]
                    t4 = time.perf_counter()
                    y = mlp_sess.run(None, {mlp_in: v})[0]
                    t5 = time.perf_counter()
                    stage_times = [t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4]
                    out_val = float(np.ravel(y)[0])
            else:
                ear = compute_ear(lm, ww, hh)
                t3 = time.perf_counter()
                stage_times = [t1 - t0, t2 - t1, t3 - t2]
                out_val = float(ear)

            n += 1
            if n > args.warmup:
                ms = [d * 1000.0 for d in stage_times]
                for k, d in zip(stage_names, ms):
                    t[k].append(d)
                total = sum(ms)
                e2e.append(total)
                outputs.append(out_val)
                n_kept += 1
                if args.csv:
                    per_frame.append([n] + [round(v, 4) for v in ms]
                                     + [round(total, 4), round(out_val, 6)])

            if n % 500 == 0:
                el = time.perf_counter() - t_wall0
                tmp = mon.temps[-1] if mon.temps else float("nan")
                print(f"  {n:6d} frames | {el/60:5.2f} min | "
                      f"{n/el:5.1f} fps | {tmp:.1f} C", flush=True)

            if _should_stop(args, n, t_wall0):
                break
    finally:
        cap.release()
        fe.close()
        mon.stop()

    wall = time.perf_counter() - t_wall0
    temp_end = read_temp_c()
    throttle_end = read_throttled()

    # ---------------- report ----------------
    if n_kept == 0:
        sys.exit("no frames measured (all detections failed?)")

    print()
    print("=" * 68)
    print(f"  {args.mode.upper()}  |  {os.path.basename(args.video)}")
    print("=" * 68)
    print(f"  frames processed {n} | measured {n_kept} | detect-miss {n_miss} "
          f"({100*n_miss/max(n,1):.1f}%) | clip loops {loops}")
    print(f"  wall {wall:.1f}s | sustained {n/wall:.2f} fps")
    print()
    print(f"  {'stage':<10}{'p50':>9}{'p95':>9}{'p99':>9}{'max':>9}   (ms)")
    print("  " + "-" * 46)
    for k in stage_names:
        v = t[k]
        print(f"  {k:<10}{pct(v,50):9.2f}{pct(v,95):9.2f}{pct(v,99):9.2f}"
              f"{max(v) if v else float('nan'):9.2f}")
    print("  " + "-" * 46)
    print(f"  {'e2e':<10}{pct(e2e,50):9.2f}{pct(e2e,95):9.2f}{pct(e2e,99):9.2f}"
          f"{max(e2e):9.2f}")
    print()
    p50, p95, p99 = pct(e2e, 50), pct(e2e, 95), pct(e2e, 99)
    for label, val in [("p50", p50), ("p95", p95), ("p99", p99)]:
        status = "WITHIN" if val <= args.budget_ms else "OVER  "
        print(f"  budget {args.budget_ms:.1f} ms @ {label}: {status} "
              f"({val:.2f} ms -> {1000.0/val:.1f} fps)")
    print()
    if mon.temps:
        print(f"  temp   start {temp_start} C | min {min(mon.temps):.1f} | "
              f"max {max(mon.temps):.1f} | end {temp_end} C")
    if mon.cpu:
        print(f"  CPU    mean {statistics.mean(mon.cpu):.0f}% | max {max(mon.cpu):.0f}%")
    if mon.rss:
        print(f"  RSS    peak {max(mon.rss):.0f} MB")
    print(f"  throttled  start={throttle_start}  end={throttle_end}")
    if mon.throttle_seen:
        print(f"  ** FLAGS SEEN: {sorted(mon.throttle_seen)}")
    else:
        print("  no throttle flags observed")

    # ---------------- write ----------------
    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        import csv as _csv
        with open(args.csv, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["frame"] + stage_names + ["e2e_ms", "output"])
            w.writerows(per_frame)
        print(f"\n  per-frame CSV -> {args.csv}")

    summary = {
        "env": env_info(sessions, args),
        "frames_processed": n, "frames_measured": n_kept,
        "detect_miss": n_miss, "clip_loops": loops,
        "wall_s": round(wall, 2), "sustained_fps": round(n / wall, 3),
        "stages_ms": {k: {"p50": round(pct(t[k], 50), 3),
                          "p95": round(pct(t[k], 95), 3),
                          "p99": round(pct(t[k], 99), 3),
                          "max": round(max(t[k]), 3) if t[k] else None}
                      for k in stage_names},
        "e2e_ms": {"p50": round(p50, 3), "p95": round(p95, 3),
                   "p99": round(p99, 3), "max": round(max(e2e), 3)},
        "budget_ms": args.budget_ms,
        "temp_c": {"start": temp_start, "end": temp_end,
                   "min": min(mon.temps) if mon.temps else None,
                   "max": max(mon.temps) if mon.temps else None},
        "cpu_percent_mean": round(statistics.mean(mon.cpu), 1) if mon.cpu else None,
        "rss_mb_peak": round(max(mon.rss), 1) if mon.rss else None,
        "throttled": {"start": throttle_start, "end": throttle_end,
                      "flags_seen": sorted(mon.throttle_seen)},
    }
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  summary JSON -> {args.json}")

    print("\n  environment (record this in the paper):")
    for k, v in summary["env"].items():
        print(f"    {k:<22} {v}")


def _should_stop(args, n, t_wall0):
    if args.max_frames and n >= args.max_frames:
        return True
    if args.minutes and (time.perf_counter() - t_wall0) / 60.0 >= args.minutes:
        return True
    return False


if __name__ == "__main__":
    main()
