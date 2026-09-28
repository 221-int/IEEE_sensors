"""Tests for the paced 30 fps harness (no Pi, camera or MediaPipe needed).

    python -m unittest src.v2.deploy.test_paced_bench -v
"""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.v2.deploy import inference_policy as IP
from src.v2.deploy.bench_telemetry import LoggingPowerSampler, ThermalSampler, decode_throttle
from src.v2.deploy.check_paced_results import aggregate, qc_run
from src.v2.deploy.paced_session import make_plan
from src.v2.deploy.pacing import FakeClock, FrameScheduler, LoopingVideoSource
from src.v2.deploy.run_paced import Pipeline, execute, power_summary

P = 1 / 30

# Real `vcgencmd pmic_read_adc` text captured on the Pi 5 in the 2026 power run
# (results/v2/power_ours_480p_run1.json power_sampling.pmic_raw_sample_first).
PI_RAW = Path(__file__).resolve().parents[3] / "results/v2/power_ours_480p_run1.json"


def drive(sched, clock, proc):
    """proc(index) -> processing seconds; returns processed indices."""
    sched.begin()
    done = []
    while (slot := sched.acquire()) is not None:
        if hasattr(clock, "advance"):
            clock.advance(proc(slot.index))
        sched.complete(slot)
        done.append(slot.index)
    sched.close()
    return done


class SchedulerTests(unittest.TestCase):
    def test_on_time_frames_wait_and_never_miss(self):
        c = FakeClock(100.0)
        s = FrameScheduler(30, 30, "drop", c)
        done = drive(s, c, lambda i: 0.010)
        m = s.summary()
        self.assertEqual(done, list(range(30)))
        self.assertEqual((m["n_dropped"], m["n_deadline_miss"]), (0, 0))
        self.assertAlmostEqual(m["response_ms"]["max"], 10.0, places=6)
        self.assertAlmostEqual(m["schedule_delay_ms"]["max"], 0.0, places=6)
        self.assertAlmostEqual(m["duration_s"], 1.0, places=9)        # waits for the last slot
        self.assertAlmostEqual(m["wait_ms_total"], 29 * (P - 0.010) * 1e3, places=6)
        self.assertAlmostEqual(m["busy_fraction"], 0.3, places=6)

    def test_overrun_is_backlog_not_processing(self):
        c = FakeClock()
        s = FrameScheduler(30, 10, "drop", c)
        drive(s, c, lambda i: 0.050 if i == 5 else 0.010)
        pf, m = s.per_frame(), s.summary()
        self.assertEqual(m["n_dropped"], 0)
        self.assertTrue(pf["late"][5])                                  # 50 ms > 33.3 ms
        self.assertAlmostEqual(pf["backlog"][6], 5 * P + 0.050 - 6 * P, places=9)
        self.assertAlmostEqual(pf["finish"][6] - pf["start"][6], 0.010, places=9)
        self.assertFalse(pf["late"][6])
        self.assertEqual(m["n_deadline_miss"], 1)

    def test_drop_policy_skips_stale_frames(self):
        c = FakeClock()
        s = FrameScheduler(30, 12, "drop", c)
        done = drive(s, c, lambda i: 0.080 if i == 5 else 0.010)       # finish at 246.7 ms
        self.assertNotIn(6, done)
        self.assertIn(7, done)
        m = s.summary()
        self.assertEqual(m["n_dropped"], 1)
        self.assertEqual(m["n_deadline_miss"], 2)                       # 5 late + 6 dropped
        self.assertEqual(m["max_consecutive_miss"], 2)
        self.assertEqual(m["n_processed"] + m["n_dropped"], 12)

    def test_queue_policy_processes_everything(self):
        c = FakeClock()
        s = FrameScheduler(30, 12, "queue", c)
        done = drive(s, c, lambda i: 0.080 if i == 5 else 0.010)
        self.assertEqual(done, list(range(12)))
        pf = s.per_frame()
        self.assertGreater(pf["backlog"][6], 0)
        self.assertEqual(s.summary()["n_dropped"], 0)

    def test_drop_at_end_of_timeline(self):
        c = FakeClock()
        s = FrameScheduler(30, 5, "drop", c)
        done = drive(s, c, lambda i: 0.5 if i == 1 else 0.01)
        self.assertEqual(done, [0, 1])
        self.assertEqual(s.summary()["n_dropped"], 3)
        self.assertEqual(s.summary()["n_not_reached"], 0)

    def test_freerun_has_no_deadline(self):
        c = FakeClock()
        s = FrameScheduler(30, 10, "freerun", c)
        drive(s, c, lambda i: 0.005)
        m = s.summary()
        self.assertAlmostEqual(m["duration_s"], 0.05, places=9)
        self.assertNotIn("n_deadline_miss", m)
        self.assertIsNone(m["nominal_duration_s"])

    def test_real_clock_does_not_count_sleep_as_processing(self):
        from src.v2.deploy.pacing import MonotonicClock
        c = MonotonicClock()
        s = FrameScheduler(30, 6, "drop", c)
        drive(s, c, lambda i: 0.0)
        m = s.summary()
        self.assertLess(m["processing_ms"]["max"], 5.0)
        self.assertGreater(m["duration_s"], 0.19)


class FakeCap:
    def __init__(self, n):
        self.n, self.i = n, 0

    def isOpened(self):
        return True

    def read(self):
        if self.i >= self.n:
            return False, None
        self.i += 1
        return True, np.full((4, 4, 3), self.i - 1, np.uint8)

    def grab(self):
        return self.read()[0]

    def release(self):
        pass


class SourceTests(unittest.TestCase):
    def test_loop_seams_and_length_check(self):
        src = LoopingVideoSource("x", expected_frames=5, opener=lambda p: FakeCap(5))
        infos = [src.read()[1] for _ in range(12)]
        self.assertEqual([i["seam"] for i in infos].count(True), 2)
        self.assertTrue(infos[5]["seam"] and infos[5]["pos"] == 0 and infos[5]["loop"] == 1)
        self.assertEqual(src.loop_lengths, [5, 5])
        self.assertEqual(src.length_mismatches(), [])
        bad = LoopingVideoSource("x", expected_frames=4, opener=lambda p: FakeCap(5))
        for _ in range(6):
            bad.read()
        self.assertEqual(bad.length_mismatches(), [5])

    def test_empty_source_raises(self):
        with self.assertRaises(IOError):
            LoopingVideoSource("x", opener=lambda p: FakeCap(0)).read()


class FakeFrontend:
    """mesh None for frames whose first pixel is in `no_face`; else a dummy mesh."""

    def __init__(self, clock, no_face=(), cost=0.004):
        self.clock, self.no_face, self.cost = clock, set(no_face), cost

    def mesh(self, frame):
        self.clock.advance(self.cost)
        return None if int(frame[0, 0, 0]) in self.no_face else np.zeros((468, 2), np.float32)

    def crop_from_mesh(self, frame, mesh):
        return np.full((64, 160), 128, np.uint8) + np.arange(160, dtype=np.uint8)[None] % 7, {}


class FakeSess:
    def __init__(self, clock, fn, cost):
        self.clock, self.fn, self.cost = clock, fn, cost
        self.calls = []

    def run(self, _, feed):
        self.clock.advance(self.cost)
        self.calls.append({k: np.array(v) for k, v in feed.items()})
        return [self.fn(feed)]


class NullTelemetry:
    def __init__(self):
        self.samples, self.raw, self.ext5v = [], [], []
        self.parse_errors, self.missing_rails, self.first_raw_sample = 0, set(), None

    def start(self):
        pass

    def stop(self):
        pass

    def window(self, t0, t1):
        return {"n": 0, "duration_s": t1 - t0}


CONTRACT = {"variant": "vpres", "threshold_probability": 0.5}


def fake_pipeline(clock, no_face=(), head_p=0.9):
    enc = FakeSess(clock, lambda f: np.ones((1, 16), np.float32), 0.002)
    head = FakeSess(clock, lambda f: np.array([[head_p]], np.float32), 0.001)
    return Pipeline("ours_vpres", FakeFrontend(clock, no_face), enc, head, CONTRACT, now=clock.now), head


class PipelineTests(unittest.TestCase):
    CFG = dict(fps=30.0, pacing="drop", budget_ms=1000 / 30, warmup_s=1.0, settle_s=1.0,
               idle_pre_s=2.0, active_s=4.0, idle_post_s=1.0)

    def test_full_run_counts_seams_and_missing_positions(self):
        c = FakeClock()
        pipe, head = fake_pipeline(c, no_face={3})
        src = LoopingVideoSource("x", expected_frames=50, opener=lambda p: FakeCap(50))
        res = execute(self.CFG, pipe, src, c, NullTelemetry(), NullTelemetry())
        pc, oc = res["pacing"], res["outcomes"]
        self.assertEqual(pc["n_timeline_frames"], 120)
        self.assertEqual((pc["n_processed"], pc["n_dropped"], pc["n_deadline_miss"]), (120, 0, 0))
        self.assertAlmostEqual(pc["response_ms"]["max"], 7.0, places=6)   # 4+2+1 ms fake costs
        self.assertEqual(oc["seam_timeline_indices"], [50, 100])
        self.assertEqual(res["loop_lengths"], [50, 50])
        # clip frame 3 has no face in each pass -> 3 missing positions, never a head input
        self.assertEqual(oc["n_processed"] - oc["face"], 3)
        for call in head.calls:
            self.assertEqual(call["mask"].shape, (1, IP.EVENT_LEN))
            self.assertTrue(bool(IP.eligible_mask(call["mask"][0] > 0)))
        # each pass (active starts at clip frame 0): ring resets at the seam, 19 frames to fill
        dec = res["rec"]["decision"]
        self.assertFalse(dec[50:68].any())
        self.assertTrue(dec[68])
        # one onset per contiguous positive stretch; seams and the no-face frame break it
        self.assertEqual(oc["onset"], 3)
        ph = {e["phase"] for e in res["log"].entries}
        self.assertEqual(ph, {"warmup", "settle", "idle_pre", "active", "idle_post"})
        w = res["log"].window("active")
        self.assertAlmostEqual(w["duration_s"], 4.0, places=6)
        self.assertAlmostEqual(res["log"].window("idle_pre")["duration_s"], 2.0, places=9)

    def test_dropped_frames_become_missing_positions(self):
        c = FakeClock()
        pipe, head = fake_pipeline(c)
        slow = FakeFrontend(c, cost=0.004)
        orig = slow.mesh

        def mesh(frame):
            if int(frame[0, 0, 0]) == 30:
                c.advance(0.12)
            return orig(frame)
        slow.mesh = mesh
        pipe.fe = slow
        src = LoopingVideoSource("x", opener=lambda p: FakeCap(500))
        cfg = dict(self.CFG, warmup_s=0.0)
        res = execute(cfg, pipe, src, c, NullTelemetry(), NullTelemetry())
        pc = res["pacing"]
        self.assertEqual(pc["n_dropped"], 2)
        self.assertEqual(pc["n_processed"] + pc["n_dropped"], 120)
        # the source still advanced over the dropped frames (clip time = timeline time)
        pos = res["rec"]["pos"]
        idx = np.flatnonzero(res["per_frame"]["processed"])
        self.assertTrue(np.array_equal(pos[idx], idx))
        # the first head call after the drop sees the dropped slots as masked
        masks = [call["mask"][0] for call in head.calls]
        self.assertTrue(any(m.sum() < IP.EVENT_LEN for m in masks))

    def test_ear_mode_scores_without_threshold(self):
        c = FakeClock()

        class EarFE(FakeFrontend):
            pass
        pipe = Pipeline("ear_rule", EarFE(c), now=c.now)
        import src.v2.dataset.crop as C
        orig = C.ear_both
        C.ear_both = lambda mesh: {"mean": 0.3}
        try:
            src = LoopingVideoSource("x", opener=lambda p: FakeCap(500))
            res = execute(dict(self.CFG, warmup_s=0.0), pipe, src, c, NullTelemetry(), NullTelemetry())
        finally:
            C.ear_both = orig
        self.assertEqual(res["outcomes"]["onset"], 0)
        self.assertEqual(res["outcomes"]["decision"], 120 - 18)

    def test_cnn_mode_requires_contract(self):
        with self.assertRaises(ValueError):
            Pipeline("ours_vpres", FakeFrontend(FakeClock()), None, None, None)


class TelemetryTests(unittest.TestCase):
    def test_real_pi_pmic_text_parses_all_rails(self):
        if not PI_RAW.exists():
            self.skipTest("historical Pi result not present")
        raw = json.loads(PI_RAW.read_text(encoding="utf8"))["power_sampling"]["pmic_raw_sample_first"]
        s = LoggingPowerSampler(hz=10, cmd_fn=lambda: raw)
        s._sample_once()
        s._sample_once()
        self.assertEqual((s.parse_errors, s.missing_rails), (0, set()))
        self.assertEqual(len(s.raw), 2)
        self.assertAlmostEqual(s.samples[0][1], 2.0, delta=0.6)        # idle-ish 2 W rail sum
        self.assertAlmostEqual(s.ext5v[0][1], 5.05984, places=5)

    def test_power_window_and_summary(self):
        s = LoggingPowerSampler(hz=10, cmd_fn=lambda: None)
        s.samples = [(t / 10, 2.0 if t < 50 else 4.0) for t in range(100)]
        w = s.window(0.0, 4.95)
        self.assertEqual(w["n"], 50)
        self.assertAlmostEqual(w["mean_w"], 2.0)

        class Log:
            def window(self, ph):
                return {"idle_pre": {"t0": 0.0, "t1": 4.95}, "active": {"t0": 5.0, "t1": 9.95},
                        "idle_post": None, "warmup": None}[ph]
        p = power_summary(s, Log(), n_timeline=150, n_processed=150)
        self.assertAlmostEqual(p["delta_w"], 2.0)
        self.assertAlmostEqual(p["incremental_mj_per_timeline_frame"], 2.0 * 4.95 / 150 * 1e3)

    def test_throttle_decode_separates_now_and_sticky(self):
        d = decode_throttle(0x50005)
        self.assertEqual(d["now"], ["under-voltage", "throttled"])
        self.assertEqual(d["sticky"], ["under-voltage occurred", "throttling occurred"])
        self.assertEqual(decode_throttle(0)["raw"], "0x0")

    def test_thermal_window(self):
        vals = iter([50.0, 51.0, 53.0])
        th = ThermalSampler(readers={"temp": lambda: next(vals), "throttled": lambda: 0x4,
                                     "arm": lambda: 2400000000, "fan": lambda: {"rpm": 3000},
                                     "rss": lambda: 100.0})
        for _ in range(3):
            th.sample_once()
        w = th.window(-1e9, 1e12)
        self.assertEqual(w["temp_c"]["max"], 53.0)
        self.assertEqual(w["throttle_now_flags"], ["throttled"])


def good_result():
    return {
        "schema": "paced_run_v1", "run_id": "P_b01_ours_vpres", "mode": "ours_vpres", "block": 1,
        "config": {"fps": 30.0, "pacing": "drop", "budget_ms": 1000 / 30, "intra_threads": 2,
                   "no_spin": True, "power_hz": 7.5},
        "protocol": {"sha256": "abc", "set": "primary"},
        "env": {"machine": "aarch64"},
        "platform_start": {"cmdline_has_vcgencmd": True, "governors": {"cpu0": "performance"},
                           "throttled": {"sticky": []}},
        "platform_end": {"governors": {"cpu0": "performance"}, "throttled": {"sticky": []}},
        "gates": {"bundle": {"passed": True}, "equivalence": {"authorized": True}},
        "thermal": {"active": {"throttle_now_flags": []}}, "thermal_samples": [{}],
        "power": {"parse_errors": 0, "missing_rails": [], "idle_pre_w": 2.0, "idle_drift_w": 0.01,
                  "delta_w": 1.0, "windows": {ph: {"n": 400, "effective_hz": 7.0, "max_gap_s": 0.3}
                                              for ph in ("idle_pre", "active", "idle_post")}},
        "clip": {"sha256": "h", "expected_sha256": "h", "fps_meta": 30.0}, "loop_mismatches": [],
        "pacing": {"policy": "drop", "n_dropped": 0, "n_not_reached": 0, "n_deadline_miss": 0,
                   "budget_ms": 33.3, "response_ms": {"p99": 20.0}},
    }


class QCTests(unittest.TestCase):
    def test_clean_run_is_reportable(self):
        q = qc_run(good_result())
        self.assertTrue(q["reportable"], q)
        self.assertTrue(q["realtime_pass"] and q["workload_identical"])

    def test_desktop_and_throttle_and_pmic_failures(self):
        for mutate, needle in [
            (lambda r: r["env"].update(machine="AMD64"), "not a Raspberry Pi"),
            (lambda r: r["platform_start"]["governors"].update(cpu1="ondemand"), "governor"),
            (lambda r: r["thermal"]["active"].update(throttle_now_flags=["throttled"]), "throttle"),
            (lambda r: r["platform_end"]["throttled"].update(sticky=["throttling occurred"]), "sticky"),
            (lambda r: r["power"].update(parse_errors=3), "PMIC"),
            (lambda r: r["power"]["windows"]["active"].update(max_gap_s=5.0), "gap"),
            (lambda r: r.update(loop_mismatches=[5100]), "pass length"),
            (lambda r: r["gates"]["bundle"].update(passed=False), "bundle"),
        ]:
            r = good_result()
            mutate(r)
            q = qc_run(r)
            self.assertFalse(q["reportable"], needle)
            self.assertTrue(any(needle in h for h in q["hard_fail"]), (needle, q["hard_fail"]))

    def test_misses_are_results_not_exclusions(self):
        r = good_result()
        r["pacing"].update(n_dropped=4, n_deadline_miss=9)
        q = qc_run(r)
        self.assertTrue(q["reportable"])
        self.assertFalse(q["realtime_pass"] or q["workload_identical"])

    def test_protocol_mismatch(self):
        proto = {"run": {"fps": 30.0, "pacing": "drop", "power_hz": 7.5},
                 "sets": {"primary": {"modes": ["ours_vpres"]}}}
        self.assertTrue(qc_run(good_result(), proto, "abc")["reportable"])
        r = good_result()
        r["config"]["power_hz"] = 5.0
        self.assertFalse(qc_run(r, proto, "abc")["reportable"])
        self.assertFalse(qc_run(good_result(), proto, "other")["reportable"])
        r = good_result()
        r["mode"] = "vdrop"
        self.assertFalse(qc_run(r, proto, "abc")["reportable"])

    def test_aggregate_pairs_within_blocks_and_lists_exclusions(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for b in (1, 2):
                for mode, dw in (("ours_vpres", 1.0), ("image_cnn_head", 1.5)):
                    r = copy.deepcopy(good_result())
                    r.update(run_id=f"P_b{b:02d}_{mode}", mode=mode, block=b)
                    r["power"]["delta_w"] = dw + 0.1 * b
                    (d / r["run_id"]).mkdir()
                    (d / r["run_id"] / "run.json").write_text(json.dumps(r))
            r = copy.deepcopy(good_result())
            r.update(run_id="P_b03_ours_vpres", block=3)
            r["env"]["machine"] = "AMD64"
            (d / r["run_id"]).mkdir()
            (d / r["run_id"] / "run.json").write_text(json.dumps(r))
            s = aggregate(d)
            self.assertEqual((s["n_runs_reportable"], s["n_runs_excluded"]), (4, 1))
            pd = s["paired_within_block"]["image_cnn_head - ours_vpres"]["delta_w"]
            self.assertEqual(pd["n"], 2)
            self.assertAlmostEqual(pd["mean"], 0.5)


class CliTests(unittest.TestCase):
    def test_no_spin_flag_really_disables_spinning(self):
        from src.v2.deploy.run_paced import build_parser
        base = ["--mode", "ear_rule", "--bundle-root", "b", "--source", "s", "--out-dir", "o", "--run-id", "r"]
        ap = build_parser()
        self.assertTrue(ap.parse_args(base).no_spin)
        self.assertTrue(ap.parse_args(base + ["--no-spin"]).no_spin)
        self.assertFalse(ap.parse_args(base + ["--allow-spin"]).no_spin)


class PlanTests(unittest.TestCase):
    def test_blocks_contain_each_mode_once_and_are_seeded(self):
        modes = ["ours_vpres", "image_cnn_head", "ear_rule"]
        a, b = make_plan(modes, 5, 7, "P_"), make_plan(modes, 5, 7, "P_")
        self.assertEqual(a, b)
        for blk in range(1, 6):
            self.assertEqual(sorted(x["mode"] for x in a if x["block"] == blk), sorted(modes))
        self.assertEqual([x["order_index"] for x in a], list(range(15)))
        self.assertEqual(len({x["run_id"] for x in a}), 15)


if __name__ == "__main__":
    unittest.main()
