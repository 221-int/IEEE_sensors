"""Fixed-rate input replay for Pi latency/power runs (2026-09-28).

`run_video.py` / `run_video_power.py` read a file as fast as possible (free-run),
so their `--fps 30` never limits the workload. This module puts every source
frame on a fixed 30 Hz timeline driven by a monotonic clock and keeps four
quantities apart:

    arrival   a_i = t0 + i / fps          when frame i would leave a 30 fps camera
    ready     r_i = max(a_i, f_{i-1})     earliest moment the pipeline could take it
    start     s_i                         processing actually began (after sleeping)
    finish    f_i                         processing ended

    wait            time slept before s_i (idle; never part of processing)
    backlog         max(0, f_{i-1} - a_i) lateness inherited from the previous frame
    dispatch        s_i - r_i             OS wake-up / loop overhead
    schedule delay  s_i - a_i = backlog + dispatch
    processing      f_i - s_i             read+detect+crop+encode+head
    response        f_i - a_i = schedule delay + processing
    deadline miss   f_i > a_i + budget (default budget = one frame period), or
                    the frame was dropped

Late-frame policies (fixed per protocol, recorded in every result):
    drop     camera semantics: when a newer frame has already arrived, older
             waiting frames are dropped; their time positions stay missing in the
             19-frame window (inference_policy.advance_missing). Primary policy.
    queue    every frame is processed; lateness accumulates as backlog.
    freerun  no timeline; frames processed back to back. Throughput diagnostic
             only; deadlines are not defined and it is never a 30 fps result.

The timeline length is a fixed frame count, and paced runs always last until
the slot of the last frame has elapsed, so every mode covers the same frames
over the same wall time unless it overruns.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

POLICIES = ("drop", "queue", "freerun")
_EPS = 1e-9


class MonotonicClock:
    """perf_counter + sleep (no busy spin: spinning would itself burn power)."""

    def now(self) -> float:
        return time.perf_counter()

    def sleep_until(self, t: float) -> None:
        while True:
            d = t - time.perf_counter()
            if d <= 0:
                return
            time.sleep(d)


class FakeClock:
    """Deterministic clock for tests; `advance` models processing time."""

    def __init__(self, t: float = 0.0):
        self.t = float(t)

    def now(self) -> float:
        return self.t

    def sleep_until(self, t: float) -> None:
        self.t = max(self.t, float(t))

    def advance(self, dt: float) -> None:
        self.t += float(dt)


@dataclass
class Slot:
    index: int
    arrival: float
    ready: float
    start: float
    wait: float
    skipped: list[int] = field(default_factory=list)


class FrameScheduler:
    def __init__(self, fps: float, n_frames: int, policy: str, clock, budget_s: float | None = None):
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        if fps <= 0 or n_frames <= 0:
            raise ValueError("fps and n_frames must be positive")
        self.fps, self.n, self.policy, self.clock = float(fps), int(n_frames), policy, clock
        self.period = 1.0 / self.fps
        self.budget = self.period if budget_s is None else float(budget_s)
        self.t0: float | None = None
        self.t_end: float | None = None
        self.next_index = 0
        self.prev_finish: float | None = None
        nan = np.full(self.n, np.nan)
        self.arrival, self.ready, self.start, self.finish, self.wait = (nan.copy() for _ in range(5))
        self.dropped = np.zeros(self.n, bool)
        self.processed = np.zeros(self.n, bool)

    # -------------------------------------------------------------- timeline
    def begin(self) -> float:
        self.t0 = self.clock.now()
        return self.t0

    def arrival_time(self, i: int) -> float:
        return self.t0 + i * self.period

    def acquire(self) -> Slot | None:
        """Wait for the next frame to process; None when the timeline is over."""
        if self.t0 is None:
            raise RuntimeError("begin() first")
        k = self.next_index
        if k >= self.n:
            return None
        now = self.clock.now()
        skipped: list[int] = []
        if self.policy == "drop":
            latest = int(math.floor((now - self.t0) / self.period + _EPS))
            if latest > k:
                if latest >= self.n:          # every remaining frame expired unseen
                    self.dropped[k:] = True
                    self.next_index = self.n
                    return None
                skipped = list(range(k, latest))
                self.dropped[k:latest] = True
                k = latest
        if self.policy == "freerun":
            arrival = ready = now
        else:
            arrival = self.arrival_time(k)
            ready = arrival if self.prev_finish is None else max(arrival, self.prev_finish)
            self.clock.sleep_until(arrival)
        start = self.clock.now()
        slot = Slot(k, arrival, ready, start, max(0.0, start - now), skipped)
        self.arrival[k], self.ready[k], self.start[k], self.wait[k] = arrival, ready, start, slot.wait
        return slot

    def complete(self, slot: Slot, finish: float | None = None) -> float:
        f = self.clock.now() if finish is None else float(finish)
        self.finish[slot.index] = f
        self.processed[slot.index] = True
        self.prev_finish = f
        self.next_index = slot.index + 1
        return f

    def close(self) -> float:
        """Paced runs last until the last frame slot has elapsed (equal windows)."""
        if self.policy != "freerun":
            self.clock.sleep_until(self.t0 + self.n * self.period)
        self.t_end = self.clock.now()
        return self.t_end

    # -------------------------------------------------------------- summary
    def per_frame(self) -> dict[str, np.ndarray]:
        backlog = np.full(self.n, np.nan)
        prev = None
        for i in range(self.n):
            if self.processed[i]:
                backlog[i] = 0.0 if prev is None else max(0.0, prev - self.arrival[i])
                prev = self.finish[i]
        late = np.zeros(self.n, bool)
        if self.policy != "freerun":
            late = self.processed & (self.finish > self.arrival + self.budget + _EPS)
        return dict(arrival=self.arrival, ready=self.ready, start=self.start, finish=self.finish,
                    wait=self.wait, backlog=backlog, dropped=self.dropped,
                    processed=self.processed, late=late)

    def summary(self) -> dict:
        pf = self.per_frame()
        p = pf["processed"]
        ms = lambda a: (a[p] * 1e3).tolist()
        duration = (self.t_end - self.t0) if self.t_end is not None else None
        processing = pf["finish"] - pf["start"]
        out = {
            "policy": self.policy, "fps_nominal": self.fps, "period_ms": self.period * 1e3,
            "n_timeline_frames": self.n, "n_processed": int(p.sum()),
            "n_dropped": int(pf["dropped"].sum()),
            "n_not_reached": int(self.n - p.sum() - pf["dropped"].sum()),
            "duration_s": duration,
            "nominal_duration_s": None if self.policy == "freerun" else self.n * self.period,
            "processed_per_s": (float(p.sum()) / duration) if duration else None,
            "processing_ms": summarize_ms(ms(processing)),
            "wait_ms_total": float(np.nansum(pf["wait"]) * 1e3),
            "busy_fraction": (float(np.nansum(processing[p])) / duration) if duration else None,
            "n_overrun_processing_gt_period": int((processing[p] > self.period + _EPS).sum()),
        }
        if self.policy == "freerun":
            out.update(deadline="not defined for freerun", budget_ms=None)
            return out
        miss = pf["late"] | pf["dropped"]
        run = best = 0
        for m in miss:
            run = run + 1 if m else 0
            best = max(best, run)
        out.update(
            budget_ms=self.budget * 1e3,
            schedule_delay_ms=summarize_ms(ms(pf["start"] - pf["arrival"])),
            backlog_ms=summarize_ms(ms(pf["backlog"])),
            dispatch_ms=summarize_ms(ms(pf["start"] - pf["ready"])),
            response_ms=summarize_ms(ms(pf["finish"] - pf["arrival"])),
            n_late_completed=int(pf["late"].sum()),
            n_deadline_miss=int(miss.sum()),
            deadline_miss_rate=float(miss.mean()),
            max_consecutive_miss=int(best),
        )
        return out


def summarize_ms(v) -> dict:
    a = np.asarray(v, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)),
            "max": float(a.max())}


class LoopingVideoSource:
    """Sequential clip reader that restarts at EOF and marks the seam.

    Reopening (instead of seeking) avoids codec-dependent seek behaviour. Every
    completed pass is length-checked against `expected_frames` when given.
    """

    def __init__(self, path: str, expected_frames: int | None = None, opener=None):
        if opener is None:
            import cv2
            opener = cv2.VideoCapture
        self.path, self.expected, self._opener = path, expected_frames, opener
        self.loop_lengths: list[int] = []
        self.cap = None
        self.restart()

    def _open(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = self._opener(self.path)
        if not self.cap.isOpened():
            raise IOError(f"cannot open source: {self.path}")

    def restart(self) -> None:
        self._open()
        self.loop, self.pos = 0, 0

    def _next(self, grab_only: bool):
        seam = False
        for _ in range(2):
            if grab_only:
                ok, frame = self.cap.grab(), None
            else:
                ok, frame = self.cap.read()
            if ok:
                info = {"loop": self.loop, "pos": self.pos, "seam": seam}
                self.pos += 1
                return frame, info
            if self.pos == 0:
                raise IOError(f"source yielded no frames: {self.path}")
            self.loop_lengths.append(self.pos)
            self._open()
            self.loop += 1
            self.pos = 0
            seam = True
        raise IOError("source could not be restarted")

    def read(self):
        return self._next(False)

    def grab(self):
        return self._next(True)[1]

    def length_mismatches(self) -> list[int]:
        if self.expected is None:
            return []
        return [n for n in self.loop_lengths if n != self.expected]

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
