"""experiment — 피험자별 캘리브레이션 실험 기록·통계 (박사님 프로토콜).

절차(피험자별):
  1) 편안히 눈 뜨고 OPEN_SEC 초   -> 개안 EAR 분포 -> baseline
  2) 눈 감고 CLOSED_SEC 초        -> 폐안 EAR 분포 -> closed_floor
  3) "지금 깜빡" 지시마다 STD 가 정확히 1회로 세는지 TRIALS 회 검증
  4) 모든 프레임 EAR + 시행 결과 + 요약 통계를 CSV 로 기록(통계 분석용)

임계값 계산·시행 카운트 로직은 MediaPipe/OpenCV 없이 단위 테스트 가능.
"""
import csv
import os

import numpy as np

from . import config
from .std import BlinkSTD


class CalibrationRecorder:
    def __init__(self, subject,
                 open_pct=config.BASELINE_PERCENTILE,
                 closed_pct=config.CLOSED_PERCENTILE):
        self.subject = subject
        self.open_pct = open_pct
        self.closed_pct = closed_pct
        self._open, self._closed = [], []
        self.baseline = None
        self.closed_floor = None
        self.t_open = None
        self.t_closed = None
        self.frames = []       # (subject, phase, time, ear)
        self.trials = []       # (subject, trial, expected, detected, correct)
        self._cur = None       # 진행 중 시행 상태

    # ── phase 1/2: 개안·폐안 EAR 수집 ──────────────────────────────────────
    def add_open(self, t, ear):
        self._open.append(ear)
        self.frames.append((self.subject, "open", t, ear))

    def add_closed(self, t, ear):
        self._closed.append(ear)
        self.frames.append((self.subject, "closed", t, ear))

    def finalize_thresholds(self):
        """개안/폐안 분포로 baseline·floor·STD 임계값 확정."""
        self.baseline = float(np.percentile(self._open, self.open_pct)) if self._open else None
        self.closed_floor = float(np.percentile(self._closed, self.closed_pct)) if self._closed else None
        b = self.baseline or 0.0
        f = self.closed_floor or 0.0
        band = max(b - f, 1e-6)
        self.t_closed = f + config.STD_CLOSE_FRAC * band
        self.t_open = f + config.STD_OPEN_FRAC * band
        return self.summary_calib()

    def new_std(self):
        return BlinkSTD(self.baseline, self.closed_floor)

    # ── phase 3: 깜빡임 검증 시행 (라이브: begin/feed/end) ──────────────────
    def begin_trial(self, trial_idx):
        self._cur = {"idx": trial_idx, "std": self.new_std(), "detected": 0}

    def feed_trial(self, t, ear):
        c = self._cur
        self.frames.append((self.subject, f"trial{c['idx']}", t, ear))
        if c["std"].update(t, ear):
            c["detected"] += 1
            return True
        return False

    def end_trial(self, expected=1):
        c = self._cur
        correct = int(c["detected"] == expected)
        self.trials.append((self.subject, c["idx"], expected, c["detected"], correct))
        self._cur = None
        return c["detected"], correct

    def record_trial(self, trial_idx, ear_stream, expected=1):
        """오프라인/테스트용: (t,ear) 스트림 하나로 시행 1회 처리."""
        self.begin_trial(trial_idx)
        for t, e in ear_stream:
            self.feed_trial(t, e)
        return self.end_trial(expected)

    # ── 통계 요약 ──────────────────────────────────────────────────────────
    def summary_calib(self):
        def ms(a):
            return (float(np.mean(a)), float(np.std(a))) if a else (None, None)
        om, osd = ms(self._open)
        cm, csd = ms(self._closed)
        return {"subject": self.subject, "baseline": self.baseline,
                "closed_floor": self.closed_floor, "t_open": self.t_open,
                "t_closed": self.t_closed, "open_ear_mean": om, "open_ear_std": osd,
                "closed_ear_mean": cm, "closed_ear_std": csd,
                "n_open": len(self._open), "n_closed": len(self._closed)}

    def summary_trials(self):
        n = len(self.trials)
        correct = sum(c for *_, c in self.trials)
        return {"subject": self.subject, "trials": n, "correct": correct,
                "accuracy": (correct / n) if n else 0.0}


# ── CSV 기록 ────────────────────────────────────────────────────────────────
def write_frames_csv(rec, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "phase", "time", "ear"])
        w.writerows(rec.frames)


def write_trials_csv(rec, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "trial", "expected", "detected", "correct"])
        w.writerows(rec.trials)


def append_summary_csv(rec, path):
    c = rec.summary_calib()
    tr = rec.summary_trials()
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["subject", "baseline", "closed_floor", "t_open", "t_closed",
                        "open_ear_mean", "open_ear_std", "closed_ear_mean", "closed_ear_std",
                        "n_open", "n_closed", "trials", "correct", "accuracy"])
        w.writerow([c["subject"], c["baseline"], c["closed_floor"], c["t_open"], c["t_closed"],
                    c["open_ear_mean"], c["open_ear_std"], c["closed_ear_mean"], c["closed_ear_std"],
                    c["n_open"], c["n_closed"], tr["trials"], tr["correct"], tr["accuracy"]])
