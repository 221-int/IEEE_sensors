"""4개 모드(ours/ear/image_cnn_max/image_cnn_head) 전력 측정을 반복 실행하고
run별 JSON을 모아 summary.json 을 만드는 러너.

§2-4 요구사항을 여기서 강제한다:
  - 동일 입력 영상 · 동일 프레임 수 (전체 호출에 동일 --source/--duration)
  - 실행 순서 무작위화
  - 실행 간 쿨다운 60초 이상
  - 4모드 스레드 수·onnxruntime 설정 동일 (동일 --intra-threads/--no-spin 전달)
  - 폴링 주기 동일 (동일 --power-hz 전달)

`run_video_power.py` 자체는 건드리지 않는다 — 이 파일은 그걸 반복 호출하는
별도 유틸이다.

사용 예 (Pi 에서):
    python -m src.v2.deploy.power_bench_runner \
        --source 0 --width 640 --height 480 --duration 300 \
        --intra-threads 2 --no-spin --power-hz 7.5 --idle-seconds 30 \
        --repeats 3 --cooldown 60

⚠️ 이 스크립트도 커밋하지 말 것 (전력측정 작업 지시 §2-6).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import subprocess
import sys
import time

import numpy as np

MODES = ["ours", "ear", "image_cnn_max", "image_cnn_head",
        "image_cnn_bilstm", "image_cnn_lstm"]
RESULTS_DIR = "results/v2"


def run_one(mode: str, run_index: int, common_args: list[str], python_exe: str) -> str:
    out_path = os.path.join(RESULTS_DIR, f"power_{mode}_480p_run{run_index}.json")
    cmd = [python_exe, "-m", "src.v2.deploy.run_video_power",
           "--mode", mode, "--run-index", str(run_index), "--out", out_path,
           *common_args]
    print(f"\n=== {mode} run{run_index} ===\n  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return out_path


def aggregate(mode: str, repeats: int) -> dict:
    run_files = sorted(glob.glob(os.path.join(RESULTS_DIR, f"power_{mode}_480p_run*.json")))
    runs = []
    for p in run_files:
        try:
            runs.append((p, json.load(open(p, encoding="utf-8"))))
        except Exception as e:
            print(f"  ⚠️ {p} 읽기 실패: {e}")

    kept = [(p, r) for p, r in runs if not r.get("discard_recommended", False)]
    discarded = [p for p, r in runs if r.get("discard_recommended", False)]

    def field_stats(key: str, nested: str | None = None) -> dict:
        vals = []
        for _, r in kept:
            v = r.get(nested, {}).get(key) if nested else r.get(key)
            if v is not None:
                vals.append(v)
        if not vals:
            return {"mean": None, "std": None, "n": 0}
        a = np.asarray(vals, dtype=np.float64)
        return {"mean": float(a.mean()), "std": float(a.std(ddof=0)) if len(a) > 1 else 0.0,
                "n": int(a.size)}

    summary = {
        "mode": mode,
        "n_runs_found": len(runs),
        "n_runs_kept": len(kept),
        "n_runs_discarded": len(discarded),
        "discarded_run_files": discarded,
        "run_files": [p for p, _ in runs],
        "power_idle_w": field_stats("power_idle_w"),
        "power_active_w": field_stats("power_active_w"),
        "power_delta_w": field_stats("power_delta_w"),
        "energy_per_frame_mJ": field_stats("energy_per_frame_mJ"),
        "battery_hours_at_delta_power": field_stats(
            "battery_hours_at_delta_power", nested="battery_estimate"),
        "_warning": (None if len(kept) >= repeats else
                     f"유효 run 이 {len(kept)}개뿐 (목표 {repeats}). "
                     "스로틀/파싱실패로 폐기된 run 이 있는지 확인할 것"),
    }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--duration", type=float, default=300.0)
    ap.add_argument("--intra-threads", type=int, default=2)
    ap.add_argument("--no-spin", action="store_true", default=True)
    ap.add_argument("--power-hz", type=float, default=7.5)
    ap.add_argument("--idle-seconds", type=float, default=30.0)
    ap.add_argument("--battery-mah", type=float, default=5000.0)
    ap.add_argument("--battery-cell-v", type=float, default=3.7)
    ap.add_argument("--battery-eta", type=float, default=0.87)
    ap.add_argument("--repeats", type=int, default=3, help="모드당 반복 횟수 (최소 3)")
    ap.add_argument("--cooldown", type=float, default=60.0, help="run 사이 쿨다운(초). 60s 이상")
    ap.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    ap.add_argument("--seed", type=int, default=None, help="실행 순서 무작위화 시드(재현용)")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    if args.cooldown < 60:
        print(f"⚠️ --cooldown {args.cooldown}s < 60s. 발열 누적으로 뒤에 도는 모드가 불리해질 수 있다")
    if args.repeats < 3:
        print(f"⚠️ --repeats {args.repeats} < 3. run-to-run 평균±표준편차 신뢰도가 낮다")

    common_args = [
        "--source", str(args.source), "--width", str(args.width),
        "--height", str(args.height), "--fps", str(args.fps),
        "--duration", str(args.duration), "--intra-threads", str(args.intra_threads),
        "--power-hz", str(args.power_hz), "--idle-seconds", str(args.idle_seconds),
        "--battery-mah", str(args.battery_mah), "--battery-cell-v", str(args.battery_cell_v),
        "--battery-eta", str(args.battery_eta),
    ]
    if args.no_spin:
        common_args.append("--no-spin")

    # 무작위 순서 (모드, run_index) 조합을 통째로 섞는다 — 특정 모드가 항상
    # 먼저/나중에 돌지 않게. 같은 모드의 반복끼리 순서는 유지(1,2,3..).
    rng = random.Random(args.seed)
    schedule = [(m, k) for m in args.modes for k in range(1, args.repeats + 1)]
    rng.shuffle(schedule)

    print(f"실행 스케줄 ({len(schedule)}개, 쿨다운 {args.cooldown}s): {schedule}")

    for i, (mode, k) in enumerate(schedule):
        run_one(mode, k, common_args, args.python)
        if i < len(schedule) - 1:
            print(f"  쿨다운 {args.cooldown}s ...")
            time.sleep(args.cooldown)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    for mode in args.modes:
        summary = aggregate(mode, args.repeats)
        out_path = os.path.join(RESULTS_DIR, f"power_{mode}_480p_summary.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
        print(f"\n{mode}: delta={summary['power_delta_w']}  "
              f"energy/frame={summary['energy_per_frame_mJ']}  -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())