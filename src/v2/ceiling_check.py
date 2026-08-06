"""이 과제가 **천장에 닿았는가** — 구조 절제가 아무것도 못 움직이는 이유를 가른다.

    python -m src.v2.ceiling_check

왜 필요한가
----------
T3-8·T3-5 에서 인코더 구조를 2.6배(9.22↔24.21 MMAC), D 를 8배(8↔64) 흔들었는데
6개 조건이 전부 PR-AUC 0.9818~0.9850 안에 들어왔다 (`results/v2/cmp_*.json`).
두 가지로 읽힌다. **어느 쪽인지에 따라 논문 서술이 정반대가 된다.**

  (A) 천장   남은 오차가 **모든 조건이 똑같이 틀리는** 이벤트다 (라벨 잡음·본질적 모호성).
             → 구조를 바꿔도 안 움직이는 게 당연하다. Method 의 구조 주장을 접고
                Limitations 에 "이 프로토콜은 구조를 판별할 해상도가 없다"고 쓴다.
  (B) 잡음   조건마다 **다른 이벤트**를 틀린다. 개별 모델에 여유가 있는데 시드 분산에
             묻힌 것이다. → 런 수를 늘리면 차이가 드러날 수 있다.

가르는 방법 — 셋을 함께 본다
---------------------------
1. **공통 오차 비율**  18런(6조건×3시드) 이 **전부** 틀린 이벤트가 전체 오차의 몇 %인가.
   높으면 (A).
2. **앙상블 이득**  조건들의 점수를 순위 평균해 합치면 PR-AUC 가 오르는가.
   오르면 오차가 서로 달랐다는 뜻이라 (B). 안 오르면 (A).
3. **오차 집중도**  남은 오차가 소수 피험자에 몰려 있는가. 몰려 있으면 천장이 아니라
   **전이 실패**다 (U1 같은 사례. PROTOCOL §0 참조).

비교 기준선이 반드시 필요하다
---------------------------
"조건 간 오차가 겹친다"만으로는 (A) 라고 못 한다. **같은 조건의 시드끼리도** 겹치기
때문이다. 그래서 조건 간 겹침을 **시드 간 겹침과 나란히** 낸다. 둘이 비슷하면
조건은 시드와 구별되지 않는다 = 구조가 아무것도 하지 않는다.

규칙: 지표는 `common/thresholds.py`, 부트스트랩은 `common/stats.py` 만 쓴다.
읽기 전용이다.
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import combinations

import numpy as np

from src.v2.common import repro
from src.v2.common import thresholds as TH

OUT = "results/v2/ceiling_check.json"
DEFAULT = ["abl_sym16", "abl_vpres_ref", "abl_vfull", "dim_D8", "dim_D32", "dim_D64"]


def load(name: str) -> tuple[dict, dict]:
    """결과 JSON + 사이드카를 (fold, seed) 로 읽는다. thr 은 JSON 쪽에 있다."""
    with open(f"results/v2/{name}.json", encoding="utf-8") as f:
        meta = json.load(f)
    sd = f"results/v2/{name}_scores"
    runs = {}
    for r in meta["runs"]:
        k = (int(r["fold"]), int(r["seed"]))
        d = np.load(os.path.join(sd, f"fold{k[0]}_seed{k[1]}.npz"), allow_pickle=True)
        runs[k] = {"score": d["ours"].astype(np.float64),
                   "y": d["y"].astype(np.int64),
                   "subject": d["subject"].astype(np.int64),
                   "thr": float(r["ours"]["thr"])}
    return meta, runs


def ranks01(x: np.ndarray) -> np.ndarray:
    """0~1 순위. 조건마다 logit 스케일이 달라 그대로 평균하면 안 된다."""
    o = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(np.float64)
    return o / max(len(x) - 1, 1)


def main() -> int:
    repro.ensure_hashseed()
    repro.seal(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="*", default=DEFAULT)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    conds = {n: load(n) for n in args.conditions}
    folds = sorted({k[0] for _, r in conds.values() for k in r})
    seeds = sorted({k[1] for _, r in conds.values() for k in r})
    print(f"조건 {len(conds)}개 × fold {folds} × seed {seeds} = "
          f"{len(conds)*len(folds)*len(seeds)}런")

    out: dict = {"conditions": args.conditions, "folds": folds, "seeds": seeds}
    tot_err = tot_ev = 0
    all_wrong_n = 0
    ens_rows, single_rows = [], []
    inter_cond, inter_seed = [], []
    per_run_err: list = []
    subj_rate: dict[int, float] = {}      # 런당 평균 오차 **건수** (이벤트 수 아님)
    subj_n: dict[int, int] = {}

    for f in folds:
        # 이 fold 의 이벤트는 모든 조건·시드가 공유한다. 검사한 뒤 쓴다.
        ref = conds[args.conditions[0]][1][(f, seeds[0])]
        y, sub = ref["y"], ref["subject"]
        for n, (_, runs) in conds.items():
            for s in seeds:
                r = runs[(f, s)]
                if not (np.array_equal(r["y"], y) and np.array_equal(r["subject"], sub)):
                    raise SystemExit(f"{n} fold{f} seed{s}: 분할이 다르다. 비교 불가.")

        # 런별 오차 지표 (각 런이 val 에서 얼린 자기 임계값을 쓴다)
        err = {}
        for n, (_, runs) in conds.items():
            for s in seeds:
                r = runs[(f, s)]
                pred = (r["score"] >= r["thr"]).astype(np.int64)
                err[(n, s)] = pred != y

        E = np.stack([err[k] for k in err])           # (18, n_ev)
        all_wrong = E.all(axis=0)
        any_wrong = E.any(axis=0)
        # 🔴 주의 1: any_wrong("18런 중 한 번이라도 틀림")을 오차율로 읽으면 안 된다.
        # 런당 오차율보다 8배 부풀려진다(실측 5.93% vs 23.88%). 기준은 **런당**이다.
        # 🔴 주의 2: fold 별 오차 건수를 그대로 평균하면 안 된다. all_wrong_n 은 fold 를
        # 합산한 값이라 분모(런당 오차)도 fold 합산이어야 한다. 안 맞추면 정확히 2배 어긋난다.
        per_run_err.append(E.sum(axis=1))       # (n_run,) — 아래에서 fold 합산한다
        tot_err += int(any_wrong.sum())
        tot_ev += len(y)
        all_wrong_n += int(all_wrong.sum())

        # 겹침 — 조건 간(시드 고정) vs 시드 간(조건 고정). Jaccard.
        def jac(a, b):
            u = (a | b).sum()
            return float((a & b).sum() / u) if u else float("nan")

        for s in seeds:
            for n1, n2 in combinations(args.conditions, 2):
                inter_cond.append(jac(err[(n1, s)], err[(n2, s)]))
        for n in args.conditions:
            for s1, s2 in combinations(seeds, 2):
                inter_seed.append(jac(err[(n, s1)], err[(n, s2)]))

        # 앙상블 — 조건별로 시드 평균한 뒤, 조건들을 순위 평균
        per_cond = {}
        for n, (_, runs) in conds.items():
            per_cond[n] = np.mean([ranks01(runs[(f, s)]["score"]) for s in seeds], axis=0)
            single_rows.append((n, f, TH.average_precision(
                TH.canonical(per_cond[n], True), y)))
        ens = np.mean([per_cond[n] for n in args.conditions], axis=0)
        ens_rows.append((f, TH.average_precision(TH.canonical(ens, True), y)))

        rate = E.mean(axis=0)             # 이벤트별 "런 중 몇 %가 틀렸나"
        for u in np.unique(sub):
            m = sub == u
            subj_rate[int(u)] = subj_rate.get(int(u), 0.0) + float(rate[m].sum())
            subj_n[int(u)] = subj_n.get(int(u), 0) + int(m.sum())

    # ---------------------------------------------------------------- 보고
    print(f"\n[1] 공통 오차 — 몇 %가 '어떤 런으로도 못 맞히는' 오차인가")
    # fold 를 **합산**한다 (concat 이 아니라 sum). 런 하나는 모든 fold 를 평가하므로
    # 런당 오차 = fold 별 오차의 합이어야 tot_ev·all_wrong_n 과 분모가 맞는다.
    pr = np.sum(per_run_err, axis=0)          # 런별 총 오차 건수 (fold 합산)
    mean_run_err = float(pr.mean())
    share = all_wrong_n / max(mean_run_err, 1)
    print(f"  런당 오차        평균 {mean_run_err:.0f} / {tot_ev:,} "
          f"({mean_run_err/tot_ev:.2%}), 범위 {pr.min()}~{pr.max()}")
    print(f"  18런 전부 틀림   {all_wrong_n:,}  = **전형적 런 오차의 {share:.1%}**")
    print(f"  1런 이상 틀림    {tot_err:,}  ({tot_err/tot_ev:.2%}) "
          f"← 이것을 오차율로 읽지 말 것. 런당의 {tot_err/mean_run_err:.1f}배다")
    print(f"  → 오차의 {1-share:.0%} 는 런마다 바뀐다 = 시드 잡음이지 본질적 한계가 아니다")

    print(f"\n[2] 오차 겹침 (Jaccard) — 조건 간 vs 시드 간")
    ic, is_ = float(np.mean(inter_cond)), float(np.mean(inter_seed))
    print(f"  조건 간 (구조·D 가 다름) {ic:.3f}")
    print(f"  시드 간 (조건은 같음)   {is_:.3f}")
    print(f"  비 = {ic/is_:.3f}  " +
          ("→ 조건을 바꾼 것이 시드를 바꾼 것과 구별되지 않는다"
           if ic / is_ > 0.95 else "→ 조건 간 오차가 시드 간보다 덜 겹친다"))

    print(f"\n[3] 앙상블 이득 — 6조건 순위 평균")
    best_by_fold = {}
    for n, f, ap_ in single_rows:
        best_by_fold[f] = max(best_by_fold.get(f, 0), ap_)
    gains = []
    for f, e in ens_rows:
        g = e - best_by_fold[f]
        gains.append(g)
        print(f"  fold {f}: 최고 단일 {best_by_fold[f]:.4f} → 앙상블 {e:.4f}  ({g:+.4f})")
    mg = float(np.mean(gains))
    print(f"  평균 {mg:+.4f}  " +
          ("→ 합쳐도 안 오른다. 같은 이벤트를 틀리고 있다 = 천장"
           if mg < 0.002 else "→ 합치면 오른다. 조건마다 다른 오차 = 아직 여유"))

    print(f"\n[4] 오차 집중도 — 남은 오차가 누구에게 있나 (런당 기준)")
    order = sorted(subj_rate, key=lambda u: -subj_rate[u])
    tot_rate = sum(subj_rate.values())
    cum = np.cumsum([subj_rate[u] for u in order]) / max(tot_rate, 1e-9)
    ev_share = sum(subj_n[u] for u in order[:3]) / max(tot_ev, 1)
    print(f"  피험자 {len(order)}명 중 상위 3명이 오차의 {cum[2]:.1%} "
          f"(이벤트 점유율은 {ev_share:.1%}), 상위 5명이 {cum[4]:.1%}")
    print(f"    {'U':>4}{'n':>7}{'런당 오차율':>12}{'오차 기여':>11}")
    for u in order[:5]:
        print(f"    {u:>4}{subj_n[u]:>7}{subj_rate[u]/subj_n[u]:>11.2%}"
              f"{subj_rate[u]/tot_rate:>10.1%}")
    print("  ⚠️ 이 오차는 **각 런이 val 에서 얼린 임계값** 기준이다. 순위(PR-AUC)가"
          " 좋아도 피험자별 점수 이동이 있으면 높게 나온다 — 둘을 구분해 읽을 것")

    verdict = ("ceiling" if (share > 0.5 and mg < 0.002) else
               "architecture_irrelevant" if (ic / is_ > 0.95 and mg < 0.002) else
               "headroom")
    out.update({
        "n_events": tot_ev,
        "mean_errors_per_run": mean_run_err,
        "mean_error_rate_per_run": mean_run_err / tot_ev,
        "n_events_wrong_in_at_least_one_run": tot_err,
        "n_events_all_runs_wrong": all_wrong_n,
        "share_of_typical_run_error_always_wrong": share,
        "jaccard_between_conditions": ic, "jaccard_between_seeds": is_,
        "jaccard_ratio": ic / is_,
        "ensemble": {"per_fold": [{"fold": f, "ensemble_ap": e,
                                   "best_single_ap": best_by_fold[f]} for f, e in ens_rows],
                     "mean_gain_over_best_single": mg,
                     "_note": "best_single 은 이미 시드 3개를 순위평균한 값이다. "
                              "따라서 이 이득은 **조건을 더 합쳐서 얻는 것**만 잰다"},
        "error_concentration": {
            "top3_share": float(cum[2]), "top5_share": float(cum[4]),
            "top3_event_share": float(ev_share),
            "_note": "런당 오차 기준. 각 런이 val 에서 얼린 임계값을 쓴다",
            "per_subject": [{"subject": u, "errors_per_run": subj_rate[u],
                             "n": subj_n[u], "rate": subj_rate[u] / subj_n[u],
                             "share_of_error": subj_rate[u] / tot_rate} for u in order]},
        "verdict": verdict,
        "_caveat": "탐색 런(fold 0·1, 21명) 기반이다. 확정 값으로 인용하지 말 것.",
        "env": repro.env_fingerprint(),
    })
    print(f"\n판정: **{out['verdict']}**")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    os.replace(tmp, args.out)
    print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
