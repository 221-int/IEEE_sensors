"""EAR 베이스라인을 **정식 프로토콜로** 재고, 비열등 마진 delta 를 제안한다.

    python -m src.v2.ear_baseline_folds

무엇을 하는가
------------
`results/v2/phase1_csv58.json` 에 저장된 이벤트 단위 EAR 특징으로,
얼려둔 층화 5-fold 를 돌려 EAR 베이스라인의 성능과 **그 변동폭**을 냅니다.

    test = fold i, val = fold i+1, train = 나머지   (피험자 분리)
    임계값은 **val 에서 골라 얼려서** test 에 적용
    격자는 src/v2/common/thresholds.py 하나만 사용

왜 필요한가
----------
우리 인코더가 "EAR 에 비열등하다"고 말하려면 마진 delta 가 필요한데,
**delta 가 베이스라인 자체의 fold 간 변동보다 작으면 어떤 방법도 판정을 못 받습니다**
(측정 잡음에 묻힘). 그래서 EAR 이 fold 를 바꿨을 때 얼마나 흔들리는지를 먼저 재고,
그 크기로 delta 를 정합니다. **이 작업은 크롭이 없어도 됩니다** — EAR 은 영상이
필요 없으므로 전처리 전에 끝낼 수 있습니다.

산출물
    results/v2/ear_baseline_folds.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from src.v2.common import repro, splits, stats
from src.v2.common import thresholds as TH

IN = "results/v2/phase1_csv58.json"
OUT = "results/v2/ear_baseline_folds.json"

# EAR 특징 두 가지. 부호 규약이 다르므로 반드시 canonical() 을 통과시킨다.
FEATURES = {
    # 창 안 상대 하강폭. 클수록 blink.
    "drop_ratio": ("drop", True),
    # 창 최저 EAR 절대값. 작을수록 blink. (고전적 임계값 규칙)
    "min_ear": ("min_ear", False),
}


def load_events(path: str) -> dict[str, np.ndarray]:
    with open(path, encoding="utf-8") as f:
        users = json.load(f)["users"]
    cols: dict[str, list] = {k: [] for k in
                             ("subject", "is_blink", "drop", "min_ear", "n_missing")}
    for u, r in users.items():
        e = r.get("_events")
        if e is None:
            raise SystemExit(
                f"User {u} 에 `_events` 가 없습니다. phase1_csv58 을 --redo 로 다시 "
                f"돌려 이벤트 원자료를 남기십시오.")
        n = len(e["is_blink"])
        cols["subject"] += [int(u)] * n
        for k in ("is_blink", "drop", "min_ear", "n_missing"):
            cols[k] += list(e[k])
    return {k: np.asarray(v) for k, v in cols.items()}


def main() -> int:
    repro.ensure_hashseed()
    repro.seal(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=IN)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--max-missing", type=int, default=19,
                    help="이벤트 19프레임 중 결측이 이보다 많으면 제외 (기본: 제외 안 함)")
    args = ap.parse_args()

    d = load_events(args.inp)
    keep = d["n_missing"] <= args.max_missing
    for k in d:
        d[k] = d[k][keep]
    subj, y = d["subject"], d["is_blink"].astype(int)
    assign = splits.load_folds()

    print(f"이벤트 {len(y):,}개  (blink {int(y.sum()):,} / unblink {int((y==0).sum()):,})  "
          f"피험자 {len(np.unique(subj))}명")
    rep = splits.chance_report(y)
    print(f"클래스 균형: {rep}")
    print()

    out: dict = {"env": repro.env_fingerprint(), "n_events": int(len(y)),
                 "max_missing": args.max_missing, "features": {}}

    for fname, (col, higher) in FEATURES.items():
        s = TH.canonical(d[col], higher_fires=higher)
        TH.AUDIT.reset()

        per_fold = []
        print(f"[{fname}]  (higher_fires={higher})")
        print(f"  {'fold':>4}{'n_test':>8}{'val acc':>9}{'test acc':>10}"
              f"{'test AUC':>10}{'test AP':>9}{'thr':>10}")
        for i in range(splits.N_FOLDS):
            m = splits.subject_masks(subj, assign, splits.fold_rotation(i))
            if min(m["train"].sum(), m["val"].sum(), m["test"].sum()) == 0:
                raise SystemExit(
                    f"fold {i} 의 train/val/test 중 비어 있는 것이 있습니다 "
                    f"({m['train'].sum()}/{m['val'].sum()}/{m['test'].sum()}).\n"
                    f"  fold 는 58명 전원 기준으로 얼려져 있습니다. 일부 사용자만 담긴 "
                    f"파일로는 돌릴 수 없습니다 — 58명 전량으로 다시 만드십시오.")
            pick = TH.select_threshold(s[m["val"]], y[m["val"]],
                                       TH.crit_accuracy(), name=f"ear/{fname}")
            te = TH.evaluate_at(s[m["test"]], y[m["test"]], pick["thr"])
            auc = TH.roc_auc(s[m["test"]], y[m["test"]])
            ap_ = TH.average_precision(s[m["test"]], y[m["test"]])
            per_fold.append({"fold": i, "n_test": int(m["test"].sum()),
                             "val_acc": pick["val_metrics"]["accuracy"],
                             "test_acc": te["accuracy"], "test_auc": auc,
                             "test_ap": ap_, "thr": pick["thr"],
                             "test_precision": te["precision"]})
            print(f"  {i:>4}{m['test'].sum():>8}{pick['val_metrics']['accuracy']:>9.4f}"
                  f"{te['accuracy']:>10.4f}{auc:>10.4f}{ap_:>9.4f}{pick['thr']:>10.4f}")
        TH.AUDIT.require_uniform()

        accs = [f["test_acc"] for f in per_fold]
        aucs = [f["test_auc"] for f in per_fold]
        opt = float(np.mean([f["val_acc"] - f["test_acc"] for f in per_fold]))

        # 피험자 클러스터 부트스트랩 (주 지표)
        def auc_of(rows, s=s, y=y):
            return TH.roc_auc(s[rows], y[rows])

        boot = stats.subject_bootstrap(auc_of, subj, n_boot=args.n_boot, seed=0)
        fb_auc = stats.fold_bootstrap(aucs)
        fb_acc = stats.fold_bootstrap(accs)
        sug = stats.suggest_delta(aucs, boot)

        print(f"  fold 평균  acc {np.mean(accs):.4f} ± {np.std(accs, ddof=1):.4f}   "
              f"AUC {np.mean(aucs):.4f} ± {np.std(aucs, ddof=1):.4f}")
        print(f"  val 낙관 편향 (val-test acc) {opt:+.4f}")
        print(f"  피험자 부트스트랩 AUC {boot['point']:.4f} "
              f"[{boot['ci_lo']:.4f}, {boot['ci_hi']:.4f}]  (폭 {boot['ci_hi']-boot['ci_lo']:.4f})")
        print(f"  delta 후보  tight {sug['delta_tight']:.4f}  loose {sug['delta_loose']:.4f}")
        print()

        out["features"][fname] = {
            "higher_fires": higher, "per_fold": per_fold,
            "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs, ddof=1)),
            "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs, ddof=1)),
            "val_optimism": opt,
            "subject_bootstrap_auc": boot,
            "fold_bootstrap_auc": fb_auc, "fold_bootstrap_acc": fb_acc,
            "delta_suggestion": sug,
        }

    best = max(out["features"], key=lambda k: out["features"][k]["auc_mean"])
    out["baseline_choice"] = {
        "feature": best,
        "why": "두 변형 중 강한 쪽을 베이스라인으로 삼는다. 약한 쪽을 골라 놓고 "
               "이기면 그것은 방법의 성능이 아니라 베이스라인 선택의 결과다.",
        "auc_mean": out["features"][best]["auc_mean"],
    }
    print(f"베이스라인 채택: **{best}** (AUC {out['features'][best]['auc_mean']:.4f})")
    print(out["baseline_choice"]["why"])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    os.replace(tmp, args.out)
    print(f"\n  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
