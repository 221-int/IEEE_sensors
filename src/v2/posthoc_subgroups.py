"""train_encoder.json 사후 분석 — 배치별 / 안경별 / 이득의 출처.

    python -m src.v2.posthoc_subgroups

⚠️ **이 스크립트가 내는 숫자는 풀링 PR-AUC 가 아니다.**
`train_encoder.json` 에는 런별 `by_subject`(피험자별 PR-AUC)만 남아 있고 원시 test
점수는 없다(2026-08-03 에 사이드카 저장을 추가했지만 이번 런은 그 전 코드로 돌았다).
따라서 여기서 계산하는 것은 **피험자별 PR-AUC 의 평균**이며, 이벤트를 섞어 계산한
풀링 값과 다른 숫자다. PROTOCOL §9-1 에 같은 종류의 격차가 기록돼 있다
(EAR: 사용자별 중앙 0.930 vs 섞으면 0.909. 배포 조건의 값은 후자).

→ **서브그룹에 δ=0.02 비열등 판정을 붙이지 않는다.** 여기의 CI 는 피험자 재표집으로
   얻은 평균의 불확실도이지 프로토콜 판정이 아니다.

부트스트랩·판정 함수는 `src/v2/common/stats.py` 만 쓴다(실험 스크립트가 직접 구현 금지).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

from src.v2.common import repro, stats

RESULT = "results/v2/train_encoder.json"
CONTAM = "results/v2/train_encoder_contaminated.json"
INDEX = "data/processed/v2/index.npz"
GLASSES = "data/raw/mEBAL2/glasses_labels_58.csv"
OUT = "results/v2/posthoc_subgroups.json"


# ------------------------------------------------------------------ 적재
def load_by_subject(path: str) -> dict:
    """런별 by_subject 를 피험자 단위로 접는다.

    피험자 하나는 정확히 한 fold 의 test 에만 들어가고 시드가 3개이므로,
    시드 3개를 **평균**해 피험자당 값 하나를 만든다. 시드 간 std 도 남긴다 —
    조건 차이가 시드 std 보다 작으면 결론은 "미측정"이다(PROTOCOL §2).
    """
    d = json.load(open(path, encoding="utf-8"))
    acc: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in d["runs"]:
        for u, v in r["by_subject"].items():
            a = acc[int(u)]
            a["n"].append(v["n"])
            a["fold"].append(r["fold"])
            for k in ("ours_pr_auc", "ear_head_pr_auc", "ear_rule_pr_auc"):
                a[k].append(v[k])
    out = {}
    for u, a in acc.items():
        if len(set(a["fold"])) != 1:
            raise SystemExit(f"피험자 {u} 가 여러 fold 에 나타납니다. fold 파일 확인 필요.")
        out[u] = {
            "n": int(np.mean(a["n"])), "fold": a["fold"][0], "n_seeds": len(a["ours_pr_auc"]),
            **{k: float(np.mean(a[k])) for k in
               ("ours_pr_auc", "ear_head_pr_auc", "ear_rule_pr_auc")},
            **{k + "_seed_std": float(np.std(a[k], ddof=1)) if len(a[k]) > 1 else float("nan")
               for k in ("ours_pr_auc", "ear_head_pr_auc", "ear_rule_pr_auc")},
        }
    return out, d


def load_strata(subjects: list[int]) -> dict:
    """배치(2020/2022) 와 안경 라벨을 **파일에서 직접** 읽는다(문서 말고)."""
    idx = np.load(INDEX)
    v = idx["e_valid"].astype(bool)
    es, eb, ey = idx["e_subject"][v], idx["e_batch2020"][v], idx["e_is_blink"][v]
    g = {int(r["user"]): int(r["glasses"])
         for r in csv.DictReader(open(GLASSES, encoding="utf-8"))}
    out = {}
    for u in subjects:
        k = es == u
        b = np.unique(eb[k])
        if len(b) != 1:
            raise SystemExit(f"피험자 {u} 의 배치 라벨이 하나가 아닙니다: {b}")
        out[u] = {"batch2020": int(b[0]), "glasses": g[u],
                  "n_events": int(k.sum()), "prevalence": float(ey[k].mean())}
    return out


# ------------------------------------------------------------------ 집계
def group_stats(vals: dict, members: list[int], n_boot: int, seed: int = 0) -> dict:
    """피험자 집합 하나의 요약. 부트스트랩은 **피험자를 재표집**한다."""
    if not members:
        return {"n_subjects": 0}
    o = np.array([vals[u]["ours_pr_auc"] for u in members])
    e = np.array([vals[u]["ear_head_pr_auc"] for u in members])
    r = np.array([vals[u]["ear_rule_pr_auc"] for u in members])
    gain = o - e
    # subject_bootstrap 은 행 인덱스를 받는다. 여기서는 **행 = 피험자** 다.
    bs = stats.subject_bootstrap(lambda rows: float(gain[rows].mean()),
                                 np.arange(len(members)), n_boot=n_boot, seed=seed)
    n = np.array([vals[u]["n"] for u in members], float)
    return {
        "n_subjects": len(members),
        "n_events": int(n.sum()),
        "ours_mean": float(o.mean()), "ours_std": float(o.std(ddof=1)),
        "ear_head_mean": float(e.mean()), "ear_head_std": float(e.std(ddof=1)),
        "ear_rule_mean": float(r.mean()),
        "gain_mean": float(gain.mean()), "gain_median": float(np.median(gain)),
        "gain_ci": [bs["ci_lo"], bs["ci_hi"]],
        "gain_weighted_by_events": float((gain * n).sum() / n.sum()),
        "n_gain_negative": int((gain < 0).sum()),
        "worst_subject": int(members[int(np.argmin(o))]),
        "worst_ours": float(o.min()),
        "seed_std_median_ours": float(np.median(
            [vals[u]["ours_pr_auc_seed_std"] for u in members])),
        # 오류비 (1-ours)/(1-ear_head). 절대 PR-AUC 이득은 천장(1-ear_head)에 눌리므로,
        # "강한 베이스라인 구간에서도 줄었는가"는 이 비율로 봐야 한다.
        "error_ratio_median": float(np.median((1 - o) / (1 - e))),
        "error_ratio_below1": int(((1 - o) / (1 - e) < 1).sum()),
        "loo": loo_influence(vals, members),
    }


def loo_influence(vals: dict, members: list[int]) -> dict:
    """피험자 한 명을 빼면 그룹 결론이 얼마나 흔들리는가.

    PROTOCOL 부록 #9 는 "n=1 값을 58명 값처럼 쓰지 말 것"을 적어 뒀지만, 반대 방향의
    같은 사고 — **한 명이 그룹 평균을 끌고 가는 것** — 은 막지 못한다. 그래서
    그룹마다 leave-one-out 을 돌려 최대 영향자를 항상 함께 낸다.

    부호가 바뀌는 사람이 있으면 그 그룹의 결론은 **그 한 명의 결론**이다.
    """
    o = np.array([vals[u]["ours_pr_auc"] for u in members])
    e = np.array([vals[u]["ear_head_pr_auc"] for u in members])
    gain = o - e
    full = float(gain.mean())
    shifts = np.array([float(np.delete(gain, i).mean()) - full for i in range(len(gain))])
    j = int(np.argmax(np.abs(shifts)))
    flips = [int(members[i]) for i in range(len(gain))
             if np.sign(np.delete(gain, i).mean()) != np.sign(full)]
    return {
        "gain_full": full,
        "max_influence_user": int(members[j]),
        "max_influence_shift": float(shifts[j]),
        "gain_without_max": full + float(shifts[j]),
        "sign_flipping_users": flips,
        "top3": [{"user": int(members[i]), "shift": float(shifts[i]),
                  "gain_of_user": float(gain[i])}
                 for i in np.argsort(-np.abs(shifts))[:3]],
    }


def gain_source(vals: dict, members: list[int], n_boot: int) -> dict:
    """이득이 'EAR 이 망가진 소수'에서 오는가.

    우리 이득이 베이스라인이 약한 피험자에 몰려 있으면, 그것은 "표현이 좋다"가 아니라
    "그 피험자에서 랜드마크가 눈꺼풀을 못 따라간다"는 뜻이다. 세 각도로 본다.

      1) 순위 상관   gain 과 ear_head 성능의 Spearman rho. 강한 음수면 의심
      2) 집중도      상위 몇 명이 전체 이득의 몇 %를 차지하는가
      3) 절단 재계산 ear_head 하위 사분위를 빼도 이득이 남는가
    """
    o = np.array([vals[u]["ours_pr_auc"] for u in members])
    e = np.array([vals[u]["ear_head_pr_auc"] for u in members])
    gain = o - e

    def spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        return float(np.corrcoef(ra, rb)[0, 1])

    order = np.argsort(-gain)                       # 이득 큰 순
    total = gain.sum()
    conc = {}
    for k in (1, 3, 5, 10):
        if k <= len(gain):
            conc[f"top{k}_share"] = float(gain[order[:k]].sum() / total) if total else float("nan")
    q1 = float(np.percentile(e, 25))
    keep = e > q1                                   # 하위 사분위 제외
    bs_keep = stats.subject_bootstrap(
        lambda rows: float(gain[keep][rows].mean()),
        np.arange(int(keep.sum())), n_boot=n_boot, seed=0)
    return {
        "spearman_gain_vs_ear_head": spearman(gain, e),
        "pearson_gain_vs_ear_head": float(np.corrcoef(gain, e)[0, 1]),
        "concentration": conc,
        "ear_head_q1": q1,
        "excl_weak_ear": {
            "n_subjects": int(keep.sum()),
            "gain_mean": float(gain[keep].mean()),
            "gain_ci": [bs_keep["ci_lo"], bs_keep["ci_hi"]],
            "ours_mean": float(o[keep].mean()),
            "ear_head_mean": float(e[keep].mean()),
        },
        "weakest_ear_subjects": [
            {"user": int(members[i]), "ear_head": float(e[i]), "ours": float(o[i]),
             "gain": float(gain[i])}
            for i in np.argsort(e)[:8]
        ],
        "largest_gain_subjects": [
            {"user": int(members[i]), "ear_head": float(e[i]), "ours": float(o[i]),
             "gain": float(gain[i])}
            for i in order[:8]
        ],
    }


def contamination_delta(clean: dict, path: str = CONTAM) -> dict:
    """오염 전(U18 포함·좌석 필터 없음·구 fold) fold0 과 정리 후 fold0 의 차이.

    ⚠️ 인용 금지. 조건이 여러 개 동시에 다르므로 **어느 요인의 기여인지 분해되지
    않는다**(U18 제외 / 좌석 이벤트 제외 / fold 재구성 / 코드 변경이 겹쳐 있다).
    """
    if not os.path.exists(path):
        return {"available": False}
    c = json.load(open(path, encoding="utf-8"))
    cr = [r for r in c["runs"] if r["fold"] == 0]
    kr = [r for r in clean["runs"] if r["fold"] == 0]
    if not cr or not kr:
        return {"available": False}

    def agg(runs, key):
        v = [r[key]["pr_auc"] for r in runs]
        return {"mean": float(np.mean(v)), "n_runs": len(v),
                "std": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                "values": [float(x) for x in v]}

    out = {"available": True,
           "contaminated_config": {k: c["config"].get(k) for k in
                                   ("folds", "seeds", "max_epochs", "patience", "ear_feats")},
           "clean_config": {k: clean["config"].get(k) for k in
                            ("folds", "seeds", "max_epochs", "patience", "ear_feats")},
           "contaminated_has_by_subject": bool(cr[0].get("by_subject")),
           "n_test": {"contaminated": cr[0]["n_test"], "clean": kr[0]["n_test"]}}
    for key in ("ours", "ear_head", "ear_rule"):
        out[key] = {"contaminated": agg(cr, key), "clean": agg(kr, key)}
        out[key]["shift"] = out[key]["clean"]["mean"] - out[key]["contaminated"]["mean"]
    dc = [r["ours"]["pr_auc"] - r[r["baseline_used"]]["pr_auc"] for r in cr]
    dk = [r["ours"]["pr_auc"] - r[r["baseline_used"]]["pr_auc"] for r in kr]
    out["diff"] = {"contaminated": float(np.mean(dc)), "clean": float(np.mean(dk)),
                   "clean_seed_std": float(np.std(dk, ddof=1)) if len(dk) > 1 else None,
                   "shift": float(np.mean(dk) - np.mean(dc))}
    return out


# ------------------------------------------------------------------ main
def main() -> int:
    repro.ensure_hashseed()
    repro.seal(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default=RESULT)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    vals, raw = load_by_subject(args.result)
    subs = sorted(vals)
    strata = load_strata(subs)
    print(f"피험자 {len(subs)}명 (시드 {vals[subs[0]]['n_seeds']}개 평균), "
          f"이벤트 {sum(strata[u]['n_events'] for u in subs):,}")
    print(f"판정(주 지표, 풀링) = {raw['verdict']['verdict']}  "
          f"차이 {raw['verdict']['paired_bootstrap']['point']:+.4f} "
          f"CI [{raw['verdict']['paired_bootstrap']['ci_lo']:+.4f}, "
          f"{raw['verdict']['paired_bootstrap']['ci_hi']:+.4f}]  δ={raw['verdict']['delta']}")

    groups = {
        "all": subs,
        "batch2020": [u for u in subs if strata[u]["batch2020"] == 1],
        "batch2022": [u for u in subs if strata[u]["batch2020"] == 0],
        "glasses": [u for u in subs if strata[u]["glasses"] == 1],
        "no_glasses": [u for u in subs if strata[u]["glasses"] == 0],
    }
    res = {g: group_stats(vals, m, args.n_boot) for g, m in groups.items()}

    print(f"\n{'그룹':<12}{'인원':>5}{'이벤트':>9}{'ours':>9}{'ear_head':>10}"
          f"{'이득':>9}{'이득 95%CI':>22}{'열세':>5}{'오류비':>8}")
    for g, s in res.items():
        lo, hi = s["gain_ci"]
        print(f"{g:<12}{s['n_subjects']:>5}{s['n_events']:>9,}{s['ours_mean']:>9.4f}"
              f"{s['ear_head_mean']:>10.4f}{s['gain_mean']:>+9.4f}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}{s['n_gain_negative']:>5}"
              f"{s['error_ratio_median']:>8.3f}")

    print(f"\nleave-one-out — 한 명이 그룹 결론을 끌고 가는가")
    for g, s in res.items():
        l = s["loo"]
        flip = f"  ⚠ 부호 뒤집는 사용자 {l['sign_flipping_users']}" if l["sign_flipping_users"] else ""
        print(f"  {g:<12} 최대영향 U{l['max_influence_user']:<3} "
              f"({l['gain_full']:+.4f} → {l['gain_without_max']:+.4f}, "
              f"{l['max_influence_shift']:+.4f}){flip}")

    src = {g: gain_source(vals, groups[g], args.n_boot) for g in ("all", "batch2020", "batch2022")}
    s = src["all"]
    print(f"\n이득의 출처 (전체 57명)")
    print(f"  Spearman(이득, ear_head 성능) = {s['spearman_gain_vs_ear_head']:+.3f}")
    print(f"  상위 집중도: " + "  ".join(
        f"{k} {v:.1%}" for k, v in s["concentration"].items()))
    ex = s["excl_weak_ear"]
    print(f"  ear_head 하위 사분위(<{s['ear_head_q1']:.4f}) 제외 → {ex['n_subjects']}명, "
          f"이득 {ex['gain_mean']:+.4f} CI [{ex['gain_ci'][0]:+.4f}, {ex['gain_ci'][1]:+.4f}]")

    contam = contamination_delta(raw)
    if contam["available"]:
        print(f"\n오염 전후 (fold0, 인용 금지)")
        for k in ("ours", "ear_head", "ear_rule"):
            print(f"  {k:<9} {contam[k]['contaminated']['mean']:.4f} → "
                  f"{contam[k]['clean']['mean']:.4f}  ({contam[k]['shift']:+.4f})")
        print(f"  차이      {contam['diff']['contaminated']:+.4f} → "
              f"{contam['diff']['clean']:+.4f}  ({contam['diff']['shift']:+.4f})")

    out = {
        "_caveat": "피험자별 PR-AUC 의 평균이며 풀링 값이 아니다. δ 판정에 쓰지 말 것. "
                   "원시 test 점수가 저장되지 않은 런이라 풀링 계산이 불가능하다.",
        "env": repro.env_fingerprint(),
        "source_result": args.result,
        "primary_verdict": raw["verdict"],
        "n_subjects": len(subs),
        "groups": res,
        "gain_source": src,
        "contamination": contam,
        "per_subject": {str(u): {**vals[u], **strata[u]} for u in subs},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    os.replace(tmp, args.out)
    print(f"\n  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
