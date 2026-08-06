"""두 학습 변형(구조 또는 D)의 짝지은 비교 — T3-8 절제 실험 · T3-5 D 스윕용.

    python -m src.v2.compare_variants --a results/v2/abl_vpres_ref.json \
                                      --b results/v2/abl_sym16.json

왜 새 스크립트가 필요한가
------------------------
`train_encoder.py` 의 `verdict()` 는 **한 런 안에서 ours vs EAR** 만 판정한다.
`posthoc_subgroups.py` 는 **ours vs ear_head** 를 서브그룹별로 본다.
**서로 다른 두 런(구조 A vs 구조 B)을 짝지어 비교하는 경로는 없었다.**
T3-8 은 그것을 요구한다 — "vpres 의 34% 추가 연산이 정당화되는가".

짝짓기의 근거
------------
`run_fold` 의 test 집합은 fold 만으로 결정된다(`splits.fold_rotation`). 따라서 같은
(fold, seed) 의 두 사이드카는 **같은 이벤트를 같은 순서로** 담는다. 그래서 이벤트
단위로 짝지을 수 있고, 짝지으면 피험자 효과가 상쇄되어 CI 가 좁아진다
(PROTOCOL §9-1: "두 방법의 CI 를 따로 구해 비교하면 불필요하게 넓어진다").
→ 가정에 기대지 않고 `y` 와 `subject` 가 원소 단위로 같은지 **검사한 뒤** 짝짓는다.

규칙
----
- 부트스트랩은 `src/v2/common/stats.py` 만 쓴다 (직접 구현 금지)
- PR-AUC 는 `src/v2/common/thresholds.py` 의 `average_precision` 만 쓴다
- 이 스크립트는 **읽기 전용**이다. 모델·사이드카·fold 파일을 쓰지 않는다
- 탐색(fold 0·1) 런을 비교하면 결과에 `exploratory: true` 가 박힌다.
  **그 숫자는 확정 값으로 인용하지 않는다** (PROTOCOL 부록 #5, TASKS §3-2)

두 추정량을 함께 낸다
--------------------
PROTOCOL §0 이 실측해 둔 것: 런을 섞어 AP 를 한 번 계산하면 fold 별로 계산해
평균한 것보다 0.002~0.003 낮게 나온다. 차이를 볼 때는 두 변형에 같은 방향으로
작용하므로 대부분 상쇄되지만, **어느 추정량인지 밝히지 않고 쓰는 것이 문제**였다.
그래서 둘 다 낸다:

  pooled   런을 전부 concat 해 AP 를 한 번. `train_encoder.verdict()` 와 같은 추정량
  per_run  런마다 AP 를 내고 그 차이를 평균. 시드/폴드 분산을 볼 수 있다
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from src.v2.common import repro, stats
from src.v2.common import thresholds as TH

OUT = "results/v2/compare_variants.json"


# ------------------------------------------------------------------ 적재
def scores_dir_of(result_json: str) -> str:
    """`train_encoder.scores_dir()` 와 같은 규칙. --out 에서 파생된다."""
    return os.path.splitext(result_json)[0] + "_scores"


def load_variant(result_json: str) -> dict:
    """결과 JSON + 런별 사이드카를 (fold, seed) 로 키를 잡아 읽는다."""
    with open(result_json, encoding="utf-8") as f:
        meta = json.load(f)
    sd = scores_dir_of(result_json)
    if not os.path.isdir(sd):
        raise SystemExit(
            f"점수 사이드카가 없습니다: {sd}\n"
            "  사이드카 저장 이전 코드로 돈 런은 이 비교에 쓸 수 없습니다.")
    runs = {}
    for r in meta["runs"]:
        fold, seed = int(r["fold"]), int(r["seed"])
        p = os.path.join(sd, f"fold{fold}_seed{seed}.npz")
        if not os.path.exists(p):
            raise SystemExit(f"사이드카 누락: {p}")
        d = np.load(p, allow_pickle=True)
        runs[(fold, seed)] = {
            "ours": d["ours"].astype(np.float64),
            "y": d["y"].astype(np.int64),
            "subject": d["subject"].astype(np.int64),
            "pr_auc_json": float(r["ours"]["pr_auc"]),
        }
    return {"meta": meta, "runs": runs, "path": result_json}


def arch_facts(cfg: dict) -> dict:
    """구조의 정적 비용. `encoder.analyse()` 가 유일한 출처다(문서 표를 베끼지 않는다).

    🔴 `--front` 를 반드시 함께 본다. front 가 image_cnn 인데 `arch` 만 읽으면
    **vpres 의 비용(12.41 MMAC / 79,424)을 image_cnn(31.81 / 471,536) 자리에 찍는다.**
    실제로 그렇게 잘못 찍힌 적이 있다(2026-08-06). config 에 front 가 없는 옛 런은
    `ours` 로 본다 — --front 도입 이전 런이기 때문이다.
    """
    from src.v2.model import encoder as E
    front = cfg.get("front", "ours")
    d = int(cfg.get("latent", 16))

    if front != "ours":
        out_dim = 1 if front == "image_cnn_max" else d
        a = E.analyse_image_cnn(out_dim)
        c, hh, ww = a["feat"]
        out = {"arch": front, "d_latent": out_dim,
               "feat": [c, hh, ww], "vstride": None,
               "aperture_at_bottleneck_px": float("nan"),
               "conv_mmac": a["conv_mmac"], "fc_mmac": 0.0,
               "total_mmac": a["total_mmac"], "flat_dim": a["flat_dim"]}
        try:
            m = E.build_image_cnn(out_dim)
            out["encoder_params"] = int(sum(p.numel() for p in m.parameters()))
            # max 변형의 시간 처리는 파라미터가 없다(mEBAL 원문 §5.1 max pooling)
            out["head_params"] = (0 if front == "image_cnn_max"
                                  else int(sum(p.numel()
                                               for p in E.build_head(d).parameters())))
            out["total_params"] = out["encoder_params"] + out["head_params"]
        except ImportError:
            out["encoder_params"] = out["head_params"] = out["total_params"] = None
        return out

    arch = cfg.get("arch", "vpres")
    r = E.analyse(arch, d)
    out = {"arch": arch, "d_latent": d,
           "feat": list(r["feat"]), "vstride": r["vstride"],
           "aperture_at_bottleneck_px": r["aperture_at_bottleneck_px"],
           "conv_mmac": r["conv_mmac"], "fc_mmac": r["fc_mmac"],
           "total_mmac": r["total_mmac"]}
    try:                                    # 파라미터 수는 torch 가 있을 때만
        enc, head = E.build(arch, d), E.build_head(d)
        out["encoder_params"] = int(sum(p.numel() for p in enc.parameters()))
        out["head_params"] = int(sum(p.numel() for p in head.parameters()))
        out["total_params"] = out["encoder_params"] + out["head_params"]
    except ImportError:
        out["encoder_params"] = out["head_params"] = out["total_params"] = None
    return out


# ------------------------------------------------------------------ 짝짓기
def align(a: dict, b: dict) -> tuple[list, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """공통 (fold, seed) 만 골라 concat 한다. 정합성을 **검사한 뒤** 짝짓는다."""
    keys = sorted(set(a["runs"]) & set(b["runs"]))
    if not keys:
        raise SystemExit("두 런에 공통 (fold, seed) 가 없습니다. 비교할 수 없습니다.")
    only_a = sorted(set(a["runs"]) - set(b["runs"]))
    only_b = sorted(set(b["runs"]) - set(a["runs"]))
    if only_a or only_b:
        print(f"  ⚠️ 한쪽에만 있는 런은 제외합니다. A만 {only_a} / B만 {only_b}")

    sa, sb, ys, subs = [], [], [], []
    for k in keys:
        ra, rb = a["runs"][k], b["runs"][k]
        if len(ra["y"]) != len(rb["y"]):
            raise SystemExit(f"{k}: test 이벤트 수가 다릅니다 {len(ra['y'])} vs {len(rb['y'])}")
        if not np.array_equal(ra["y"], rb["y"]):
            raise SystemExit(f"{k}: 정답 배열이 다릅니다. 같은 분할이 아닙니다.")
        if not np.array_equal(ra["subject"], rb["subject"]):
            raise SystemExit(f"{k}: 피험자 배열이 다릅니다. 같은 분할이 아닙니다.")
        sa.append(ra["ours"]); sb.append(rb["ours"])
        ys.append(ra["y"]); subs.append(ra["subject"])
    return (keys, np.concatenate(sa), np.concatenate(sb),
            np.concatenate(ys), np.concatenate(subs))


def compare(a: dict, b: dict, n_boot: int, delta: float, seed: int = 0) -> dict:
    keys, sa, sb, y, sub = align(a, b)
    ca, cb = TH.canonical(sa, True), TH.canonical(sb, True)

    def diff(rows):
        return (TH.average_precision(ca[rows], y[rows])
                - TH.average_precision(cb[rows], y[rows]))

    boot = stats.subject_bootstrap(diff, sub, n_boot=n_boot, seed=seed)

    # 런별 추정량 — 시드/폴드 분산이 차이보다 큰지 보려면 이쪽이 필요하다 (PROTOCOL §2)
    per_run, ap_a, ap_b = [], [], []
    for k in keys:
        ra, rb = a["runs"][k], b["runs"][k]
        pa = TH.average_precision(TH.canonical(ra["ours"], True), ra["y"])
        pb = TH.average_precision(TH.canonical(rb["ours"], True), rb["y"])
        ap_a.append(pa); ap_b.append(pb)
        per_run.append({"fold": k[0], "seed": k[1], "a": pa, "b": pb, "diff": pa - pb})
    d_run = np.array([r["diff"] for r in per_run])

    # fold 안에서 시드만 바뀔 때의 변동. 조건 차이가 이보다 작으면 결론은 "미측정"
    seed_stds = []
    for f in sorted({k[0] for k in keys}):
        va = [r["a"] for r in per_run if r["fold"] == f]
        vb = [r["b"] for r in per_run if r["fold"] == f]
        if len(va) > 1:
            seed_stds += [float(np.std(va, ddof=1)), float(np.std(vb, ddof=1))]
    max_seed_std = max(seed_stds) if seed_stds else float("nan")

    return {
        "n_runs": len(keys), "runs_compared": [list(k) for k in keys],
        "n_events_pooled": int(len(y)), "n_subjects": int(len(np.unique(sub))),
        "pooled": {
            "a_pr_auc": TH.average_precision(ca, y),
            "b_pr_auc": TH.average_precision(cb, y),
            "paired_bootstrap": boot,
            "verdict_vs_delta": stats.non_inferiority(boot["ci_lo"], boot["ci_hi"], delta),
        },
        "per_run": {
            "a_mean": float(np.mean(ap_a)), "a_std": float(np.std(ap_a, ddof=1)),
            "b_mean": float(np.mean(ap_b)), "b_std": float(np.std(ap_b, ddof=1)),
            "diff_mean": float(d_run.mean()),
            "diff_std": float(d_run.std(ddof=1)),
            "fold_bootstrap": stats.fold_bootstrap(list(d_run)),
            "runs": per_run,
        },
        "max_within_fold_seed_std": max_seed_std,
        "undermeasured": bool(abs(d_run.mean()) < max_seed_std),
    }


# ------------------------------------------------------------------ main
def main() -> int:
    repro.ensure_hashseed()
    repro.seal(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="기준 변형 결과 JSON (예: vpres)")
    ap.add_argument("--b", required=True, help="비교 변형 결과 JSON (예: sym16)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--delta", type=float, default=0.02,
                    help="PROTOCOL §9-1 의 δ. 참고용으로만 붙는다 — "
                         "이 판정은 EAR 대비 비열등이 아니라 구조 간 비교다")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    A, B = load_variant(args.a), load_variant(args.b)
    fa, fb = arch_facts(A["meta"]["config"]), arch_facts(B["meta"]["config"])
    res = compare(A, B, args.n_boot, args.delta)

    explor = sorted(set(A["meta"]["config"]["folds"]) | set(B["meta"]["config"]["folds"]))
    exploratory = explor != [0, 1, 2, 3, 4]

    na = f"{fa['arch']}/D{fa['d_latent']}"
    nb = f"{fb['arch']}/D{fb['d_latent']}"
    print(f"A = {na}  ({args.a})")
    print(f"B = {nb}  ({args.b})")
    print(f"공통 런 {res['n_runs']}개, 이벤트 {res['n_events_pooled']:,}, "
          f"피험자 {res['n_subjects']}명")
    if exploratory:
        print(f"  ⚠️ 탐색 런이다 (fold {explor}). **확정 값으로 인용 금지**")

    print(f"\n{'':>14}{'PR-AUC(런평균)':>16}{'PR-AUC(풀링)':>14}{'conv MMAC':>11}"
          f"{'총 MMAC':>9}{'병목px':>8}{'enc params':>12}")
    for nm, f_, key in ((na, fa, "a"), (nb, fb, "b")):
        m = res["per_run"][f"{key}_mean"]; s = res["per_run"][f"{key}_std"]
        print(f"{nm:>14}{f'{m:.4f}±{s:.4f}':>16}{res['pooled'][f'{key}_pr_auc']:>14.4f}"
              f"{f_['conv_mmac']:>11.2f}{f_['total_mmac']:>9.2f}"
              f"{f_['aperture_at_bottleneck_px']:>8.2f}"
              f"{(f_['encoder_params'] or 0):>12,}")

    pb = res["pooled"]["paired_bootstrap"]
    print(f"\n차이 (A − B), 짝지은 피험자 부트스트랩")
    print(f"  풀링   {pb['point']:+.4f}  95% CI [{pb['ci_lo']:+.4f}, {pb['ci_hi']:+.4f}]")
    fbt = res["per_run"]["fold_bootstrap"]
    print(f"  런평균 {res['per_run']['diff_mean']:+.4f} ± {res['per_run']['diff_std']:.4f}"
          f"  (런 부트스트랩 [{fbt['ci_lo']:+.4f}, {fbt['ci_hi']:+.4f}])")
    print(f"  fold 내 시드 std 최대 {res['max_within_fold_seed_std']:.4f}")
    if res["undermeasured"]:
        print("  🔴 조건 차이가 시드 std 보다 작다 → 결론은 **미측정**이다 (PROTOCOL §2). "
              "'기각'이 아니다")
    if pb["ci_lo"] > 0:
        print(f"  → CI 하한 > 0: A 가 B 보다 높다")
    elif pb["ci_hi"] < 0:
        print(f"  → CI 상한 < 0: B 가 A 보다 높다")
    else:
        print(f"  → CI 가 0 을 포함한다. 두 구조의 차이를 이 표본으로 구분하지 못한다")

    out = {
        "env": repro.env_fingerprint(),
        "a": {"path": args.a, "name": na, "facts": fa,
              "config": A["meta"]["config"]},
        "b": {"path": args.b, "name": nb, "facts": fb,
              "config": B["meta"]["config"]},
        "exploratory": exploratory,
        "_caveat": ("탐색 런(fold 0·1)이다. 확정 값으로 인용하지 말 것."
                    if exploratory else None),
        "delta_reference": args.delta,
        "result": res,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    os.replace(tmp, args.out)
    print(f"\n  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
