"""부트스트랩 신뢰구간과 비열등 판정 — v2 단일 구현.

왜 피험자 단위 클러스터 부트스트랩인가
------------------------------------
한 피험자의 이벤트 수백 개는 **서로 독립이 아닙니다**(같은 사람, 같은 세션, 같은 조명).
이벤트를 독립 표본처럼 재표집하면 신뢰구간이 실제보다 훨씬 좁게 나오고, 그러면
"차이가 유의하다"가 너무 쉽게 나옵니다. 그래서 **피험자를 통째로** 재표집합니다.

fold 단위 부트스트랩(5개 값 재표집)도 함께 내지만, 5개는 너무 적어 거칠기 때문에
**주 지표는 피험자 클러스터 부트스트랩**으로 둡니다.

비열등 판정
----------
    (ours - baseline) 차이의 95% CI 하한 > -delta   ->  비열등
    CI 가 -delta 를 걸침                            ->  판정 유보 (동등이 아님)
    CI 하한 > 0                                     ->  우월

"차이가 유의하지 않다"는 동등의 근거가 아닙니다. 표본이 적을수록 아무거나
동등해 보이므로, **delta 를 결과 보기 전에 고정**해야 합니다.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np

Verdict = Literal["superior", "non_inferior", "inconclusive", "inferior"]


def subject_bootstrap(
    metric: Callable[[np.ndarray], float],
    subject: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """피험자를 통째로 재표집해 metric 의 분포를 낸다.

    metric: 선택된 **행 인덱스 배열**을 받아 스칼라를 돌려주는 함수.
            (부트스트랩 표본마다 지표를 다시 계산해야 하므로 인덱스로 넘깁니다.)
    """
    sub = np.asarray(subject)
    subs = np.unique(sub)
    idx_of = {s: np.flatnonzero(sub == s) for s in subs}
    rng = np.random.default_rng(seed)
    point = float(metric(np.arange(len(sub))))
    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(subs, size=len(subs), replace=True)
        rows = np.concatenate([idx_of[s] for s in pick])
        vals[b] = metric(rows)
    lo, hi = np.percentile(vals[np.isfinite(vals)], [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": point, "ci_lo": float(lo), "ci_hi": float(hi),
            "std": float(np.nanstd(vals)), "n_boot": int(n_boot),
            "n_subjects": int(len(subs)), "alpha": alpha}


def fold_bootstrap(per_fold: list[float], n_boot: int = 10000,
                   alpha: float = 0.05, seed: int = 0) -> dict:
    """fold 값 재표집. fold 가 보통 5개뿐이라 **거칩니다** — 참고용입니다."""
    v = np.asarray([x for x in per_fold if np.isfinite(x)], float)
    if v.size < 2:
        return {"point": float(v[0]) if v.size else float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"), "n_folds": int(v.size)}
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, v.size), replace=True).mean(1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": float(v.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "std": float(v.std(ddof=1)), "n_folds": int(v.size),
            "per_fold": [float(x) for x in v]}


def non_inferiority(ci_lo: float, ci_hi: float, delta: float) -> Verdict:
    """차이(ours - baseline)의 CI 와 마진 delta -> 판정."""
    if delta <= 0:
        raise ValueError("delta 는 양수여야 합니다(허용 열세 폭).")
    if ci_lo > 0:
        return "superior"
    if ci_lo > -delta:
        return "non_inferior"
    if ci_hi < -delta:
        return "inferior"
    return "inconclusive"


def suggest_delta(baseline_fold_values: list[float],
                  baseline_subject_ci: dict | None = None) -> dict:
    """delta 후보를 **베이스라인 자체의 변동**에서 제안한다.

    원칙: 우리가 허용하는 열세 폭이 **베이스라인이 fold 를 바꿨을 때 저절로 흔들리는
    폭보다 작으면**, 어떤 방법도 비열등 판정을 못 받습니다(측정 잡음에 묻힘).
    반대로 지나치게 크면 판정이 무의미해집니다.

    그래서 두 값을 제시하고, 사람이 하나를 골라 **결과 보기 전에** 얼립니다.
      tight : fold 간 표준편차       — 통과하기 어렵지만 주장 강도가 높다
      loose : fold 간 범위의 절반    — 통과 가능성이 높지만 주장 강도가 낮다
    """
    v = np.asarray([x for x in baseline_fold_values if np.isfinite(x)], float)
    out = {
        "baseline_fold_mean": float(v.mean()),
        "baseline_fold_std": float(v.std(ddof=1)) if v.size > 1 else None,
        "baseline_fold_range": float(v.max() - v.min()) if v.size else None,
        "delta_tight": float(v.std(ddof=1)) if v.size > 1 else None,
        "delta_loose": float((v.max() - v.min()) / 2) if v.size else None,
        "note": "결과를 본 뒤에 고르지 마십시오. test 를 열기 전에 하나를 골라 "
                "docs/v2/PROTOCOL.md §9-1 에 적고 얼립니다.",
    }
    if baseline_subject_ci:
        out["baseline_subject_ci_width"] = float(
            baseline_subject_ci["ci_hi"] - baseline_subject_ci["ci_lo"])
    return out
