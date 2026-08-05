"""분할·표집·chance 보고 — v2 단일 구현.

왜 이 파일이 존재하는가
----------------------
mEBAL2 58명은 **이벤트 수가 사람마다 10개에서 1,055개로 105배 차이납니다**
(실측: results/v2/facts_mebal2_events.json). 여기서 두 가지가 따라옵니다.

1. 피험자를 무작위로 5-fold 로 자르면 fold 간 이벤트 수가 중앙 1.83배, 최악 3.77배
   벌어집니다(200회 시뮬레이션). -> **이벤트 수로 층화**하고 결과를 파일에 얼립니다.

2. 재식별 프로브에서 chance 를 1/58 = 0.017 로 쓰면 **틀립니다.** 표집을 균형 맞추지
   않으면 아무것도 안 배운 분류기가 최다 사용자만 찍어 0.0734 를 얻습니다.
   -> 모든 프로브는 `chance_report()` 로 **균등 chance 와 다수 클래스 기준선을 함께**
      보고합니다. 하나만 쓰는 것을 API 로 막습니다.

축별 분할 규칙 (v2 확정)
-----------------------
    유용성(깜빡임 분류)  : 피험자 분리. 층화 5-fold 회전, test=fold i, val=fold i+1
    프라이버시(재식별)   : 피험자 공유 + 시간 분리. 세션 앞 70% 학습 / 뒤 30% 평가
                          그리고 무작위 분할을 **나란히** 보고 (둘의 격차가 단서)

fold 는 58명 전원에 대해 한 번 정해 얼립니다. 나중에 사용자를 제외할 때는
**fold 를 다시 짜지 않고 마스크로 뺍니다.** 다시 짜면 이전 결과와 비교가 끊깁니다.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

FOLDS_PATH = os.path.join(os.path.dirname(__file__), "folds_5fold.json")
FACTS_PATH = "results/v2/facts_mebal2_events.json"
N_SUBJECTS = 58
N_FOLDS = 5

# 시간 블록 분할 경계. 세션 앞 70% 학습 / 뒤 30% 평가.
TIME_BLOCK_FRAC = 0.70


# ============================================================ chance 보고
@dataclass
class ChanceReport:
    n_samples: int
    n_classes: int
    uniform_chance: float          # 1 / n_classes — 균형 표집일 때만 의미 있음
    majority_baseline: float       # 최다 클래스 비율 — 항상 유효한 하한
    balanced: bool                 # 클래스별 개수가 모두 같은가
    counts_min: int
    counts_max: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()

    def __str__(self) -> str:
        tag = "균형" if self.balanced else f"불균형 {self.counts_min}~{self.counts_max}"
        return (f"n={self.n_samples} classes={self.n_classes} ({tag}) "
                f"uniform={self.uniform_chance:.4f} majority={self.majority_baseline:.4f}")


def chance_report(y: np.ndarray) -> ChanceReport:
    """어떤 프로브든 정확도를 보고할 때 **반드시** 함께 내는 기준선.

    균등 chance 만 쓰면 불균형 표집에서 성능을 4배 과대평가하게 됩니다.
    """
    y = np.asarray(y).ravel()
    cls, cnt = np.unique(y, return_counts=True)
    return ChanceReport(
        n_samples=int(y.size), n_classes=int(cls.size),
        uniform_chance=float(1.0 / max(cls.size, 1)),
        majority_baseline=float(cnt.max() / y.size),
        balanced=bool(cnt.min() == cnt.max()),
        counts_min=int(cnt.min()), counts_max=int(cnt.max()),
    )


# ============================================================ fold (층화)
def stratified_folds(counts: dict[int, int], k: int = N_FOLDS) -> dict[int, int]:
    """이벤트 수로 균형 잡힌 피험자 분리 fold 배정. 완전 결정적(시드 없음).

    규칙: 이벤트 수 내림차순(동수면 subject 오름차순)으로, 현재 합이 가장 작은
    fold(동수면 fold 번호 작은 쪽)에 배정. LPT 스케줄링입니다.

    시드를 안 쓰는 이유: fold 는 데이터의 성질이지 실험 조건이 아닙니다. 시드를 두면
    "어떤 시드의 fold 였나"가 또 하나의 자유도가 됩니다.
    """
    order = sorted(counts.keys(), key=lambda s: (-counts[s], s))
    load = [0] * k
    assign: dict[int, int] = {}
    for s in order:
        f = min(range(k), key=lambda j: (load[j], j))
        assign[s] = f
        load[f] += counts[s]
    return assign


def fold_balance(counts: dict[int, int], assign: dict[int, int],
                 k: int = N_FOLDS) -> dict:
    load = [0] * k
    n_sub = [0] * k
    for s, f in assign.items():
        load[f] += counts[s]
        n_sub[f] += 1
    return {"events_per_fold": load, "subjects_per_fold": n_sub,
            "max_over_min": float(max(load) / max(min(load), 1))}


def stratified_folds_2way(counts: dict[int, int], batch: dict[int, str],
                          k: int = N_FOLDS) -> dict[int, int]:
    """이벤트 수 + 배치 이중 층화. 결정적(시드 없음). **v2 의 실제 fold 생성기.**

    2026-08-03: `src/v2/phase3_apply_flags.py` 에서 여기로 옮겼다. 규칙 #4 는
    "격자·시드·분할·프로브는 src/v2/common/ 하나만" 인데 분할 생성기만 실험
    스크립트에 남아 있었다. 그 결과 게이트(G-1b)가 얼린 배정을 **재현해 볼 수
    없었다** — 검사하려면 실험 스크립트를 임포트해야 했고, 그건 규칙이 두 곳에
    사는 상태다. 로직은 바꾸지 않았다(복사·이동).

    **배치 안에서** 이벤트 수 LPT 를 돌리되 배치별 인원 상한 ceil(n_b/k) 를 건다.

    세 후보를 실측 비교해 고른 방식이다 (57명, 유효 이벤트 27,758 기준):

        방식                      전체 max/min   2022 비중 범위   2022 인원
        전역 LPT + 인원 상한          1.016        12~62%  <-- 나쁨   [4,4,3,4,4]
        배치별 LPT                   1.131        26~35%             [1,3,5,5,5] <-- 나쁨
        배치별 뱀 순서                1.359        24~39%             [3,4,4,4,4]
        **배치별 LPT + 인원 상한**     1.144        26~36%             [3,4,4,4,4]

    전체 이벤트 균형(1.016 -> 1.144)을 조금 포기하고 **fold 별 배치 구성**을 잡았다.
    이유: 2022 배치는 EAR AUC 0.885 로 2020(0.933)보다 어렵다. fold 하나의 test 가
    62% 2022 이면 그 fold 만 성능이 낮게 나오고, 그 차이가 fold 분산으로 잡혀
    비열등 판정(delta=0.02)을 흐린다. 전체 이벤트 수 차이는 부트스트랩이 피험자
    단위라 영향이 훨씬 작다.
    또 fold 하나의 2022 인원이 1명이면 배치별 분리 보고가 무의미해진다.
    """
    from math import ceil
    assign: dict[int, int] = {}
    for b in sorted(set(batch.values())):
        us = sorted([u for u in counts if batch[u] == b], key=lambda x: (-counts[x], x))
        cap = ceil(len(us) / k)
        load = [0] * k
        n = [0] * k
        for u in us:
            cand = [j for j in range(k) if n[j] < cap] or list(range(k))
            f = min(cand, key=lambda j: (load[j], n[j], j))
            assign[u] = f
            load[f] += counts[u]
            n[f] += 1
    return assign


def freeze_folds(facts_path: str = FACTS_PATH, out_path: str = FOLDS_PATH,
                 k: int = N_FOLDS, force: bool = False) -> dict:
    """실측 이벤트 수로 fold 를 만들어 **파일에 얼립니다.** 한 번만 실행합니다.

    🔴 2026-08-03: **이미 있는 파일을 덮어쓰지 않습니다.**

    실제로 사고가 났습니다. `gate_minus1` 이 fold 균형을 "검사"하면서 이 함수를
    불렀고, 그 결과 PROTOCOL §4 가 얼려 둔 **57명·배치 이중 층화** 배정이
    이 함수의 구버전 기준(이벤트 수만 층화, 58명, U18 포함)으로 조용히 덮어써졌습니다.
    학습 결과와 fold 가 어긋나면 그 뒤의 모든 숫자가 무의미해집니다.

    현재 fold 의 생성 주체는 `phase3_apply_flags` 입니다. 이 함수는 **v1 경로**이며,
    다시 얼리려면 `force=True` 를 명시해야 합니다. 검사만 하려면
    `k` 를 주고 `dry_run` 대신 `load_folds()` 로 읽으십시오.
    """
    if os.path.exists(out_path) and not force:
        with open(out_path, encoding="utf-8") as f:
            existing = json.load(f)
        return {**existing, "_frozen": True,
                "_note": f"{out_path} 이미 존재 — 덮어쓰지 않았습니다. "
                         "다시 얼리려면 freeze_folds(force=True)."}
    with open(facts_path, encoding="utf-8") as f:
        facts = json.load(f)
    counts = {int(u): int(v["blink1"]) for u, v in facts["per_user"].items()}
    if len(counts) != N_SUBJECTS:
        raise ValueError(f"피험자 수가 {len(counts)} 입니다. 58 이어야 합니다.")
    assign = stratified_folds(counts, k)
    payload = {
        "n_folds": k,
        "source": facts_path,
        "criterion": "blink1 이벤트 수 LPT 층화, 결정적(시드 없음)",
        "balance": fold_balance(counts, assign, k),
        "counts": {str(s): counts[s] for s in sorted(counts)},
        "assign": {str(s): assign[s] for s in sorted(assign)},
        "note": "사용자를 제외할 때 fold 를 다시 짜지 마십시오. 마스크로 빼십시오.",
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    return payload


def load_folds(path: str = FOLDS_PATH) -> dict[int, int]:
    """얼린 fold 배정을 읽습니다. 없으면 실행을 막습니다(즉석 생성 금지)."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} 가 없습니다. 먼저 `python -m src.v2.gate_minus1` 로 fold 를 얼리십시오.")
    with open(path, encoding="utf-8") as f:
        return {int(s): int(v) for s, v in json.load(f)["assign"].items()}


def fold_rotation(fold_id: int, k: int = N_FOLDS) -> dict[str, list[int]]:
    """test = fold_id, val = 다음 fold, train = 나머지.

    val 과 test 가 분리되므로 '에폭·임계값을 val 에서 골라 얼려 test 에 적용'이
    구조적으로 강제됩니다.
    """
    test = fold_id % k
    val = (fold_id + 1) % k
    train = [f for f in range(k) if f not in (test, val)]
    return {"test": [test], "val": [val], "train": train}


def subject_masks(subject: np.ndarray, assign: dict[int, int],
                  rotation: dict[str, list[int]]) -> dict[str, np.ndarray]:
    """피험자 배열 -> train/val/test 불리언 마스크 (피험자 분리)."""
    sub = np.asarray(subject).astype(int)
    fold_of = np.array([assign.get(int(s), -1) for s in sub])
    out = {k: np.isin(fold_of, v) for k, v in rotation.items()}
    unknown = int((fold_of < 0).sum())
    if unknown:
        raise ValueError(f"fold 배정에 없는 피험자 표본이 {unknown} 개 있습니다.")
    overlap = out["train"] & (out["val"] | out["test"])
    if overlap.any():
        raise AssertionError("피험자 분리가 깨졌습니다.")
    return out


# ============================================================ 시간/무작위 분할
def time_block_masks(subject: np.ndarray, t_rel: np.ndarray,
                     frac: float = TIME_BLOCK_FRAC) -> dict[str, np.ndarray]:
    """피험자별 세션 앞 frac 학습 / 뒤 평가 (피험자 공유, 시간 분리).

    재식별 프로브의 기본 분할입니다. `t_rel` 은 세션 내 상대 시각 0~1 이며
    index.npz 의 `f_t_rel` / `e_t_rel` 을 그대로 씁니다.
    """
    t = np.asarray(t_rel, dtype=np.float64)
    if t.min() < 0 or t.max() > 1:
        raise ValueError("t_rel 은 0~1 이어야 합니다.")
    tr = t <= frac
    return {"train": tr, "test": ~tr}


def random_masks(n: int, seed: int, frac: float = TIME_BLOCK_FRAC) -> dict[str, np.ndarray]:
    """무작위 분할. 시간 블록과 **나란히** 보고하기 위한 대조 조건입니다.

    이 숫자 하나만 보고하면 안 됩니다 — 인접 표본이 학습/평가에 나뉘어 들어가
    낙관 편향이 생깁니다.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = int(round(frac * n))
    tr = np.zeros(n, dtype=bool)
    tr[perm[:cut]] = True
    return {"train": tr, "test": ~tr}


# ============================================================ 균형 표집
def balanced_subject_sample(subject: np.ndarray, n_per_subject: int, seed: int,
                            min_required: int = 0) -> dict:
    """재식별 프로브용 피험자 균형 표집.

    각 피험자에서 최대 n_per_subject 개를 뽑습니다. 개수가 모자란 피험자는
    가진 만큼만 들어가므로 완전 균형이 아닐 수 있습니다 — 그래서 반환값에
    `chance` 를 함께 실어 보냅니다. 이 값을 무시하고 1/58 을 쓰지 마십시오.

    min_required > 0 이면 그보다 적은 피험자는 **제외**하고, 제외 목록을 돌려줍니다
    (민감도 분석: 50개 미만 5명 제외 조건).
    """
    sub = np.asarray(subject).astype(int)
    rng = np.random.default_rng(seed)
    keep, dropped = [], []
    for s in np.unique(sub):
        idx = np.flatnonzero(sub == s)
        if idx.size < min_required:
            dropped.append(int(s))
            continue
        take = idx if idx.size <= n_per_subject else rng.choice(
            idx, size=n_per_subject, replace=False)
        keep.append(np.sort(take))
    sel = np.sort(np.concatenate(keep)) if keep else np.empty(0, int)
    rep = chance_report(sub[sel])
    return {"index": sel, "dropped_subjects": dropped,
            "n_per_subject": int(n_per_subject), "min_required": int(min_required),
            "chance": rep.as_dict()}


# ============================================================ 자기 검사
def selftest(verbose: bool = True) -> dict:
    """torch 없이 도는 분할 로직 검사. gate_minus1 이 호출합니다."""
    res = {}

    # 1) 층화가 무작위보다 균형이 좋아야 한다
    with open(FACTS_PATH, encoding="utf-8") as f:
        facts = json.load(f)
    counts = {int(u): int(v["blink1"]) for u, v in facts["per_user"].items()}
    assign = stratified_folds(counts)
    bal = fold_balance(counts, assign)
    rng = np.random.default_rng(0)
    ev = np.array([counts[s] for s in sorted(counts)])
    rnd = []
    for _ in range(200):
        p = rng.permutation(len(ev))
        load = [ev[p[i::N_FOLDS]].sum() for i in range(N_FOLDS)]
        rnd.append(max(load) / min(load))
    res["fold_balance"] = bal
    res["fold_balance_random_median"] = float(np.median(rnd))
    res["fold_deterministic"] = (stratified_folds(counts) == assign)

    # 2) 시간 블록 / 무작위 분할이 서로 겹치지 않아야 한다
    t = np.linspace(0, 1, 1000)
    tb = time_block_masks(np.zeros(1000, int), t)
    res["time_block_disjoint"] = bool(not (tb["train"] & tb["test"]).any())
    res["time_block_train_frac"] = float(tb["train"].mean())
    rm = random_masks(1000, seed=0)
    res["random_disjoint"] = bool(not (rm["train"] & rm["test"]).any())

    # 3) chance 보고가 불균형을 잡아내야 한다
    y = np.repeat(np.arange(58), [counts[s] for s in sorted(counts)])
    cr = chance_report(y)
    res["chance_unbalanced"] = cr.as_dict()
    res["chance_detects_imbalance"] = bool(
        (not cr.balanced) and cr.majority_baseline > 4 * cr.uniform_chance)

    # 4) 균형 표집 후에는 균등 chance 가 회복되어야 한다
    sel = balanced_subject_sample(y, n_per_subject=10, seed=0)
    res["balanced_sample_chance"] = sel["chance"]
    res["balanced_sample_ok"] = bool(sel["chance"]["balanced"])

    if verbose:
        print(f"  fold  이벤트/fold {bal['events_per_fold']}  "
              f"max/min {bal['max_over_min']:.3f} (무작위였다면 중앙 "
              f"{res['fold_balance_random_median']:.2f})")
        print(f"  chance 불균형: uniform {cr.uniform_chance:.4f} vs "
              f"majority {cr.majority_baseline:.4f}")
        print(f"  균형 표집 후: {ChanceReport(**sel['chance'])}")
    return res
