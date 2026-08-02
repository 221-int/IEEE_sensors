"""임계값 격자 — v2 단일 구현.

왜 이 파일이 존재하는가
----------------------
격자를 실험 스크립트마다 따로 만들면, ours 와 baseline 이 서로 다른 격자를 밟게 되고
그러면 비교표 자체가 무효가 됩니다. 성긴 고정 격자(`np.linspace(...)`)는 특히
**낮은 false-alarm 구간을 통째로 건너뜁니다** — 점수 분포의 꼬리가 격자 한 칸보다
얇으면 그 구간의 임계값이 아예 후보에 없습니다.

그래서 v2의 규칙은 셋입니다.

1. 격자는 **점수 분포의 분위수**로 만든다. 고정 linspace 를 쓰지 않는다.
2. 분위수 간격은 **양 끝이 촘촘한 logit 간격**을 기본으로 한다. 저-FA 구간이 거기 있다.
3. ours 와 baseline 은 **같은 생성 함수 · 같은 파라미터 · 같은 부호 규약**을 통과한다.
   서로 다른 설정을 쓰면 `GridAudit` 이 런 종료 시 예외를 냅니다.

부호 규약
--------
점수마다 "높으면 발화"(깜빡임 확률)와 "낮으면 발화"(EAR)가 섞입니다. 이걸 각 스크립트가
알아서 처리하면 언젠가 한쪽만 뒤집힙니다. 그래서 입구에서 한 번만 정규화합니다.

    canonical(score, higher_fires) -> 항상 "높으면 발화" 로 통일된 점수

이후 모든 함수는 `pred = score >= thr` 하나만 씁니다.

선택/보고 분리
-------------
임계값은 **val 에서 고르고 얼려서 test 에 적용**합니다. API 가 그걸 강제합니다.
`select_threshold()` 는 val 배열만 받고, `evaluate_at()` 은 이미 얼린 스칼라만 받습니다.
test 점수로 격자를 만들 방법이 API 에 없습니다.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Callable, Literal

import numpy as np

# ----------------------------------------------------------------- 기본 설정
# n=257: 분위수 257개. v1 시절의 97보다 촘촘합니다. 격자 해상도가 결론을 바꾼 적이
# 있으므로 넉넉하게 두고, 대신 아래 logit 간격으로 꼬리에 해상도를 몰아줍니다.
DEFAULT_N = 257
# L=8: sigmoid(-8)=3.4e-4, sigmoid(8)=0.99966. 즉 양 끝 분위수 0.03% 까지 도달합니다.
# 음성 표본이 1만 개면 상위 0.03% = 3개까지 짚을 수 있습니다.
DEFAULT_LOGIT_L = 8.0

Spacing = Literal["logit", "linear"]


@dataclass(frozen=True)
class GridConfig:
    """격자 생성 설정. 이 값이 같아야 두 격자가 '같은 격자'입니다.

    `max_exhaustive` 가 이 설계의 핵심입니다. 최적 임계값은 **항상 데이터 점 위**에
    있으므로, 표본이 이 개수 이하이면 격자를 근사하지 않고 유일값 전체를 씁니다.
    그러면 '해상도가 모자라 저-FA 구간을 놓쳤다'는 종류의 오류가 원천적으로 없습니다.
    분위수 근사는 표본이 이보다 클 때만 쓰는 **대비책**입니다.
    """

    max_exhaustive: int = 50_000
    n: int = DEFAULT_N              # 근사 모드에서만 쓰입니다
    spacing: Spacing = "logit"      # 근사 모드에서만 쓰입니다
    logit_l: float = DEFAULT_LOGIT_L

    def key(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


DEFAULT_GRID = GridConfig()


# ----------------------------------------------------------------- 감사 장치
class GridAudit:
    """한 런 안에서 서로 다른 격자 설정이 섞이는 것을 막는 장치.

    격자 '값'은 점수 분포에 따라 달라지는 게 정상입니다. 막아야 하는 것은 **설정**이
    갈리는 경우입니다. 그래서 설정 키만 비교합니다.
    """

    def __init__(self) -> None:
        self._seen: dict[str, list[str]] = {}

    def register(self, name: str, cfg: GridConfig, class_aware: bool) -> None:
        # class_aware 여부도 설정의 일부입니다. ours 는 라벨을 주고 baseline 은 안 주면
        # 두 격자가 다른 방식으로 만들어지므로, 그것도 혼용으로 잡아야 합니다.
        self._seen.setdefault(f"{cfg.key()}|class_aware={class_aware}", []).append(name)

    def report(self) -> dict:
        return {k: sorted(set(v)) for k, v in self._seen.items()}

    def require_uniform(self) -> None:
        if len(self._seen) > 1:
            raise RuntimeError(
                "한 런에서 서로 다른 임계값 격자 설정이 쓰였습니다. 비교가 무효입니다.\n"
                + json.dumps(self.report(), ensure_ascii=False, indent=2)
            )

    def reset(self) -> None:
        self._seen.clear()


AUDIT = GridAudit()


# ----------------------------------------------------------------- 부호 규약
def canonical(score: np.ndarray, higher_fires: bool) -> np.ndarray:
    """점수를 항상 '높으면 발화'로 통일합니다.

    higher_fires=False (예: EAR — 낮을수록 감은 눈) 이면 부호를 뒤집습니다.
    이 함수를 통과한 뒤에는 어떤 코드도 부등호 방향을 다시 생각하지 않습니다.
    """
    s = np.asarray(score, dtype=np.float64)
    return s if higher_fires else -s


# ----------------------------------------------------------------- 격자 생성
def _quantile_levels(cfg: GridConfig) -> np.ndarray:
    if cfg.spacing == "linear":
        return np.linspace(0.0, 1.0, cfg.n)
    if cfg.spacing == "logit":
        z = np.linspace(-cfg.logit_l, cfg.logit_l, cfg.n)
        return 1.0 / (1.0 + np.exp(-z))
    raise ValueError(f"unknown spacing {cfg.spacing!r}")


def threshold_grid(
    score_canonical: np.ndarray,
    y: np.ndarray | None = None,
    cfg: GridConfig = DEFAULT_GRID,
    name: str = "unnamed",
) -> np.ndarray:
    """canonical 점수 -> 임계값 격자 (오름차순, 중복 제거).

    `score_canonical` 은 반드시 `canonical()` 을 통과한 **val 점수**여야 합니다.

    두 모드 — Phase −1 게이트가 잡아낸 설계 오류의 결과
    -------------------------------------------------
    처음에는 '양 끝이 촘촘한 logit 분위수'면 저-FA 구간을 짚는다고 생각했는데
    **틀렸습니다.** 양성+음성이 섞인 점수에서 분포의 양 끝은 '아주 확실한 음성'과
    '아주 확실한 양성' 쪽입니다. 정작 낮은 false-alarm 을 결정하는 지점은
    **음성 분포의 위쪽 꼬리**이고, 섞인 분포에서는 그게 한가운데쯤에 있습니다.
    게이트가 이걸 숫자로 잡았습니다(해상도 활용률 23%).

    그래서 v2 는 근사를 아예 그만둡니다.

      exhaustive (기본)  유일값 <= cfg.max_exhaustive 이면 **유일값 전체 + 중점**.
                         최적 임계값은 항상 데이터 점 위에 있으므로 이게 정확해가며,
                         '해상도 부족' 이라는 실패 모드가 존재하지 않습니다.
                         이산 코드(K=4)도 자동으로 옳게 처리됩니다.
      quantile (대비책)  표본이 그보다 크면 클래스별 분위수를 합쳐 근사합니다.
                         라벨이 있으면 음성의 위쪽 꼬리 / 양성의 아래쪽 꼬리에
                         해상도를 몰아줍니다.

    ours 와 baseline 이 **똑같이** 이 함수를 통과하므로 공정성은 유지됩니다
    (한쪽만 라벨을 주면 AUDIT 이 혼용으로 잡습니다).

    격자에는 항상 아래 두 경계가 포함됩니다.
        최소값 바로 아래 : 전부 발화 (recall 1, FA 1)
        +inf             : 아무것도 발화 안 함 (recall 0, FA 0)
    """
    AUDIT.register(name, cfg, class_aware=y is not None)
    s_all = np.asarray(score_canonical, dtype=np.float64)
    finite = np.isfinite(s_all)
    s = s_all[finite]
    if s.size == 0:
        raise ValueError(f"[{name}] 유한한 점수가 하나도 없습니다.")

    uniq = np.unique(s)
    if uniq.size <= cfg.max_exhaustive:
        mids = (uniq[:-1] + uniq[1:]) / 2.0 if uniq.size > 1 else np.empty(0)
        parts = [uniq, mids]
    else:
        lin = GridConfig(max_exhaustive=cfg.max_exhaustive, n=cfg.n,
                         spacing="linear", logit_l=cfg.logit_l)
        parts = [np.quantile(s, _quantile_levels(lin))]
        if y is not None:
            yy = np.asarray(y).ravel()[finite]
            lv = _quantile_levels(cfg)          # cfg.spacing (기본 logit)
            for cls in np.unique(yy):
                sc = s[yy == cls]
                if sc.size >= 2:
                    parts.append(np.quantile(sc, lv))

    lo = np.nextafter(float(uniq[0]), -np.inf)
    grid = np.concatenate([*parts, [lo, np.inf]])
    return np.unique(grid[np.isfinite(grid) | np.isposinf(grid)])


def grid_mode(n_unique: int, cfg: GridConfig = DEFAULT_GRID) -> str:
    return "exhaustive" if n_unique <= cfg.max_exhaustive else "quantile"


# ----------------------------------------------------------------- 스윕/지표
def sweep(
    score_canonical: np.ndarray,
    y_true: np.ndarray,
    grid: np.ndarray,
) -> dict[str, np.ndarray]:
    """격자 전체에 대한 혼동행렬 벡터화 스윕.

    반환 키: thr, tp, fp, fn, tn, recall, precision, fa (= fp / n_neg), accuracy
    `fa` 는 **음성 표본 기준 false-alarm rate** 입니다. 프레임 FA 와 구분하십시오 —
    v2 에서 연속 프레임 FA 는 측정하지 않습니다(라벨 커버리지 문제).
    """
    s = np.asarray(score_canonical, dtype=np.float64)
    y = np.asarray(y_true).astype(np.int64)
    if s.shape[0] != y.shape[0]:
        raise ValueError(f"길이 불일치: score {s.shape[0]} vs y {y.shape[0]}")
    g = np.asarray(grid, dtype=np.float64)

    order = np.argsort(s, kind="stable")
    ss, yy = s[order], y[order]
    n_pos, n_neg = int(yy.sum()), int((yy == 0).sum())
    # cum_pos[i] = ss[:i] 안의 양성 수  ->  thr 이상 발화 시 tp = n_pos - cum_pos[idx]
    cum_pos = np.concatenate([[0], np.cumsum(yy)])
    cum_all = np.arange(len(yy) + 1)

    idx = np.searchsorted(ss, g, side="left")
    tp = n_pos - cum_pos[idx]
    fired = len(yy) - cum_all[idx]
    fp = fired - tp
    fn = n_pos - tp
    tn = n_neg - fp

    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(n_pos > 0, tp / max(n_pos, 1), np.nan)
        precision = np.where(fired > 0, tp / np.maximum(fired, 1), np.nan)
        fa = np.where(n_neg > 0, fp / max(n_neg, 1), np.nan)
    accuracy = (tp + tn) / len(yy)
    return dict(thr=g, tp=tp, fp=fp, fn=fn, tn=tn,
                recall=recall, precision=precision, fa=fa, accuracy=accuracy)


# ----------------------------------------------------------------- 선택 기준
Criterion = Callable[[dict[str, np.ndarray]], np.ndarray]


def crit_accuracy() -> Criterion:
    """균형 1:1 이벤트 분류의 기본 기준."""
    return lambda sw: sw["accuracy"]


def crit_f1() -> Criterion:
    def f(sw):
        p, r = sw["precision"], sw["recall"]
        with np.errstate(invalid="ignore"):
            v = 2 * p * r / (p + r)
        return np.nan_to_num(v, nan=-1.0)
    return f


def crit_recall_at_fa(budget: float) -> Criterion:
    """FA <= budget 안에서 recall 최대. 예산 밖은 -1 로 눌러 후보에서 뺍니다."""
    def f(sw):
        v = np.where(sw["fa"] <= budget, sw["recall"], -1.0)
        return np.nan_to_num(v, nan=-1.0)
    return f


def select_threshold(
    val_score_canonical: np.ndarray,
    val_y: np.ndarray,
    criterion: Criterion = None,
    cfg: GridConfig = DEFAULT_GRID,
    name: str = "unnamed",
) -> dict:
    """val 에서만 임계값을 고르고 얼립니다.

    반환: {thr, criterion_value, val_metrics(dict of scalars), grid_size, grid_cfg}
    이 함수는 test 배열을 받지 않습니다. 받을 수 없게 만들어 둔 것이 요점입니다.
    """
    criterion = criterion or crit_accuracy()
    grid = threshold_grid(val_score_canonical, y=val_y, cfg=cfg, name=name)
    sw = sweep(val_score_canonical, val_y, grid)
    obj = criterion(sw)
    # 동점이면 **가장 보수적인(높은) 임계값**을 고릅니다. 결정적이어야 하므로
    # argmax 대신 마지막 최댓값 위치를 씁니다.
    best = int(np.max(np.flatnonzero(obj == np.max(obj))))
    return dict(
        thr=float(grid[best]),
        criterion_value=float(obj[best]),
        val_metrics={k: float(sw[k][best]) for k in
                     ("recall", "precision", "fa", "accuracy")},
        val_counts={k: int(sw[k][best]) for k in ("tp", "fp", "fn", "tn")},
        grid_size=int(grid.size),
        grid_cfg=asdict(cfg),
        name=name,
    )


def evaluate_at(
    test_score_canonical: np.ndarray,
    test_y: np.ndarray,
    thr: float,
) -> dict:
    """얼린 임계값 하나를 test 에 적용합니다. 스칼라만 받습니다."""
    if not np.isscalar(thr):
        raise TypeError("thr 은 val 에서 얼린 스칼라여야 합니다.")
    sw = sweep(test_score_canonical, test_y, np.array([float(thr)]))
    out = {k: float(sw[k][0]) for k in ("recall", "precision", "fa", "accuracy")}
    out.update({k: int(sw[k][0]) for k in ("tp", "fp", "fn", "tn")})
    out["thr"] = float(thr)
    return out


# ----------------------------------------------------------------- 임계값 무관
def average_precision(score_canonical: np.ndarray, y_true: np.ndarray) -> float:
    """PR-AUC (average precision). 임계값 선택과 무관한 지표.

    sklearn 의존을 피하려고 직접 구현합니다(Pi 쪽 환경까지 같은 코드를 씁니다).
    동점 점수는 함께 처리합니다 — 동점을 무시하면 값이 낙관적으로 부풀려집니다.
    """
    s = np.asarray(score_canonical, dtype=np.float64)
    y = np.asarray(y_true).astype(np.int64)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="stable")
    ss, yy = s[order], y[order]
    # 동점 그룹 경계
    bound = np.flatnonzero(np.concatenate([np.diff(ss) != 0, [True]]))
    tp = np.cumsum(yy)[bound]
    fired = (bound + 1).astype(np.float64)
    precision = tp / fired
    recall = tp / n_pos
    prev_r = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - prev_r) * precision))


def roc_auc(score_canonical: np.ndarray, y_true: np.ndarray) -> float:
    """ROC-AUC. 동점은 평균 순위로 처리합니다(Mann-Whitney U).

    균형 1:1 이벤트 분류의 임계값 무관 지표입니다. 동점을 무시하고 계산하면
    이산 점수(코드·양자화 출력)에서 값이 부풀려집니다.
    """
    s = np.asarray(score_canonical, np.float64)
    y = np.asarray(y_true).astype(np.int64)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s), np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(s):                      # 동점 그룹에 평균 순위 부여
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def grid_fingerprint(grid: np.ndarray) -> str:
    """격자 값 자체의 해시. 리포트에 남겨 두면 사후 감사가 됩니다."""
    return hashlib.sha256(np.asarray(grid, np.float64).tobytes()).hexdigest()[:16]
