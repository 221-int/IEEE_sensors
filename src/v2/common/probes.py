"""프로브(신원 누출 측정) — v2 단일 구현.

왜 이 파일이 존재하는가
----------------------
표현 A 와 표현 B 의 신원 누출을 비교하려면, **프로브의 용량과 학습 절차와 분할이
완전히 같아야** 합니다. 표현마다 프로브를 따로 짜면 무엇이 표현 차이이고 무엇이
프로브 차이인지 알 수 없습니다. 그래서 여기 하나만 둡니다.

선형과 MLP 를 **둘 다** 냅니다. 이유가 있습니다 — 사후 선형 투영 제거 같은 기법은
선형 프로브만 떨어뜨리고 MLP 프로브는 그대로 둘 수 있습니다. 그러면 그건 프라이버시
기전이 아니라 프로브 회피입니다. 한 종류만 재면 그 구분이 불가능합니다.

표준화 주의
-----------
입력 표준화(mean/std)는 **train 에서만 적합**하고 test 에 적용합니다. 전체에서
적합하면 test 통계가 학습에 새어 들어갑니다. 표현마다 스케일이 달라 학습 난이도가
갈리는 것을 막으려면 표준화는 필요하지만, 새는 방식이면 안 됩니다.

chance
------
모든 결과에 `splits.chance_report()` 가 붙습니다. mEBAL2 58명은 이벤트 수가
불균형이라 1/58 을 chance 로 쓰면 틀립니다. 이 모듈은 균등 chance 와 다수 클래스
기준선을 **항상 같이** 반환합니다.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from src.v2.common import repro, splits

ProbeKind = Literal["linear", "mlp"]

DEFAULT_EPOCHS = 60
DEFAULT_BATCH = 256
DEFAULT_LR = 1e-3
DEFAULT_HIDDEN = 128
DEFAULT_DROPOUT = 0.3


def _torch():
    try:
        import torch  # noqa: PLC0415
        return torch
    except ImportError as e:
        raise ImportError(
            "프로브에는 torch 가 필요합니다. 데스크탑(CUDA) 환경에서 실행하십시오.") from e


def _standardize(X: np.ndarray, train: np.ndarray) -> np.ndarray:
    """train 통계로만 표준화."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X[:, None]
    mu = X[train].mean(0, keepdims=True)
    sd = X[train].std(0, keepdims=True) + 1e-6
    return (X - mu) / sd


def _build(kind: ProbeKind, d: int, n_cls: int, hidden: int, dropout: float):
    torch = _torch()
    nn = torch.nn
    if kind == "linear":
        return nn.Linear(d, n_cls)
    if kind == "mlp":
        return nn.Sequential(nn.Linear(d, hidden), nn.ReLU(True),
                             nn.Dropout(dropout), nn.Linear(hidden, n_cls))
    raise ValueError(kind)


def run_probe(
    X: np.ndarray,
    y: np.ndarray,
    masks: dict[str, np.ndarray],
    kind: ProbeKind,
    seed: int,
    epochs: int = DEFAULT_EPOCHS,
    batch: int = DEFAULT_BATCH,
    lr: float = DEFAULT_LR,
    hidden: int = DEFAULT_HIDDEN,
    dropout: float = DEFAULT_DROPOUT,
    device: str | None = None,
) -> dict:
    """단일 프로브 1회 학습 -> test 정확도.

    masks: {"train": bool[n], "test": bool[n]}  (splits.time_block_masks 등)
    반환: {accuracy, chance, n_train, n_test, kind, seed, dim}
    """
    torch = _torch()
    repro.seal(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    y = np.asarray(y).astype(np.int64).ravel()
    tr, te = np.asarray(masks["train"], bool), np.asarray(masks["test"], bool)
    if (tr & te).any():
        raise AssertionError("train/test 마스크가 겹칩니다.")
    if tr.sum() == 0 or te.sum() == 0:
        raise ValueError(f"빈 분할입니다: train {tr.sum()} test {te.sum()}")

    Xs = _standardize(X, tr)
    classes = np.unique(y)
    remap = {c: i for i, c in enumerate(classes)}
    yy = np.array([remap[v] for v in y], np.int64)

    net = _build(kind, Xs.shape[1], len(classes), hidden, dropout).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = torch.nn.CrossEntropyLoss()

    ds = torch.utils.data.TensorDataset(torch.from_numpy(Xs[tr]),
                                        torch.from_numpy(yy[tr]))
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True,
                                     **repro.dataloader_kwargs(seed))
    for _ in range(epochs):
        net.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            lossf(net(xb), yb).backward()
            opt.step()

    net.eval()
    with torch.no_grad():
        pred = net(torch.from_numpy(Xs[te]).to(device)).argmax(1).cpu().numpy()
    acc = float((pred == yy[te]).mean())

    return {
        "accuracy": acc,
        "kind": kind,
        "seed": int(seed),
        "dim": int(Xs.shape[1]),
        "n_train": int(tr.sum()),
        "n_test": int(te.sum()),
        # chance 는 **평가 집합 기준**으로 보고합니다. 여기가 우리가 맞히는 대상입니다.
        "chance": splits.chance_report(y[te]).as_dict(),
    }


def run_representation(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    subject: np.ndarray,
    t_rel: np.ndarray,
    seeds: tuple[int, ...] = (0, 1, 2),
    kinds: tuple[ProbeKind, ...] = ("linear", "mlp"),
    **kw,
) -> dict:
    """표현 하나에 대해 (선형/MLP) x (시간블록/무작위) x 시드 를 전부 돌립니다.

    v2 재식별 보고의 최소 단위가 이 **4칸 x 3시드** 입니다. 한 칸만 인용하지 마십시오.
    """
    out: dict = {"rep": name, "dim": int(np.atleast_2d(X).shape[-1]), "cells": {}}
    split_defs = {
        "time_block": lambda s: splits.time_block_masks(subject, t_rel),
        "random": lambda s: splits.random_masks(len(y), seed=s),
    }
    for split_name, make in split_defs.items():
        for kind in kinds:
            accs, runs = [], []
            for s in seeds:
                r = run_probe(X, y, make(s), kind, seed=s, **kw)
                accs.append(r["accuracy"])
                runs.append(r)
            out["cells"][f"{split_name}/{kind}"] = {
                "mean": float(np.mean(accs)),
                "std": float(np.std(accs)),
                "per_seed": accs,
                "chance": runs[0]["chance"],
                "n_train": runs[0]["n_train"],
                "n_test": runs[0]["n_test"],
            }
    tb = out["cells"]["time_block/linear"]["mean"]
    rd = out["cells"]["random/linear"]["mean"]
    out["random_minus_timeblock"] = float(rd - tb)
    return out


# ------------------------------------------------------------ 집계 공격
def aggregate(X: np.ndarray, subject: np.ndarray, n: int, seed: int,
              mode: Literal["mean", "hist"] = "mean",
              n_bins: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """다중 이벤트 집계 공격용 특징.

    한 사람이 이벤트를 수백 개 내보내므로, 표현 하나의 누출 상한이 낮아도 **N개를
    모으면 신원이 복원될 수 있습니다.** N in {1, 10, 100} 로 이 곡선을 그립니다.

    mode="mean": 무작위로 고른 N개 표현의 평균 (연속 표현용)
    mode="hist": N개 코드의 히스토그램 (이산 코드용, n_bins=K 필요)
                 -> 이산 병목 설계안의 상한이 다중 관측에서 깨지는지 재는 경로입니다.
    """
    rng = np.random.default_rng(seed)
    sub = np.asarray(subject).astype(int)
    Xa = np.atleast_2d(np.asarray(X))
    if Xa.shape[0] != sub.shape[0]:
        Xa = Xa.T
    feats, labs = [], []
    for s in np.unique(sub):
        idx = np.flatnonzero(sub == s)
        n_groups = idx.size // n
        if n_groups == 0:
            continue
        perm = rng.permutation(idx)[: n_groups * n].reshape(n_groups, n)
        for g in perm:
            if mode == "mean":
                feats.append(Xa[g].mean(0))
            elif mode == "hist":
                if n_bins <= 0:
                    raise ValueError("mode='hist' 에는 n_bins=K 가 필요합니다.")
                codes = Xa[g].ravel().astype(int)
                feats.append(np.bincount(codes, minlength=n_bins) / n)
            else:
                raise ValueError(mode)
            labs.append(s)
    if not feats:
        raise ValueError(f"N={n} 로 묶을 수 있는 피험자가 없습니다.")
    return np.asarray(feats, np.float32), np.asarray(labs, np.int64)


# ------------------------------------------------------------ 사후 투영 제거
def fit_identity_subspace(Z: np.ndarray, subject: np.ndarray, k: int) -> np.ndarray:
    """피험자 평균들이 만드는 상위 k차원 방향 -> (d, k) 정규직교 기저.

    사후 제거용입니다. 이 기저로 투영 제거한 표현이 **선형 프로브만** 떨어지고
    MLP 프로브는 그대로면, 그건 프라이버시 기전이 아니라 프로브 회피입니다.
    그 판정을 위해 제거 전/후 x 선형/MLP 4칸을 전부 재야 합니다.
    """
    Z = np.asarray(Z, np.float64)
    sub = np.asarray(subject).astype(int)
    means = np.stack([Z[sub == s].mean(0) for s in np.unique(sub)])
    means = means - means.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(means, full_matrices=False)
    return np.ascontiguousarray(vt[:k].T)


def project_out(Z: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """z <- (I - V V^T) z. 배포 비용은 d x d 행렬곱 1회입니다."""
    Z = np.asarray(Z, np.float64)
    V = np.asarray(basis, np.float64)
    return (Z - Z @ V @ V.T).astype(np.float32)


# ------------------------------------------------------------ 자기 검사
def selftest_numpy_only() -> dict:
    """torch 없이 확인 가능한 부분(집계·투영 제거)만 검사합니다."""
    rng = np.random.default_rng(0)
    sub = np.repeat(np.arange(5), 40)
    ident = rng.normal(size=(5, 8))
    Z = ident[sub] + 0.1 * rng.normal(size=(200, 8))

    Xa, ya = aggregate(Z, sub, n=10, seed=0, mode="mean")
    codes = rng.integers(0, 4, size=(200, 1))
    Xh, yh = aggregate(codes, sub, n=10, seed=0, mode="hist", n_bins=4)

    V = fit_identity_subspace(Z, sub, k=4)
    Zp = project_out(Z, V)

    def scatter(A: np.ndarray) -> float:
        """피험자 간 산포(=중심화한 피험자 평균들의 노름).

        중심화가 필요합니다. 투영 제거는 **전역 평균**을 건드리지 않으므로,
        중심화 없이 재면 모든 피험자에 공통인 전역 평균 성분이 남아
        '제거가 덜 됐다'로 잘못 읽힙니다. 우리가 지우려는 것은 피험자 간 차이입니다.
        """
        m = np.stack([A[sub == s].mean(0) for s in np.unique(sub)])
        return float(np.linalg.norm(m - m.mean(0, keepdims=True)))

    before, after = scatter(Z), scatter(Zp)
    return {
        "aggregate_mean_shape": list(Xa.shape),
        "aggregate_mean_groups_per_subject": int((ya == 0).sum()),
        "aggregate_hist_rowsum_is_one": bool(np.allclose(Xh.sum(1), 1.0)),
        "between_subject_scatter_before": before,
        "between_subject_scatter_after": after,
        "projection_removes_class_means": bool(after < 0.05 * before),
    }
