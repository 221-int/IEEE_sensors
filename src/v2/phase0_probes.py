"""Phase 0 — 학습 전에 재식별의 좌표축을 만든다.

    python -m src.v2.phase0_probes

왜 학습보다 먼저인가
------------------
나중에 우리 벡터가 재식별 0.4 를 기록했다고 하자. 그 숫자만으로는 **원인을 모른다.**
목적함수 때문인가, 눈 크롭 자체가 원래 신원 판별적이라서인가, 아니면 조명인가.
학습을 하기 전에 아래를 재두면 그 해석이 가능해진다.

    (a) 랜덤 초기화 인코더  학습을 안 했는데도 새면 → **원인은 목적함수가 아니라 입력**이다
    (b) 원본 픽셀           재식별의 천장. 어떤 표현도 이보다 더 샐 수는 없다
    (c) 픽셀 PCA-32         차원만 줄였을 때의 값
    (d) (밝기, 선명도)      세션 광학 서명. 이미 측정됨(0.2018) — 여기서 재확인
    (e) EAR 스칼라          최소 정보 기준선

정규화 비교
----------
같은 프로브를 `norm=none`(/255) 과 `norm=frame_standardize` 두 조건에서 돌린다.
**프레임 표준화가 실제로 누출을 얼마나 줄이는지**를 학습 전에 확정하기 위해서다.

모든 프로브는 피험자 **균형 표집**을 쓰고 선형·MLP × 시간블록·무작위 4칸을 낸다.
chance 는 균등(1/58 = 0.0172)과 다수 클래스 둘 다 보고한다.

산출물: results/v2/phase0_probes.json
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from src.v2.common import probes, repro, splits
from src.v2.dataset import crop as C

DATA = "data/processed/v2"
OUT = "results/v2/phase0_probes.json"


def randomized_pca(X: np.ndarray, k: int, seed: int = 0, oversample: int = 10):
    """train 에서 적합하는 랜덤 투영 기반 PCA (sklearn 의존 없이).

    -> (mean, components (k, d)).  X 는 (n, d) float32.
    """
    rng = np.random.default_rng(seed)
    mu = X.mean(0, keepdims=True)
    A = X - mu
    Om = rng.standard_normal((A.shape[1], k + oversample), dtype=np.float32)
    Y = A @ Om
    Q, _ = np.linalg.qr(Y)
    for _ in range(2):                      # power iteration (정확도 보강)
        Q, _ = np.linalg.qr(A.T @ Q)
        Q, _ = np.linalg.qr(A @ Q)
    B = Q.T @ A
    _, _, Vt = np.linalg.svd(B, full_matrices=False)
    return mu, Vt[:k]


def make_pca_transform(k: int, seed: int = 0):
    """probes.run_representation 의 fit_transform 훅. **train 에서만 적합한다.**"""
    def _fn(X: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
        mu, V = randomized_pca(np.asarray(X, np.float32)[train_mask], k, seed)
        return ((np.asarray(X, np.float32) - mu) @ V.T).astype(np.float32)
    return _fn


def main() -> int:
    repro.ensure_hashseed()
    repro.seal(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--margin-tag", default="m22")
    ap.add_argument("--n-per-subject", type=int, default=200)
    ap.add_argument("--pca-dim", type=int, default=32)
    ap.add_argument("--latent", type=int, default=16)
    ap.add_argument("--arch", default="vpres")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--only", default=None,
                    help="표현 이름에 이 문자열이 든 것만 돌린다. 예: --only 광학")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        raise SystemExit("torch 가 필요합니다. 데스크탑(CUDA)에서 실행하십시오.")

    idx = np.load(os.path.join(args.data, "index.npz"))
    frames = np.load(os.path.join(args.data, f"frames_{args.margin_tag}.npy"),
                     mmap_mode="r")
    subj_all = idx["f_subject"].astype(int)
    if len(subj_all) != frames.shape[0]:
        raise SystemExit("index 와 memmap 길이가 다릅니다. phase3_merge 를 다시 도십시오.")

    samp = splits.balanced_subject_sample(subj_all, args.n_per_subject, seed=0)
    sel = samp["index"]
    subj = subj_all[sel]
    trel = idx["f_t_rel"][sel].astype(np.float64)
    ear = idx["f_ear"][sel][:, 2].astype(np.float32)          # mean EAR
    bright = idx[f"f_brightness_{args.margin_tag}"][sel].astype(np.float32)
    sharp = idx[f"f_sharpness_{args.margin_tag}"][sel].astype(np.float32)

    print(f"표집 {len(sel):,}장  {splits.ChanceReport(**samp['chance'])}")
    raw = np.asarray(frames[sel])                              # (n, 64, 160) uint8
    print(f"픽셀 로드 {raw.nbytes / 1e6:.0f} MB")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out: dict = {"env": repro.env_fingerprint(), "margin_tag": args.margin_tag,
                 "sample": samp["chance"], "arch": args.arch, "latent": args.latent,
                 "results": {}}

    for norm in ("none", "frame_standardize"):
        px = C.batch_input(raw, norm=norm)[:, 0]               # (n, 64, 160) float32
        flat = px.reshape(len(px), -1)

        # 랜덤 초기화 인코더 (학습 없음)
        repro.seal(0)
        enc = None
        try:
            from src.v2.model import encoder as E
            enc = E.build(args.arch, args.latent).to(dev).eval()
            zs = []
            with torch.no_grad():
                for i in range(0, len(px), 512):
                    xb = torch.from_numpy(px[i:i + 512]).unsqueeze(1).to(dev)
                    zs.append(enc(xb).cpu().numpy())
            Zrand = np.concatenate(zs)
        except Exception as e:                                  # noqa: BLE001
            print(f"  ! 랜덤 인코더 생략: {type(e).__name__}: {e}")
            Zrand = None

        # (d') 정규화된 **픽셀에서 직접 재계산한** 광학 지표.
        # (d) 는 인덱스에 저장된 원본 값이라 정규화의 영향을 받지 않는다 — 두 조건에서
        # 값이 똑같이 나와 "정규화가 광학 교란을 줄이는가"를 답하지 못했다.
        # 여기서는 실제 입력 이미지에 같은 추출기를 적용하므로 두 조건이 비교 가능하다.
        # frame_standardize 후에는 평균 0 · 표준편차 1 이 되어 밝기·대비가 상수가 되고,
        # 남는 것은 라플라시안 분산(= 원본 sharpness / contrast^2)뿐이다.
        ph = np.array([[p["brightness"], p["contrast"], p["sharpness"]]
                       for p in (C.photometrics(im) for im in px)], np.float32)
        reps = [
            ("(b) 원본 픽셀", flat, None),
            (f"(c) 픽셀 PCA-{args.pca_dim}", flat, make_pca_transform(args.pca_dim)),
            ("(d) 밝기+선명도 (저장값)", np.stack([bright, sharp], 1), None),
            ("(d') 광학 재계산 3차원", ph, None),
            ("(e) EAR 스칼라", ear[:, None], None),
        ]
        if Zrand is not None:
            reps.insert(0, (f"(a) 랜덤 인코더 D={args.latent}", Zrand, None))
        if args.only:
            reps = [r for r in reps if args.only in r[0]]
            if not reps:
                raise SystemExit(f"--only '{args.only}' 에 맞는 표현이 없습니다.")

        print(f"\n=== norm = {norm} ===")
        print(f"  {'표현':<26}{'차원':>7}{'TB-lin':>9}{'TB-mlp':>9}{'RD-lin':>9}{'RD-mlp':>9}")
        out["results"][norm] = {}
        for name, X, ft in reps:
            t0 = time.perf_counter()
            r = probes.run_representation(name, X, subj, subj, trel,
                                          seeds=tuple(args.seeds), epochs=args.epochs,
                                          fit_transform=ft, device=dev)
            c = r["cells"]
            print(f"  {name:<26}{r['dim']:>7}"
                  f"{c['time_block/linear']['mean']:>9.4f}{c['time_block/mlp']['mean']:>9.4f}"
                  f"{c['random/linear']['mean']:>9.4f}{c['random/mlp']['mean']:>9.4f}"
                  f"   ({time.perf_counter() - t0:.0f}s)")
            out["results"][norm][name] = r

        # --only 로 일부만 돌렸으면 기존 파일을 덮어쓰지 않고 병합한다
        prev = {}
        if os.path.exists(args.out) and os.path.getsize(args.out) > 0:
            try:
                prev = json.load(open(args.out, encoding="utf-8"))
            except json.JSONDecodeError:
                prev = {}
        merged = prev if args.only and prev.get("results") else out
        if merged is prev:
            for nm, block in out["results"].items():
                merged.setdefault("results", {}).setdefault(nm, {}).update(block)
            merged["env"] = out["env"]
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        tmp = args.out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=1)
        os.replace(tmp, args.out)

    print(f"\n  chance = {samp['chance']['uniform_chance']:.4f} "
          f"(다수 클래스 {samp['chance']['majority_baseline']:.4f})")
    print("  TB = 시간블록 분할, RD = 무작위 분할")
    print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
