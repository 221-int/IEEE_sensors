"""Phase 5 — 학습된 우리 벡터가 신원을 얼마나 흘리는가. PROTOCOL §11 Phase 5 게이트.

    python -m src.v2.phase5_reid

게이트: 4칸(선형·MLP × 시간블록·무작위) + 집계 공격 N∈{1,10,100} **보고 완료**.
특정 임계값 미달을 실패로 두지 않는다 — 이번 라운드에서 재식별은 **정량화 대상**이지
성립 조건이 아니다(§11). 억제 기전은 범위 밖이므로 **개선 주장을 하지 않는다**(§8-ter).

왜 기준선을 다시 재는가 — 🔴 중요
--------------------------------
`results/v2/phase0_probes.json` 의 해석 기준선(랜덤 인코더 0.6329 / 픽셀 0.9099 /
광학 0.2018 / EAR 0.0508)은 **58명·좌석 필터 이전·전체 프레임**에서 나왔다
(`sample.n_classes = 58`). 우리 벡터는 **57명·유효 이벤트 프레임**에서 잰다.
모집단이 다르므로 **그 숫자를 그대로 나란히 놓으면 안 된다.**

그래서 이 스크립트는 **같은 표집에서 기준선을 전부 다시 잰다.** Phase 0 값은
참고로만 싣고 "비교 불가"를 JSON 에 명시한다.

무엇을 재는가
------------
    ours        학습된 인코더 15개(fold×seed) 각각의 D=16 벡터
    (a)         랜덤 초기화 인코더 — 학습의 기여
    (b)         원본 픽셀 — 천장
    (c)         픽셀 PCA-32 — 차원만 줄였을 때
    (d')        광학 3차원(밝기·대비·선명도, 실제 입력에서 재계산) — 조명 바닥
    (e)         EAR 스칼라 — 최소 정보 기준선

**seen / unseen 분해**: 인코더 하나는 57명 중 일부만 학습에 봤다. 학습에 본 사람의
누출이 더 크면, 재식별 숫자는 "표현이 신원을 담는다"가 아니라 "그 사람을 외웠다"를
섞어 재는 것이다. 시간블록 MLP 칸에서 피험자별 정확도를 갈라 낸다.

표집 (§6)
--------
유효 이벤트에 실제로 쓰인 프레임만 쓴다(좌석·결측 필터 통과분, U18 제외).
피험자당 200장 — Phase 0 과 같은 값이고 57명 전원이 충족한다(최소 U46 378장).

집계 공격은 200장으로는 N=100 에서 피험자당 2묶음뿐이라 학습이 불가능하다.
그래서 **별도 코호트**(피험자당 `--agg-per-subject`, 미달자 제외)를 쓰고 인원과
chance 를 따로 보고한다. §6 의 "두 숫자가 크게 다르면 그 자체를 보고" 규칙을 따른다.

chance (§5): 균등과 다수 클래스를 **항상 병기**한다. 1/57 만 쓰지 않는다.
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
MODELS = "models/v2/train_encoder_final"   # 2026-08-07: 체크포인트 경로가
# --out 에서 파생되도록 바뀌었다(train_encoder.models_dir). 확정 런의 --out 이
# results/v2/train_encoder_final.json 이므로 여기다.
OUT = "results/v2/phase5_reid.json"

# Phase 0 값 — **비교 불가**. 모집단이 다르다(58명 / 좌석 필터 이전 / 전체 프레임).
PHASE0_58 = {"random_encoder_tb_mlp": 0.6329, "pixels_tb_mlp": 0.9099,
             "pca32_tb_mlp": 0.8656, "photometric_tb_mlp": 0.2018,
             "ear_tb_mlp": 0.0508, "uniform_chance": 1 / 58}


def valid_frame_mask(idx) -> np.ndarray:
    """유효 이벤트에 실제로 쓰인 프레임만 True.

    좌석 오염 이벤트와 결측 초과 이벤트를 뺀 뒤 남은 프레임이다. 이걸 안 하면
    **옆자리 사람의 얼굴이 들어간 프레임으로 재식별을 재게 된다** — 그 숫자는
    표현의 누출량이 아니다.
    """
    v = idx["e_valid"].astype(bool)
    rows = idx["e_rows"][v]
    m = np.zeros(len(idx["f_subject"]), bool)
    r = rows[rows >= 0]
    m[r] = True
    return m


def randomized_pca(X: np.ndarray, k: int, seed: int = 0, oversample: int = 10):
    rng = np.random.default_rng(seed)
    mu = X.mean(0, keepdims=True)
    Xc = X - mu
    P = rng.standard_normal((Xc.shape[1], k + oversample), dtype=np.float32)
    Q, _ = np.linalg.qr(Xc @ P)
    B = Q.T @ Xc
    _, _, Vt = np.linalg.svd(B, full_matrices=False)
    return mu, Vt[:k]


def make_pca_transform(k: int, seed: int = 0):
    def f(X, train_mask):
        mu, comp = randomized_pca(np.asarray(X, np.float32)[train_mask], k, seed)
        return (np.asarray(X, np.float32) - mu) @ comp.T
    return f


def encode_all(px: np.ndarray, state_front, arch: str, latent: int, dev, batch: int = 512):
    """프레임 → D차원 벡터. state_front 가 None 이면 랜덤 초기화 인코더."""
    import torch
    from src.v2.model import encoder as E
    enc = E.build(arch, latent).to(dev)
    if state_front is not None:
        enc.load_state_dict(state_front)
    enc.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(px), batch):
            xb = torch.from_numpy(px[i:i + batch]).unsqueeze(1).to(dev)
            out.append(enc(xb).cpu().numpy())
    return np.concatenate(out)


def seen_unseen(Z: np.ndarray, subj: np.ndarray, trel: np.ndarray,
                fold: int, seeds: list[int], **kw) -> dict:
    """시간블록 MLP 칸에서 학습에 본 피험자 / 못 본 피험자의 누출을 가른다."""
    assign = splits.load_folds()
    rot = splits.fold_rotation(fold)
    role = {u: ("seen" if assign[u] in rot["train"] else "unseen") for u in assign}
    accs = {"seen": [], "unseen": []}
    for s in seeds:
        m = splits.time_block_masks(subj, trel)
        r = probes.run_probe(Z, subj, m, "mlp", seed=s, return_pred=True, **kw)
        yt, pr = r["y_test"], r["pred"]
        for tag in ("seen", "unseen"):
            k = np.array([role[int(u)] == tag for u in yt])
            if k.any():
                accs[tag].append(float((pr[k] == yt[k]).mean()))
    return {t: {"mean": float(np.mean(v)), "n_seeds": len(v)} for t, v in accs.items() if v}


def main() -> int:
    repro.ensure_hashseed()
    repro.seal(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--tag", default="m22")
    ap.add_argument("--arch", default="vpres")
    ap.add_argument("--latent", type=int, default=16)
    ap.add_argument("--n-per-subject", type=int, default=200)
    ap.add_argument("--pca-dim", type=int, default=32)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--model-seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--agg-n", nargs="*", type=int, default=[1, 10, 100])
    ap.add_argument("--agg-per-subject", type=int, default=2000)
    ap.add_argument("--skip-baselines", action="store_true",
                    help="기준선 재측정을 건너뛴다. ⚠️ 그러면 Phase 0(58명) 값과 "
                         "비교할 수밖에 없고 그건 모집단이 다르다")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        raise SystemExit("torch 가 필요합니다. 데스크탑(CUDA)에서 실행하십시오.")

    idx = np.load(os.path.join(args.data, "index.npz"))
    frames = np.load(os.path.join(args.data, f"frames_{args.tag}.npy"), mmap_mode="r")
    if len(idx["f_subject"]) != frames.shape[0]:
        raise SystemExit("index 와 memmap 길이가 다릅니다.")

    ok = valid_frame_mask(idx)
    pool = np.flatnonzero(ok)
    samp = splits.balanced_subject_sample(idx["f_subject"][pool].astype(int),
                                          args.n_per_subject, seed=0)
    sel = pool[samp["index"]]
    subj = idx["f_subject"][sel].astype(int)
    trel = idx["f_t_rel"][sel].astype(np.float64)
    ear = idx["f_ear"][sel][:, 2].astype(np.float32)
    ch = splits.chance_report(subj)
    print(f"표집 {len(sel):,}장  {ch}")
    print(f"  (유효 이벤트 프레임 {int(ok.sum()):,} 중에서. 좌석·결측 필터 통과분, U18 제외)")

    raw = np.asarray(frames[sel])
    px = C.batch_input(raw)[:, 0]                       # 정규화는 crop.INPUT_NORM 하나만
    flat = px.reshape(len(px), -1)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    out = {"env": repro.env_fingerprint(), "config": vars(args),
           "input_norm": C.INPUT_NORM, "sample": ch.as_dict(),
           "n_valid_frames_pool": int(ok.sum()),
           "phase0_58_reference": PHASE0_58,
           "phase0_comparability": (
               "비교 불가. Phase 0 은 58명·좌석 필터 이전·전체 프레임에서 쟀다. "
               "해석 기준선은 이 실행에서 재측정한 값을 쓴다."),
           "ours": {}, "baselines": {}, "aggregate": {}}

    # ---------------------------------------------------------- ours
    t0 = time.perf_counter()
    cells_acc: dict[str, list] = {}
    for f in args.folds:
        for ms in args.model_seeds:
            d = os.path.join(MODELS, f"fold{f}_seed{ms}", "encoder.pt")
            if not os.path.exists(d):
                continue
            Z = encode_all(px, torch.load(d, map_location=dev), args.arch, args.latent, dev)
            r = probes.run_representation(f"ours fold{f} seed{ms}", Z, subj, subj, trel,
                                          seeds=tuple(args.seeds), epochs=args.epochs)
            su = seen_unseen(Z, subj, trel, f, args.seeds, epochs=args.epochs)
            out["ours"][f"fold{f}_seed{ms}"] = {"cells": r["cells"], "seen_unseen": su}
            for c, v in r["cells"].items():
                cells_acc.setdefault(c, []).append(v["mean"])
            print(f"  fold{f} seed{ms}  TB-MLP {r['cells']['time_block/mlp']['mean']:.4f}  "
                  f"seen {su.get('seen',{}).get('mean',float('nan')):.4f} / "
                  f"unseen {su.get('unseen',{}).get('mean',float('nan')):.4f}", flush=True)
    out["ours_summary"] = {c: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)),
                               "n_models": len(v), "min": float(np.min(v)),
                               "max": float(np.max(v))}
                           for c, v in cells_acc.items()}
    su_all = {t: [m["seen_unseen"][t]["mean"] for m in out["ours"].values()
                  if t in m["seen_unseen"]] for t in ("seen", "unseen")}
    out["ours_seen_unseen"] = {t: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1))}
                               for t, v in su_all.items() if v}

    # ---------------------------------------------------------- 기준선 (같은 표집)
    if not args.skip_baselines:
        ph = np.array([[p["brightness"], p["contrast"], p["sharpness"]]
                       for p in (C.photometrics(im) for im in px)], np.float32)
        repro.seal(0)
        Zrand = encode_all(px, None, args.arch, args.latent, dev)
        reps = [("(a) 랜덤 인코더 D=16", Zrand, None),
                ("(b) 원본 픽셀", flat, None),
                (f"(c) 픽셀 PCA-{args.pca_dim}", flat, make_pca_transform(args.pca_dim)),
                ("(d') 광학 3차원", ph, None),
                ("(e) EAR 스칼라", ear[:, None], None)]
        for name, X, ft in reps:
            r = probes.run_representation(name, X, subj, subj, trel,
                                          seeds=tuple(args.seeds), fit_transform=ft,
                                          epochs=args.epochs)
            out["baselines"][name] = r
            print(f"  {name:<22} TB-MLP {r['cells']['time_block/mlp']['mean']:.4f}", flush=True)

    # ---------------------------------------------------------- 집계 공격
    samp2 = splits.balanced_subject_sample(idx["f_subject"][pool].astype(int),
                                           args.agg_per_subject, seed=0,
                                           min_required=args.agg_per_subject)
    sel2 = pool[samp2["index"]]
    subj2 = idx["f_subject"][sel2].astype(int)
    ch2 = splits.chance_report(subj2)
    print(f"\n집계 코호트: {ch2}  제외 {samp2['dropped_subjects']}")
    out["aggregate_cohort"] = {"chance": ch2.as_dict(),
                               "dropped_subjects": samp2["dropped_subjects"],
                               "n_per_subject": args.agg_per_subject}
    px2 = C.batch_input(np.asarray(frames[sel2]))[:, 0]
    best = max(out["ours"], key=lambda k: out["ours"][k]["cells"]["time_block/mlp"]["mean"])
    f_b, s_b = best.split("_")
    Zb = encode_all(px2, torch.load(os.path.join(
        MODELS, f"{f_b}_{s_b}", "encoder.pt"), map_location=dev),
        args.arch, args.latent, dev)
    out["aggregate_model"] = best
    for n in args.agg_n:
        Xa, ya = probes.aggregate(Zb, subj2, n, seed=0, mode="mean")
        accs = []
        for s in args.seeds:
            m = splits.random_masks(len(ya), seed=s)
            accs.append(probes.run_probe(Xa, ya, m, "mlp", seed=s,
                                         epochs=args.epochs)["accuracy"])
        out["aggregate"][f"N={n}"] = {
            "mean": float(np.mean(accs)), "std": float(np.std(accs)),
            "n_groups": int(len(ya)), "chance": splits.chance_report(ya).as_dict()}
        print(f"  N={n:<4} 정확도 {np.mean(accs):.4f}  묶음 {len(ya)}", flush=True)

    out["minutes"] = round((time.perf_counter() - t0) / 60, 1)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    os.replace(tmp, args.out)

    print(f"\n{'='*66}")
    print(f"{'표현':<24}{'TB-선형':>10}{'TB-MLP':>10}{'무작위-MLP':>12}")
    o = out["ours_summary"]
    print(f"{'ours (15모델 평균)':<24}{o['time_block/linear']['mean']:>10.4f}"
          f"{o['time_block/mlp']['mean']:>10.4f}{o['random/mlp']['mean']:>12.4f}")
    for name, r in out["baselines"].items():
        c = r["cells"]
        print(f"{name:<24}{c['time_block/linear']['mean']:>10.4f}"
              f"{c['time_block/mlp']['mean']:>10.4f}{c['random/mlp']['mean']:>12.4f}")
    print(f"\nchance: 균등 {ch.uniform_chance:.4f} / 다수 {ch.majority_baseline:.4f}")
    if "ours_seen_unseen" in out:
        su = out["ours_seen_unseen"]
        print(f"seen {su['seen']['mean']:.4f} vs unseen {su['unseen']['mean']:.4f}  "
              f"(차이 {su['seen']['mean']-su['unseen']['mean']:+.4f})")
    print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
