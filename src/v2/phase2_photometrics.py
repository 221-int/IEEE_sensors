"""밝기 정규화 결정 — 58명 광학 지표 분산과 (밝기, 선명도) 프로브.

    python -m src.v2.phase2_photometrics

무엇을 정하려는가
----------------
`to_input_tensor` 가 /255 만 할지, 사용자별 밝기 정규화를 넣을지. 이 결정의 근거는
"보기 좋은가"가 아니라 **지름길을 막는가** 입니다.

사람마다 녹화 밝기가 다르면 재식별 프로브가 얼굴이 아니라 **밝기를 외워서** 사람을
맞힐 수 있습니다. 그러면 우리가 재는 것이 "표현이 신원을 담는가"가 아니라
"조명이 사람마다 다른가"가 됩니다. 깜빡임 분류기도 같은 지름길을 탈 수 있습니다.

그래서 두 가지를 잽니다.
  1. **F비** = 사용자 간 분산 / 사용자 내 분산. 크면 광학값이 곧 사용자 표식이다
  2. **(밝기, 선명도) 2차원 프로브** — 이 두 숫자만으로 58명을 몇 % 맞히는가.
     이게 높으면 재식별 결과의 상당 부분이 신원이 아니라 세션의 광학 서명이다

정규화 후보
----------
프레임 단위 표준화 `(x - mean) / std` 는 **인과적**이라 Pi 에서 실시간으로 가능합니다
(녹화 전체 통계를 쓰는 방식은 미래를 보는 것이라 배포 불가).
이 변환은 밝기를 상수로 만들고 선명도를 `sharpness / contrast^2` 로 바꿉니다.
그래서 정규화 후의 잔여 누출은 그 값의 F비로 미리 볼 수 있습니다.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from src.v2.common import repro, splits

SHARDS = "data/processed/v2/shards/u*.npz"
OUT = "results/v2/photometrics_58.json"


def f_ratio(values: list[np.ndarray]) -> dict:
    """사용자별 배열 리스트 -> 사용자 간/내 분산비.

    within 은 사용자 내 분산의 (표본수 가중) 평균, between 은 사용자 평균의 분산.
    F 가 1 이면 사용자 구분이 없다는 뜻이고, 클수록 값 자체가 사용자 표식이다.
    """
    m = np.array([v.mean() for v in values])
    n = np.array([len(v) for v in values], float)
    within = float(np.average([v.var() for v in values], weights=n))
    between = float(m.var(ddof=1))
    return {"between_std": float(np.sqrt(between)), "within_std": float(np.sqrt(within)),
            "f_ratio": float(between / max(within, 1e-12)),
            "user_mean_min": float(m.min()), "user_mean_max": float(m.max())}


def main() -> int:
    repro.ensure_hashseed()
    repro.seal(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin-tag", default="m22", choices=["m22", "m18"])
    ap.add_argument("--n-per-subject", type=int, default=200,
                    help="프로브용 피험자당 표집 프레임 수 (균형 표집)")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    fs = sorted(glob.glob(SHARDS))
    if not fs:
        raise SystemExit(f"샤드가 없습니다: {SHARDS}")
    t = args.margin_tag

    users, B, S, C_, T, U = [], [], [], [], [], []
    for f in fs:
        d = np.load(f)
        u = int(d["user"])
        b = d[f"f_brightness_{t}"].astype(np.float64)
        s = d[f"f_sharpness_{t}"].astype(np.float64)
        c = d[f"f_contrast_{t}"].astype(np.float64)
        users.append(u); B.append(b); S.append(s); C_.append(c)
        T.append(d["f_t_rel"].astype(np.float64))
        U.append(np.full(len(b), u))

    # 프레임 단위 표준화 후 남는 값: 밝기는 상수가 되고 선명도는 sharpness/contrast^2
    Sn = [s / np.maximum(c, 1e-6) ** 2 for s, c in zip(S, C_)]

    out: dict = {"env": repro.env_fingerprint(), "margin_tag": t,
                 "n_users": len(users), "n_frames": int(sum(len(b) for b in B)),
                 "f_ratio": {}}
    print(f"샤드 {len(users)}명  프레임 {out['n_frames']:,}  배율 {t}")
    print()
    print("=== F비 (사용자 간 분산 / 사용자 내 분산) ===")
    for name, vals in (("brightness", B), ("sharpness", S), ("contrast", C_),
                       ("sharpness_after_norm", Sn)):
        r = f_ratio(vals)
        out["f_ratio"][name] = r
        print(f"  {name:22s} F {r['f_ratio']:7.2f}   사용자간 std {r['between_std']:8.2f}"
              f"   사용자내 std {r['within_std']:8.2f}   범위 {r['user_mean_min']:.1f}~{r['user_mean_max']:.1f}")

    # ---- (밝기, 선명도) 2차원 프로브 ----
    X = np.stack([np.concatenate(B), np.concatenate(S)], axis=1)
    subj = np.concatenate(U)
    trel = np.concatenate(T)
    samp = splits.balanced_subject_sample(subj, args.n_per_subject, seed=0)
    idx = samp["index"]
    print()
    print(f"=== (밝기, 선명도) 2차원 프로브 ===")
    print(f"  표집 {len(idx):,}개  {splits.ChanceReport(**samp['chance'])}")
    out["probe_sample"] = samp["chance"]

    try:
        from src.v2.common import probes
    except ImportError:
        probes = None
    try:
        import torch  # noqa: F401
        has_torch = True
    except ImportError:
        has_torch = False

    if not has_torch:
        print("  ! torch 가 없어 프로브를 건너뜁니다. 데스크탑에서 다시 실행하십시오.")
        out["probe"] = {"status": "SKIPPED", "reason": "torch 없음"}
    else:
        res = probes.run_representation(
            "photometrics_2d", X[idx], subj[idx], subj[idx], trel[idx],
            seeds=tuple(args.seeds), epochs=60)
        out["probe"] = res
        print(f"  {'조건':<22}{'평균':>9}{'std':>8}")
        for k, v in res["cells"].items():
            print(f"  {k:<22}{v['mean']:>9.4f}{v['std']:>8.4f}")
        print(f"  무작위 − 시간블록 = {res['random_minus_timeblock']:+.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    os.replace(tmp, args.out)
    print(f"\n  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
