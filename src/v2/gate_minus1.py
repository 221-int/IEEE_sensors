"""Phase −1 게이트 — 공용 모듈이 실제로 사고를 막는지 검사합니다.

    python -m src.v2.gate_minus1

통과 못 하면 종료 코드가 0이 아니고, 그 상태에서는 v2 의 어떤 ± 숫자도 보고하지
않습니다. 산출물: results/v2/gate_minus1.json

검사 항목 (전부 숫자 기준)
    G-1a  fold 층화 균형        max/min <= 1.05
    G-1b  fold 결정성           두 번 계산해 동일
    G-1c  격자 저-FA 도달       logit 격자가 FA <= 0.001 을 짚는다 (linear 는 못 짚음)
    G-1d  격자 설정 혼용 차단    서로 다른 설정이 섞이면 예외
    G-1e  chance 불균형 감지     majority > 4 x uniform 을 잡아낸다
    G-1f  균형 표집             표집 후 클래스 균형 회복
    G-1g  학습 결정성 (torch)   같은 시드 2회 -> 손실 비트 단위 동일
    G-1h  프로브 보조 로직       집계/투영 제거 수치 검사

G-1g 는 torch 가 있어야 돌아갑니다. 이 프로젝트에서 결정성 봉인이 안 된 상태로
보고된 ± 가 문제였으므로, **torch 환경(데스크탑 CUDA)에서 반드시 한 번 통과**시킨
뒤 그 JSON 을 커밋하십시오.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

from src.v2.common import probes, repro, splits
from src.v2.common import thresholds as th

OUT = "results/v2/gate_minus1.json"


# --------------------------------------------------------------- 개별 검사
def check_folds() -> dict:
    payload = splits.freeze_folds()
    bal = payload["balance"]
    again = splits.stratified_folds(
        {int(k): int(v) for k, v in payload["counts"].items()})
    same = again == {int(k): int(v) for k, v in payload["assign"].items()}
    return {
        "events_per_fold": bal["events_per_fold"],
        "subjects_per_fold": bal["subjects_per_fold"],
        "max_over_min": bal["max_over_min"],
        "deterministic": bool(same),
        "PASS_G1a": bool(bal["max_over_min"] <= 1.05),
        "PASS_G1b": bool(same),
        "frozen_to": splits.FOLDS_PATH,
    }


def check_grid_tail() -> dict:
    """저-FA 구간을 격자가 실제로 짚는지.

    이 프로젝트에서 세 번 사고를 낸 지점입니다. 성긴 격자는 음성 분포의 꼬리를
    건너뛰어 '그 임계값이 존재하지 않는' 상태를 만듭니다.
    """
    rng = np.random.default_rng(0)
    neg = rng.beta(2, 8, size=10_000)          # 음성 10,000 (낮은 점수)
    pos = rng.beta(8, 2, size=10_000)          # 양성 10,000
    s = th.canonical(np.concatenate([neg, pos]), higher_fires=True)
    y = np.concatenate([np.zeros(10_000), np.ones(10_000)]).astype(int)

    out = {}
    variants = {
        # v2 채택: 근사하지 않는다 (유일값 전체)
        "v2_exhaustive": (th.GridConfig(), True),
        # 대조 1: 클래스별 logit 분위수 257  (근사 모드)
        "quantile_classaware": (th.GridConfig(max_exhaustive=0, n=257, spacing="logit"), True),
        # 대조 2: 라벨 없이 섞인 분포에 logit  (처음 틀렸던 설계)
        "quantile_pooled_logit": (th.GridConfig(max_exhaustive=0, n=257, spacing="logit"), False),
        # 대조 3: 성긴 선형 분위수 97점
        "quantile_linear97": (th.GridConfig(max_exhaustive=0, n=97, spacing="linear"), False),
    }
    # 기준선: 음성 10,000개면 FA 해상도는 1e-4 이고, FA<=1% 구간에서 **원리상 구별
    # 가능한 FA 값은 100개**입니다. 좋은 격자란 임의의 대안보다 N배 촘촘한 것이 아니라
    # 표본이 허용하는 해상도를 **다 쓰는** 것입니다. 그걸 기준으로 잡습니다.
    n_neg = int((y == 0).sum())
    possible = int(n_neg * 0.01)
    for tag, (cfg, class_aware) in variants.items():
        th.AUDIT.reset()
        g = th.threshold_grid(s, y=(y if class_aware else None), cfg=cfg, name=tag)
        fa = th.sweep(s, y, g)["fa"]
        band = fa[(fa > 0) & (fa <= 0.01)]
        realized = int(np.unique(band).size)
        out[tag] = {
            "grid_size": int(g.size),
            "min_positive_fa": float(fa[fa > 0].min()) if (fa > 0).any() else None,
            "n_grid_points_below_fa_1pct": int(band.size),
            "distinct_fa_levels_below_1pct": realized,
            "fa_resolution_coverage": float(realized / max(possible, 1)),
        }
    th.AUDIT.reset()
    out["n_negatives"] = n_neg
    out["distinct_fa_levels_possible_below_1pct"] = possible
    # v2 격자는 표본이 허용하는 FA 해상도를 100% 활용해야 한다(근사가 없으므로).
    out["PASS_G1c"] = bool(out["v2_exhaustive"]["fa_resolution_coverage"] >= 0.999)
    return out


def check_grid_audit() -> dict:
    """설정이 갈리면 반드시 터져야 합니다."""
    rng = np.random.default_rng(1)
    s = th.canonical(rng.normal(size=500), higher_fires=True)

    yb = (np.arange(500) % 2).astype(int)

    def audits(calls) -> bool:
        """calls 를 돌린 뒤 혼용 예외가 났는가."""
        th.AUDIT.reset()
        for fn in calls:
            fn()
        try:
            th.AUDIT.require_uniform()
            return False
        except RuntimeError:
            return True
        finally:
            th.AUDIT.reset()

    # 같은 설정 + 다른 데이터 -> 통과해야 함
    same_ok = not audits([
        lambda: th.threshold_grid(s, y=yb, name="ours"),
        lambda: th.threshold_grid(s * 3.0, y=yb, name="ear"),
    ])
    # 다른 설정 -> 막아야 함
    cfg_raises = audits([
        lambda: th.threshold_grid(s, y=yb, name="ours"),
        lambda: th.threshold_grid(s, y=yb, cfg=th.GridConfig(n=97, spacing="linear"),
                                  name="ear"),
    ])
    # 한쪽만 라벨을 준 경우 -> 막아야 함 (조용히 불공정해지는 경로)
    label_raises = audits([
        lambda: th.threshold_grid(s, y=yb, name="ours"),
        lambda: th.threshold_grid(s, y=None, name="ear"),
    ])

    # 이산 점수(K=4 코드) 대응
    th.AUDIT.reset()
    codes = np.repeat([0.0, 1.0, 2.0, 3.0], 25)
    g_disc = th.threshold_grid(th.canonical(codes, True), name="discrete")
    th.AUDIT.reset()

    return {
        "same_cfg_different_data_ok": same_ok,
        "different_cfg_raises": cfg_raises,
        "label_asymmetry_raises": label_raises,
        "discrete_grid_size": int(g_disc.size),
        "PASS_G1d": bool(same_ok and cfg_raises and label_raises),
    }


def check_selection_protocol() -> dict:
    """val 에서 고르고 test 에 얼려 적용하는 경로가 실제로 도는지."""
    rng = np.random.default_rng(2)
    n = 4000
    y_va = rng.integers(0, 2, n)
    y_te = rng.integers(0, 2, n)
    s_va = th.canonical(rng.normal(y_va * 1.2, 1.0), higher_fires=True)
    s_te = th.canonical(rng.normal(y_te * 1.2, 1.0), higher_fires=True)

    th.AUDIT.reset()
    pick = th.select_threshold(s_va, y_va, criterion=th.crit_accuracy(), name="ours")
    test = th.evaluate_at(s_te, y_te, pick["thr"])
    ap = th.average_precision(s_te, y_te)

    # EAR 처럼 '낮으면 발화' 인 점수도 같은 경로를 통과해야 합니다.
    ear_va = -s_va
    ear_te = -s_te
    pick_e = th.select_threshold(th.canonical(ear_va, higher_fires=False), y_va,
                                 criterion=th.crit_accuracy(), name="ear")
    test_e = th.evaluate_at(th.canonical(ear_te, higher_fires=False), y_te, pick_e["thr"])
    th.AUDIT.reset()

    return {
        "val_accuracy": pick["val_metrics"]["accuracy"],
        "test_accuracy": test["accuracy"],
        "optimism_val_minus_test": float(pick["val_metrics"]["accuracy"] - test["accuracy"]),
        "test_average_precision": ap,
        "sign_convention_matches": bool(
            abs(test["accuracy"] - test_e["accuracy"]) < 1e-12),
        "PASS_G1_selection": bool(abs(test["accuracy"] - test_e["accuracy"]) < 1e-12),
    }


def check_determinism(epochs: int = 3) -> dict:
    """같은 시드로 두 번 학습해 손실이 비트 단위로 같은지."""
    try:
        import torch
    except ImportError:
        return {"status": "SKIPPED", "reason": "torch 없음 (데스크탑에서 재실행 필요)",
                "PASS_G1g": None}

    def one_run(seed: int) -> list[float]:
        repro.seal(seed)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        g = torch.Generator().manual_seed(seed)
        X = torch.randn(2048, 32, generator=g)
        y = (X[:, 0] + 0.5 * X[:, 1] > 0).long()
        net = torch.nn.Sequential(
            torch.nn.Linear(32, 64), torch.nn.ReLU(True),
            torch.nn.Dropout(0.3), torch.nn.Linear(64, 2)).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        lossf = torch.nn.CrossEntropyLoss()
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, y), batch_size=128, shuffle=True,
            **repro.dataloader_kwargs(seed))
        hist = []
        for _ in range(epochs):
            net.train()
            tot = 0.0
            for xb, yb in dl:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad()
                loss = lossf(net(xb), yb)
                loss.backward()
                opt.step()
                tot += float(loss.detach())
            hist.append(tot)
        return hist

    a, b = one_run(0), one_run(0)
    c = one_run(1)
    bit_identical = all(x.hex() == y.hex() for x, y in zip(a, b))
    return {
        "status": "RUN",
        "run_a": a, "run_b": b, "run_seed1": c,
        "bit_identical_same_seed": bool(bit_identical),
        "differs_across_seeds": bool(a != c),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "PASS_G1g": bool(bit_identical and a != c),
    }


# --------------------------------------------------------------- main
def main() -> int:
    repro.ensure_hashseed()
    repro.seal(0)

    print("Phase -1 게이트\n" + "=" * 60)
    res: dict = {"env": repro.env_fingerprint()}

    print("[G-1a/b] fold 층화 + 결정성")
    res["folds"] = check_folds()
    print(f"  이벤트/fold {res['folds']['events_per_fold']}  "
          f"max/min {res['folds']['max_over_min']:.4f}  "
          f"결정적 {res['folds']['deterministic']}")

    print("[splits] 분할·chance 자기검사")
    res["splits"] = splits.selftest()

    print("[G-1c] 격자 저-FA 도달")
    res["grid_tail"] = check_grid_tail()
    print(f"  음성 {res['grid_tail']['n_negatives']}개 -> FA<=1% 에서 구별 가능한 "
          f"FA 값 {res['grid_tail']['distinct_fa_levels_possible_below_1pct']}개")
    for k in ("v2_exhaustive", "quantile_classaware",
              "quantile_pooled_logit", "quantile_linear97"):
        v = res["grid_tail"][k]
        print(f"  {k:22s} 격자 {v['grid_size']:6d}점  짚은 FA 값 "
              f"{v['distinct_fa_levels_below_1pct']:4d}  "
              f"해상도 활용률 {v['fa_resolution_coverage']:6.1%}")

    print("[G-1d] 격자 설정 혼용 차단")
    res["grid_audit"] = check_grid_audit()
    print(f"  같은 설정/다른 데이터 통과 {res['grid_audit']['same_cfg_different_data_ok']}  "
          f"다른 설정 차단 {res['grid_audit']['different_cfg_raises']}  "
          f"라벨 비대칭 차단 {res['grid_audit']['label_asymmetry_raises']}")

    print("[선택 프로토콜] val 선택 -> test 적용, 부호 규약")
    res["selection"] = check_selection_protocol()
    print(f"  val {res['selection']['val_accuracy']:.4f} -> test "
          f"{res['selection']['test_accuracy']:.4f}  "
          f"(낙관 {res['selection']['optimism_val_minus_test']:+.4f})  "
          f"부호규약 일치 {res['selection']['sign_convention_matches']}")

    print("[G-1h] 프로브 보조 로직")
    res["probes"] = probes.selftest_numpy_only()
    print(f"  집계 히스토그램 합=1 {res['probes']['aggregate_hist_rowsum_is_one']}  "
          f"투영 제거로 피험자 간 산포 "
          f"{res['probes']['between_subject_scatter_before']:.3f} -> "
          f"{res['probes']['between_subject_scatter_after']:.4f}")

    print("[G-1g] 학습 결정성 (torch)")
    res["determinism"] = check_determinism()
    if res["determinism"]["status"] == "SKIPPED":
        print(f"  SKIPPED — {res['determinism']['reason']}")
    else:
        print(f"  같은 시드 비트 동일 {res['determinism']['bit_identical_same_seed']}  "
              f"시드 바꾸면 달라짐 {res['determinism']['differs_across_seeds']}  "
              f"({res['determinism']['device']})")

    gates = {
        "G-1a fold 균형 <=1.05": res["folds"]["PASS_G1a"],
        "G-1b fold 결정성": res["folds"]["PASS_G1b"],
        "G-1c 격자 저-FA 도달": res["grid_tail"]["PASS_G1c"],
        "G-1d 격자 혼용 차단": res["grid_audit"]["PASS_G1d"],
        "G-1e chance 불균형 감지": res["splits"]["chance_detects_imbalance"],
        "G-1f 균형 표집 회복": res["splits"]["balanced_sample_ok"],
        "G-1  부호 규약 일치": res["selection"]["PASS_G1_selection"],
        "G-1h 투영 제거 동작": res["probes"]["projection_removes_class_means"],
        "G-1g 학습 결정성": res["determinism"]["PASS_G1g"],
    }
    res["gates"] = gates
    hard_fail = [k for k, v in gates.items() if v is False]
    skipped = [k for k, v in gates.items() if v is None]
    res["verdict"] = ("FAIL" if hard_fail else ("PARTIAL" if skipped else "PASS"))

    print("\n" + "=" * 60)
    for k, v in gates.items():
        print(f"  {'PASS' if v else ('SKIP' if v is None else 'FAIL')}  {k}")
    print(f"\n  판정: {res['verdict']}")
    if skipped:
        print("  ! torch 환경(데스크탑 CUDA)에서 다시 돌려 PASS 로 만든 뒤 커밋하십시오.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"  -> {OUT}")
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
