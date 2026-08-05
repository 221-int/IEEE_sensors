"""Phase −1 게이트 — 공용 모듈이 실제로 사고를 막는지 검사합니다.

    python -m src.v2.gate_minus1

통과 못 하면 종료 코드가 0이 아니고, 그 상태에서는 v2 의 어떤 ± 숫자도 보고하지
않습니다. 산출물: results/v2/gate_minus1.json

검사 항목 (전부 숫자 기준)
    G-1a  fold 배치 구성        fold별 2022 비중이 전체 ±10%p 이내 AND 2022 인원 >= 3
                               (2026-08-03 개정. 옛 기준 "이벤트 max/min <= 1.05" 는
                                이벤트 수 단일 층화 시절의 것이라 §4 의 배치 이중
                                층화 결정과 충돌했다. 근거는 §4 의 후보 비교표)
    G-1b  fold 결정성           얼린 배정을 생성기로 재현해 동일
    G-1c  격자 저-FA 도달       logit 격자가 FA <= 0.001 을 짚는다 (linear 는 못 짚음)
    G-1d  격자 설정 혼용 차단    서로 다른 설정이 섞이면 예외
    G-1e  chance 불균형 감지     majority > 4 x uniform 을 잡아낸다
    G-1f  균형 표집             표집 후 클래스 균형 회복
    G-1g  학습 결정성 (합성 MLP)  같은 시드 2회 -> 손실 비트 단위 동일
    G-1h  프로브 보조 로직       집계/투영 제거 수치 검사
    G-1i  학습 결정성 (실제 인코더) 같은 시드 2회 -> 손실·**가중치** 비트 동일

G-1g/G-1i 는 torch 가 있어야 돌아갑니다. 이 프로젝트에서 결정성 봉인이 안 된 상태로
보고된 ± 가 문제였으므로, **torch 환경(데스크탑 CUDA)에서 반드시 한 번 통과**시킨
뒤 그 JSON 을 커밋하십시오.

⚠️ **G-1g 만으로는 부족합니다.** G-1g 가 도는 것은 Linear 2층짜리 합성 MLP 이고,
비결정성이 실제로 나오는 지점(Conv2d 역전파의 cuDNN 알고리즘 선택, BatchNorm,
memmap 데이터 경로)을 전혀 지나가지 않습니다. `train_encoder` 의 ± 를 인용하려면
**G-1i** 가 PASS 여야 합니다. (2026-08-03 추가)
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
    """얼린 fold 를 **읽어서** 검사한다. 절대 다시 얼리지 않는다.

    🔴 2026-08-03 수정. 이전 버전은 `splits.freeze_folds()` 를 호출했다. 그것은
    검사가 아니라 **생성**이었고, PROTOCOL §4 가 얼려 둔 57명·배치 이중 층화 배정을
    구버전 기준(이벤트 수만, 58명, U18 포함)으로 덮어썼다. 실제로 사고가 났다.
    게이트가 검사 대상을 파괴하면 게이트가 아니다.
    """
    if not os.path.exists(splits.FOLDS_PATH):
        return {"status": "MISSING", "PASS_G1a": None, "PASS_G1b": None,
                "reason": f"{splits.FOLDS_PATH} 없음. phase3_apply_flags 를 먼저 실행."}
    with open(splits.FOLDS_PATH, encoding="utf-8") as f:
        payload = json.load(f)
    bal = payload["balance"]
    assign = {int(k): int(v) for k, v in payload["assign"].items()}
    counts = {int(k): int(v) for k, v in payload["counts"].items()}
    two_way = "n_2020" in bal                       # 배치 이중 층화 여부

    load = [sum(counts[u] for u in assign if assign[u] == i) for i in range(payload["n_folds"])]
    out = {
        "criterion": payload.get("criterion"),
        "excluded_users": payload.get("excluded_users"),
        "n_subjects": len(assign),
        "events_per_fold": load,
        "subjects_per_fold": [sum(1 for u in assign if assign[u] == i)
                              for i in range(payload["n_folds"])],
        "max_over_min": max(load) / max(min(load), 1),
        "recorded_balance_matches": load == bal["events_per_fold"],
        "two_way_stratified": two_way,
        "frozen_at": splits.FOLDS_PATH,
    }
    # G-1b 결정성: 생성기가 시드를 쓰지 않으므로 같은 입력이면 같은 배정이어야 한다.
    if two_way:
        # 얼린 파일에 기록된 counts·batch 로 생성기를 다시 돌려 **같은 배정이 나오는지**
        # 본다. 시드를 쓰지 않는 결정적 배정이므로 하나라도 다르면 FAIL 이다.
        batch = {int(k): v for k, v in payload["batch"].items()}
        again = splits.stratified_folds_2way(counts, batch, payload["n_folds"])
        out["deterministic"] = bool(again == assign)
        out["PASS_G1b"] = bool(again == assign)
        out["n_reassigned"] = int(sum(1 for u in assign if again.get(u) != assign[u]))
    else:
        again = splits.stratified_folds(counts)
        out["deterministic"] = bool(again == assign)
        out["PASS_G1b"] = bool(again == assign)

    # ----------------------------------------------------------------- G-1a
    # 2026-08-03 개정. 옛 기준 "전체 이벤트 max/min <= 1.05" 는 **이벤트 수만
    # 층화하던 시절**의 것이고, PROTOCOL §4 는 배치 구성을 잡기 위해 1.144 를
    # 의도적으로 수용했다. 두 문서가 충돌했다.
    #
    # 새 기준은 §4 가 **결과를 보기 전에** 세 후보를 비교하며 적어 둔 근거에서
    # 그대로 가져온다. 임계값을 새로 지어낸 것이 아니다.
    #
    #   (i)  fold 별 2022 이벤트 비중이 전체 비중 ±10%p 이내
    #        §4 채택안 26~36% (전체 28.4%) 는 통과, 기각안 12~62% 는 탈락.
    #        기준이 채택/기각을 실제로 갈라야 기준이다.
    #   (ii) fold 별 2022 인원 >= 3
    #        §4: "fold 하나의 2022 인원이 1명이면 배치별 분리 보고가 무의미해진다".
    #        채택안이 [3,4,4,4,4] 이므로 3 이 그 결정의 하한이다.
    #
    # 전체 이벤트 max/min 은 **기록만 하고 판정하지 않는다.** §4 의 세 후보가
    # 1.016/1.131/1.359 로 흩어졌는데 채택안이 1.144 였다 — 그 축은 결정을
    # 가르지 않았다. 가르지 않는 축에 게이트를 걸면 게이트가 거짓 안심을 준다.
    if two_way:
        n22 = bal.get("n_2022") or []
        share = [
            sum(counts[u] for u in assign
                if assign[u] == i and payload["batch"][str(u)] == "2022") / max(load[i], 1)
            for i in range(payload["n_folds"])
        ]
        overall = sum(counts[u] for u in assign
                      if payload["batch"][str(u)] == "2022") / max(sum(load), 1)
        dev = [abs(s - overall) for s in share]
        out["batch2022_share_per_fold"] = share
        out["batch2022_share_overall"] = overall
        out["batch2022_share_max_dev_pp"] = max(dev) * 100
        out["n_2020_per_fold"] = bal.get("n_2020")
        out["n_2022_per_fold"] = n22
        ok_share = max(dev) <= 0.10
        ok_count = bool(n22) and min(n22) >= 3
        out["PASS_G1a"] = bool(ok_share and ok_count)
        out["G1a_criterion"] = ("배치 구성: fold별 2022 이벤트 비중이 전체 ±10%p 이내 "
                                "AND fold별 2022 인원 >= 3 (PROTOCOL §4)")
        out["G1a_detail"] = {"share_within_10pp": bool(ok_share),
                             "min_n_2022_ge_3": bool(ok_count),
                             "event_max_over_min_recorded_only": out["max_over_min"]}
    else:
        out["PASS_G1a"] = bool(out["max_over_min"] <= 1.05)
        out["G1a_criterion"] = "전체 이벤트 max/min <= 1.05 (이벤트 수 단일 층화 기준)"
    return out


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


def check_determinism_encoder(n_events: int = 256, epochs: int = 2,
                              batch: int = 32, data_dir: str = "data/processed/v2") -> dict:
    """G-1i — **실제 v2 인코더**로 결정성을 확인한다.

    왜 G-1g 로 부족한가
    -------------------
    G-1g 가 도는 것은 `Linear(32,64) + ReLU + Dropout + Linear(64,2)` 짜리 합성
    MLP 다. 그것이 비트 동일해도 `train_encoder` 의 ± 는 보증되지 않는다. 실제
    학습에는 G-1g 가 전혀 건드리지 않는 것들이 들어 있다.

        Conv2d 역전파   cuDNN 알고리즘 선택이 비결정적일 수 있는 대표 지점
        BatchNorm2d     배치 통계 누적 순서
        Conv1d(TCN) 헤드 + 마스크 pooling
        memmap -> crop.batch_input   데이터 경로 자체

    비결정성은 보통 **conv 역전파**에서 나온다. 그래서 그것을 안 도는 게이트로
    "학습 결정성 PASS"라고 적으면 게이트가 이름값을 못 한다.

    손실뿐 아니라 **최종 가중치까지 비트 동일**한지 본다. 손실만 보면 마지막
    step 이후의 차이를 놓친다.
    """
    try:
        import torch
    except ImportError:
        return {"status": "SKIPPED", "reason": "torch 없음 (데스크탑에서 재실행 필요)",
                "PASS_G1i": None}
    if not os.path.exists(os.path.join(data_dir, "index.npz")):
        return {"status": "SKIPPED", "reason": f"{data_dir}/index.npz 없음",
                "PASS_G1i": None}

    from src.v2.model import encoder as E
    from src.v2.train_encoder import Bundle

    b = Bundle(data_dir)
    # 피험자 분리를 유지한 채 앞쪽에서 고정 개수만 쓴다(시드와 무관한 결정적 선택).
    assign = splits.load_folds()
    m = splits.subject_masks(b.subject, assign, splits.fold_rotation(0))
    ids = np.flatnonzero(m["train"])[:n_events]

    def one_run(seed: int):
        repro.seal(seed)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        front = E.build("vpres", 16).to(dev)
        head = E.build_head(16, 19).to(dev)
        opt = torch.optim.Adam(list(front.parameters()) + list(head.parameters()), lr=1e-3)
        lossf = torch.nn.BCEWithLogitsLoss()
        g = np.random.default_rng(seed)
        hist = []
        for _ in range(epochs):
            front.train(); head.train()
            perm = g.permutation(ids)
            tot = 0.0
            for i in range(0, len(perm), batch):
                sub = perm[i:i + batch]
                x, mk, y = b.batch(sub)
                bb, t = x.shape[:2]
                xb = torch.from_numpy(x.reshape(bb * t, *x.shape[2:])).to(dev)
                z = front(xb).reshape(bb, t, -1)
                logit = head(z, torch.from_numpy(mk).to(dev))
                loss = lossf(logit, torch.from_numpy(y).float().to(dev))
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss.detach())
            hist.append(tot)
        w = [p.detach().cpu().clone() for p in
             list(front.parameters()) + list(head.parameters())]
        return hist, w, dev

    a, wa, dev = one_run(0)
    bb_, wb, _ = one_run(0)
    c, wc, _ = one_run(1)
    loss_same = all(x.hex() == y.hex() for x, y in zip(a, bb_))
    weights_same = all(torch.equal(x, y) for x, y in zip(wa, wb))
    weights_differ = any(not torch.equal(x, y) for x, y in zip(wa, wc))
    return {
        "status": "RUN", "device": dev,
        "n_events": int(len(ids)), "epochs": epochs, "batch": batch,
        "arch": "vpres", "latent": 16,
        "run_a": a, "run_b": bb_, "run_seed1": c,
        "loss_bit_identical_same_seed": bool(loss_same),
        "weights_bit_identical_same_seed": bool(weights_same),
        "weights_differ_across_seeds": bool(weights_differ),
        "PASS_G1i": bool(loss_same and weights_same and weights_differ),
    }


# --------------------------------------------------------------- main
def main() -> int:
    repro.ensure_hashseed()
    repro.seal(0)

    print("Phase -1 게이트\n" + "=" * 60)
    res: dict = {"env": repro.env_fingerprint()}

    print("[G-1a/b] fold 층화 + 결정성  (얼린 파일을 읽기만 함)")
    res["folds"] = check_folds()
    fo = res["folds"]
    if fo.get("status") == "MISSING":
        print(f"  MISSING — {fo['reason']}")
    else:
        print(f"  기준 {fo['criterion']}")
        print(f"  피험자 {fo['n_subjects']}  제외 {fo['excluded_users']}  "
              f"이벤트/fold {fo['events_per_fold']}  max/min {fo['max_over_min']:.4f}")
        if fo["two_way_stratified"]:
            print(f"  2020 인원 {fo['n_2020_per_fold']}  2022 인원 {fo['n_2022_per_fold']}")
            print(f"  재생성 배정 일치 {fo['deterministic']} "
                  f"(불일치 {fo['n_reassigned']}명)")
            print(f"  2022 이벤트 비중 "
                  f"{[f'{s:.1%}' for s in fo['batch2022_share_per_fold']]}  "
                  f"전체 {fo['batch2022_share_overall']:.1%}  "
                  f"최대편차 {fo['batch2022_share_max_dev_pp']:.1f}%p")
            print(f"  G-1a 기준: {fo['G1a_criterion']}")
            print(f"  (전체 이벤트 max/min {fo['max_over_min']:.4f} — 기록만, 판정 안 함)")

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

    print("[G-1i] 학습 결정성 — 실제 v2 인코더 (torch + 데이터)")
    res["determinism_encoder"] = check_determinism_encoder()
    de = res["determinism_encoder"]
    if de["status"] == "SKIPPED":
        print(f"  SKIPPED — {de['reason']}")
    else:
        print(f"  손실 비트 동일 {de['loss_bit_identical_same_seed']}  "
              f"가중치 비트 동일 {de['weights_bit_identical_same_seed']}  "
              f"시드 바꾸면 달라짐 {de['weights_differ_across_seeds']}  "
              f"({de['device']}, 이벤트 {de['n_events']})")

    gates = {
        "G-1a fold 배치 구성 (§4)": res["folds"]["PASS_G1a"],
        "G-1b fold 결정성": res["folds"]["PASS_G1b"],
        "G-1c 격자 저-FA 도달": res["grid_tail"]["PASS_G1c"],
        "G-1d 격자 혼용 차단": res["grid_audit"]["PASS_G1d"],
        "G-1e chance 불균형 감지": res["splits"]["chance_detects_imbalance"],
        "G-1f 균형 표집 회복": res["splits"]["balanced_sample_ok"],
        "G-1  부호 규약 일치": res["selection"]["PASS_G1_selection"],
        "G-1h 투영 제거 동작": res["probes"]["projection_removes_class_means"],
        "G-1g 학습 결정성 (합성 MLP)": res["determinism"]["PASS_G1g"],
        "G-1i 학습 결정성 (실제 인코더)": res["determinism_encoder"]["PASS_G1i"],
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
    # 🔴 2026-08-03: torch 없는 환경에서 이 게이트를 돌렸다가 **데스크탑(CUDA)이 남긴
    # G-1g/G-1i PASS 기록을 덮어썼다.** 게이트가 증거를 지운 것이 이번이 두 번째다
    # (첫 번째는 check_folds 가 얼린 fold 를 덮어쓴 것).
    # -> 이전 결과가 torch 게이트를 실제로 돌렸고 이번 실행은 건너뛰었다면,
    #    정보가 줄어드는 덮어쓰기이므로 **거부**하고 옆 파일에 쓴다.
    def _ran_torch(d: dict) -> bool:
        return any(d.get(k, {}).get("status") == "RUN"
                   for k in ("determinism", "determinism_encoder"))

    dest = OUT
    if os.path.exists(OUT):
        try:
            with open(OUT, encoding="utf-8") as f:
                prev = json.load(f)
        except (OSError, json.JSONDecodeError):
            prev = {}
        if _ran_torch(prev) and not _ran_torch(res):
            dest = OUT.replace(".json", ".no_torch.json")
            print(f"\n  ⚠️  기존 {OUT} 은 torch 게이트를 실제로 돌린 기록입니다.")
            print(f"      이번 실행은 torch 없이 돌아 SKIP 이므로 덮어쓰지 않습니다.")
            print(f"      이번 결과는 {dest} 에 씁니다.")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"  -> {dest}")
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
