"""대조군 8 — 세션 시각(t_rel) 교락. PROTOCOL §9 표 8행, §10 S2.

    python -m src.v2.control8_trel            # torch 있으면 전부, 없으면 A2/E 생략

무엇을 방어하는가
----------------
우리 인코더는 크롭 **이미지**를 본다. 이미지에는 눈꺼풀만이 아니라 **세션이 진행되며
변하는 것**(조명, 자세 피로, 카메라 노출)이 함께 들어 있다. 라벨(blink/unblink)이
세션 시각과 상관돼 있으면, 모델은 눈꺼풀 대신 "언제 찍혔는가"를 배워도 점수가 오른다.
EAR 은 랜드마크 기하만 보므로 이 지름길을 쓸 수 없다. **그러면 우리 우위의 일부는
표현의 기여가 아니다.**

실측된 위험 (PROTOCOL §10 S2): 피험자별 클래스 시간 위치 편차
Δ = mean(t_rel | blink) − mean(t_rel | unblink) 가 U46 에서 −0.495 다.

다섯 가지를 잰다
---------------
    A1  t_rel 단조 점수의 PR-AUC          (numpy)  — 시간이 라벨을 얼마나 가르나
    A2  t_rel MLP 프로브                  (torch)  — 비단조 관계까지 포함
    B   피험자별 Δ 와 |Δ|>0.2 명단         (numpy)
    C   Δ 와 **우리 이득**의 상관          (numpy)  — 가장 직접적인 검사
    D   S2 민감도: |Δ|>0.2 제외 후 이득     (numpy)
    E   ear_rule + t_rel vs ear_rule      (torch)  — 시간을 더하면 이득이 생기나

C 가 핵심이다. t_rel 이 라벨 정보를 조금 갖고 있어도, **우리 이득이 Δ 와 무관하면**
그 정보를 우리가 쓰고 있다는 증거가 아니다. 반대로 Δ 가 큰 피험자에서 이득이
크면 그것이 지름길의 흔적이다.

⚠️ 분할은 유용성 축이므로 **피험자 분리 5-fold**(PROTOCOL §3)를 쓴다. 시간 블록
분할이 아니다 — 여기서 묻는 것은 "배포 조건에서 시간이 라벨을 가르는가" 다.

격자·분할·프로브·부트스트랩은 `src/v2/common/` 만 쓴다.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from src.v2.common import probes, repro, splits, stats
from src.v2.common import thresholds as TH

DATA = "data/processed/v2"
POSTHOC = "results/v2/posthoc_subgroups.json"
OUT = "results/v2/control8_trel.json"
DEV_GATE = 0.2                    # PROTOCOL §10 S2


def load_events(data_dir: str = DATA) -> dict:
    idx = np.load(os.path.join(data_dir, "index.npz"))
    v = idx["e_valid"].astype(bool)
    return {"t": idx["e_t_rel"][v].astype(np.float64),
            "y": idx["e_is_blink"][v].astype(np.int64),
            "s": idx["e_subject"][v].astype(int),
            "ear": None, "idx": idx, "valid": v}


def ear_rule_feature(idx, valid) -> np.ndarray:
    """EAR drop_ratio — `train_encoder.Bundle.ear_feature` 와 **같은 정의**.

    비교 기준선을 여기서 새로 정의하면 두 스크립트가 갈라진다. 정의가 바뀌면
    두 곳을 같이 고쳐야 한다는 뜻이므로, 바뀔 때는 train_encoder 쪽을 기준으로 삼는다.
    """
    rows = idx["e_rows"][valid].astype(np.int64)
    e = idx["f_ear"][:, 2].astype(np.float64)
    ok = rows >= 0
    v = np.where(ok, e[np.where(ok, rows, 0)], np.nan)
    any_ok = ok.any(axis=1)
    first_i = np.argmax(ok, axis=1)
    last_i = v.shape[1] - 1 - np.argmax(ok[:, ::-1], axis=1)
    first = np.take_along_axis(v, first_i[:, None], 1)[:, 0]
    last = np.take_along_axis(v, last_i[:, None], 1)[:, 0]
    edge = (first + last) / 2.0
    with np.errstate(invalid="ignore"):
        lo = np.where(any_ok, np.nanmin(np.where(ok, v, np.inf), axis=1), np.nan)
        out = (edge - lo) / edge
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ------------------------------------------------------------------ A. 시간의 라벨 정보량
def a1_monotone(ev: dict) -> dict:
    """t_rel 을 그대로 점수로 써서 fold 별 PR-AUC.

    부호를 모르므로 **양방향 다 재고 큰 쪽을 보고**한다. 한 방향만 재면
    "시간은 라벨을 못 가른다"는 결론이 부호 선택 때문에 나올 수 있다.
    """
    assign = splits.load_folds()
    rows = []
    for f in range(5):
        m = splits.subject_masks(ev["s"], assign, splits.fold_rotation(f))
        te = np.flatnonzero(m["test"])
        y = ev["y"][te]
        up = TH.average_precision(TH.canonical(ev["t"][te], True), y)
        dn = TH.average_precision(TH.canonical(ev["t"][te], False), y)
        rows.append({"fold": f, "n_test": len(te),
                     "pr_auc_higher": up, "pr_auc_lower": dn,
                     "pr_auc_best": max(up, dn),
                     "roc_auc_higher": TH.roc_auc(TH.canonical(ev["t"][te], True), y),
                     "prevalence": float(y.mean()),
                     "chance": splits.chance_report(y).as_dict()})
    return {"per_fold": rows,
            "pr_auc_best_mean": float(np.mean([r["pr_auc_best"] for r in rows])),
            "prevalence_mean": float(np.mean([r["prevalence"] for r in rows])),
            "note": "PR-AUC 의 chance 는 유병률(≈0.50)이다. 그 근처면 시간은 라벨을 못 가른다."}


def a2_mlp(ev: dict, seeds: list[int], feats: str = "t") -> dict:
    """t_rel(또는 t_rel+EAR) MLP 프로브. torch 필요.

    비단조 관계(예: 깜빡임이 세션 중반에 몰림)는 A1 이 못 잡는다. 선형만 보고
    넘어가면 안 된다는 것이 PROTOCOL §8 의 교훈이다(광학 선형 0.077 / MLP 0.202).
    """
    assign = splits.load_folds()
    X = ev["t"][:, None] if feats == "t" else np.c_[ev["ear"], ev["t"]]
    if feats == "ear":
        X = ev["ear"][:, None]
    runs = []
    for f in range(5):
        m = splits.subject_masks(ev["s"], assign, splits.fold_rotation(f))
        masks = {"train": m["train"], "test": m["test"]}
        for kind in ("linear", "mlp"):
            for sd in seeds:
                r = probes.run_probe(X, ev["y"], masks, kind, seed=sd, return_scores=True)
                r["pr_auc"] = TH.average_precision(
                    TH.canonical(r.pop("scores"), True), r.pop("y_test"))
                r.pop("test_rows", None)
                r["fold"] = f
                runs.append(r)
    out = {"feats": feats, "runs": runs}
    for kind in ("linear", "mlp"):
        v = [r["pr_auc"] for r in runs if r["kind"] == kind]
        a = [r["accuracy"] for r in runs if r["kind"] == kind]
        out[kind] = {"pr_auc_mean": float(np.mean(v)), "pr_auc_std": float(np.std(v, ddof=1)),
                     "accuracy_mean": float(np.mean(a)),
                     "majority_baseline": runs[0]["chance"]["majority_baseline"]}
    return out


# ------------------------------------------------------------------ B/C/D. Δ 와 이득
def b_deviation(ev: dict) -> dict:
    """피험자별 클래스 시간 위치 편차 Δ."""
    out = {}
    for u in np.unique(ev["s"]):
        k = ev["s"] == u
        yy = ev["y"][k]
        if len(np.unique(yy)) < 2:
            continue
        tt = ev["t"][k]
        out[int(u)] = {"delta": float(tt[yy == 1].mean() - tt[yy == 0].mean()),
                       "n": int(k.sum())}
    d = np.array([v["delta"] for v in out.values()])
    return {"per_subject": out,
            "abs_median": float(np.median(np.abs(d))),
            "abs_max": float(np.abs(d).max()),
            "n_over_gate": int((np.abs(d) > DEV_GATE).sum()),
            "gate": DEV_GATE,
            "over_gate": sorted([{"user": u, "delta": v["delta"]}
                                 for u, v in out.items() if abs(v["delta"]) > DEV_GATE],
                                key=lambda z: -abs(z["delta"]))}


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def c_gain_vs_delta(dev: dict, posthoc_path: str, n_boot: int) -> dict:
    """**핵심 검사** — 시간 분리가 큰 피험자에서 우리 이득이 큰가.

    상관이 0 근처면, t_rel 이 라벨 정보를 조금 갖고 있어도 우리가 그것을 쓰고 있다는
    증거가 아니다. 양의 상관이면 지름길의 흔적이다.
    """
    if not os.path.exists(posthoc_path):
        return {"available": False, "reason": f"{posthoc_path} 없음. posthoc_subgroups 먼저 실행"}
    ph = json.load(open(posthoc_path, encoding="utf-8"))["per_subject"]
    us = [u for u in dev["per_subject"] if str(u) in ph]
    ad = np.array([abs(dev["per_subject"][u]["delta"]) for u in us])
    gain = np.array([ph[str(u)]["ours_pr_auc"] - ph[str(u)]["ear_head_pr_auc"] for u in us])
    ours = np.array([ph[str(u)]["ours_pr_auc"] for u in us])
    bs = stats.subject_bootstrap(lambda r: _spearman(ad[r], gain[r]),
                                 np.arange(len(us)), n_boot=n_boot, seed=0)
    hi = ad > np.median(ad)
    return {"available": True, "n_subjects": len(us),
            "spearman_absdelta_vs_gain": _spearman(ad, gain),
            "spearman_ci": [bs["ci_lo"], bs["ci_hi"]],
            "pearson_absdelta_vs_gain": float(np.corrcoef(ad, gain)[0, 1]),
            "spearman_absdelta_vs_ours": _spearman(ad, ours),
            "gain_high_absdelta": float(gain[hi].mean()),
            "gain_low_absdelta": float(gain[~hi].mean()),
            "absdelta_median": float(np.median(ad))}


def d_s2_sensitivity(dev: dict, posthoc_path: str, n_boot: int) -> dict:
    """S2 — |Δ|>0.2 인 피험자를 빼고 이득을 다시 낸다."""
    if not os.path.exists(posthoc_path):
        return {"available": False}
    ph = json.load(open(posthoc_path, encoding="utf-8"))["per_subject"]
    drop = {x["user"] for x in dev["over_gate"]}
    out = {"available": True, "dropped": sorted(drop)}
    for tag, keep in (("all", lambda u: True),
                      ("s2_excluded", lambda u: u not in drop)):
        us = [u for u in dev["per_subject"] if str(u) in ph and keep(u)]
        g = np.array([ph[str(u)]["ours_pr_auc"] - ph[str(u)]["ear_head_pr_auc"] for u in us])
        bs = stats.subject_bootstrap(lambda r: float(g[r].mean()),
                                     np.arange(len(us)), n_boot=n_boot, seed=0)
        out[tag] = {"n_subjects": len(us), "gain_mean": float(g.mean()),
                    "gain_ci": [bs["ci_lo"], bs["ci_hi"]]}
    out["shift"] = out["s2_excluded"]["gain_mean"] - out["all"]["gain_mean"]
    return out


# ------------------------------------------------------------------ main
def main() -> int:
    repro.ensure_hashseed()
    repro.seal(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--posthoc", default=POSTHOC)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--skip-torch", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    ev = load_events(args.data)
    ev["ear"] = ear_rule_feature(ev["idx"], ev["valid"])
    ev.pop("idx"); ev.pop("valid")
    print(f"이벤트 {len(ev['y']):,}  피험자 {len(np.unique(ev['s']))}  "
          f"{splits.chance_report(ev['y'])}")

    a1 = a1_monotone(ev)
    print(f"\nA1  t_rel 단조 PR-AUC (fold별 최선 방향)")
    for r in a1["per_fold"]:
        print(f"    fold {r['fold']}  높으면발화 {r['pr_auc_higher']:.4f}  "
              f"낮으면발화 {r['pr_auc_lower']:.4f}  유병률 {r['prevalence']:.4f}")
    print(f"    평균 최선 {a1['pr_auc_best_mean']:.4f}  vs 유병률 {a1['prevalence_mean']:.4f}")

    dev = b_deviation(ev)
    print(f"\nB   피험자별 Δ: |Δ| 중앙 {dev['abs_median']:.4f}  최대 {dev['abs_max']:.4f}  "
          f"|Δ|>{DEV_GATE} 인원 {dev['n_over_gate']}")
    for x in dev["over_gate"]:
        print(f"    U{x['user']}  Δ={x['delta']:+.4f}")

    c = c_gain_vs_delta(dev, args.posthoc, args.n_boot)
    if c["available"]:
        print(f"\nC   |Δ| vs 우리 이득  (n={c['n_subjects']})")
        print(f"    Spearman {c['spearman_absdelta_vs_gain']:+.3f} "
              f"CI [{c['spearman_ci'][0]:+.3f}, {c['spearman_ci'][1]:+.3f}]")
        print(f"    |Δ| 상위 절반 이득 {c['gain_high_absdelta']:+.4f}  /  "
              f"하위 절반 {c['gain_low_absdelta']:+.4f}")
        print(f"    Spearman(|Δ|, ours 성능) {c['spearman_absdelta_vs_ours']:+.3f}")

    d = d_s2_sensitivity(dev, args.posthoc, args.n_boot)
    if d["available"]:
        print(f"\nD   S2 민감도 — U{d['dropped']} 제외")
        for tag in ("all", "s2_excluded"):
            x = d[tag]
            print(f"    {tag:<12} n={x['n_subjects']:2d}  이득 {x['gain_mean']:+.4f} "
                  f"CI [{x['gain_ci'][0]:+.4f}, {x['gain_ci'][1]:+.4f}]")
        print(f"    이동 {d['shift']:+.4f}")

    a2 = e = None
    if not args.skip_torch:
        try:
            a2 = a2_mlp(ev, args.seeds, "t")
            print(f"\nA2  t_rel 학습 프로브 (5fold × {len(args.seeds)}시드)")
            for k in ("linear", "mlp"):
                print(f"    {k:<7} PR-AUC {a2[k]['pr_auc_mean']:.4f} ± {a2[k]['pr_auc_std']:.4f}"
                      f"   정확도 {a2[k]['accuracy_mean']:.4f} "
                      f"(다수 기준선 {a2[k]['majority_baseline']:.4f})")
            e_only = a2_mlp(ev, args.seeds, "ear")
            e_both = a2_mlp(ev, args.seeds, "ear_t")
            e = {"ear_only": e_only, "ear_plus_t": e_both}
            print(f"\nE   t_rel 을 EAR 에 더하면")
            for k in ("linear", "mlp"):
                print(f"    {k:<7} ear {e_only[k]['pr_auc_mean']:.4f} -> "
                      f"ear+t {e_both[k]['pr_auc_mean']:.4f}  "
                      f"({e_both[k]['pr_auc_mean']-e_only[k]['pr_auc_mean']:+.4f})")
        except ImportError as ex:
            print(f"\nA2/E  건너뜀 — {ex}")
            a2 = {"skipped": str(ex)}

    out = {"env": repro.env_fingerprint(), "config": vars(args),
           "A1_monotone": a1, "A2_learned": a2, "B_deviation": dev,
           "C_gain_vs_delta": c, "D_s2_sensitivity": d, "E_ear_plus_t": e}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    os.replace(tmp, args.out)
    print(f"\n  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
