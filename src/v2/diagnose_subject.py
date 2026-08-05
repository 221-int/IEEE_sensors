"""한 피험자에서 우리 인코더만 무너지는 이유를 찾는다. 기본 대상 U1.

    python -m src.v2.diagnose_subject                 # U1, torch 있으면 전부
    python -m src.v2.diagnose_subject --subject 16 --skip-torch

왜 필요한가
----------
U1 은 ours PR-AUC **0.8200** / ear_head **0.9802** 다(`results/v2/posthoc_subgroups.json`).
다음으로 나쁜 사람이 0.9361 이므로 완전한 이상치이고, **안경군 이득 결론을 혼자
끌고 간다**(제외 시 +0.0019 → +0.0120). 논문 서술 전에 원인이 필요하다.

배제된 가설
----------
    좌석 오염(다른 사람이 찍힘)   ear_head 가 **같은 얼굴 선택·같은 랜드마크**로
                                 0.9802 를 낸다. 다른 사람이었다면 EAR 도 무너진다.
    크롭 기하 (클리핑·업스케일)    U1 은 eye_clipped 0.0000 / lid_clipped 0.0000 /
                                 cubic 0.0000 / padded 0.0009 (crop_margin_check)
    결측                          최종 인덱스에서 U1 결측 0

남은 가설 — 이 스크립트가 재는 것
--------------------------------
    A  크롭에 신호가 있는가       이미지 대리값만으로 blink/unblink 를 가르나
    B  외형이 학습 분포 밖인가     밝기·눈영역 대비·암부 비율의 (어두움 × 안경) 교차
    C  전이 실패인가 본질적 난이도인가 (torch)
                                 U1 을 **학습에 본** fold 2·3·4 모델도 U1 을 못 맞히나

C 가 핵심이다. U1 은 fold 1 의 test 이므로 fold 1 모델은 U1 을 본 적이 없다. 그러나
fold 2·3·4 모델은 U1 을 train 에 넣고 학습했고 체크포인트가 전부 남아 있다.
학습에 봐도 못 맞히면 **픽셀에서 본질적으로 어려운 것**이고, 학습에 보면 맞히면
**전이 실패**다. 둘은 논문 서술이 완전히 다르다.

**역할 분담** — per-event 오차 감사(어떤 이벤트가 틀렸나, 플래그와의 연관)는
`src/v2/u1_audit.py` 가 이미 한다. 이 파일은 그것과 겹치지 않는 것만 잰다:
57명 대비 외형 위치, 크롭에 신호가 있는지, fold 교차 재생.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from src.v2.common import repro, splits
from src.v2.common import thresholds as TH

DATA = "data/processed/v2"
MODELS = "models/v2"
RESULT = "results/v2/train_encoder.json"
POSTHOC = "results/v2/posthoc_subgroups.json"
GLASSES = "data/raw/mEBAL2/glasses_labels_58.csv"
OUT = "results/v2/diagnose_subject.json"
FIGDIR = "docs/v2/figures"


# ------------------------------------------------------------------ A. 크롭에 신호가 있는가
def image_proxy_ap(data_dir: str, subjects: list[int], n_events: int = 400) -> dict:
    """학습 없이 크롭 픽셀만으로 blink/unblink 를 가른다.

    세 대리값을 쓴다. 전부 **프레임 자기 통계로 표준화한 뒤** 계산하므로 밝기·대비의
    영향을 받지 않는다(§8 의 입력 정규화와 같은 취지).

        depth     눈 영역 행 평균 프로파일의 최대−최소. 홍채 띠가 사라지면 줄어든다
        vgrad     세로 방향 경계 에너지
        darkfrac  어두운 화소 비율

    이벤트마다 EAR drop_ratio 와 **같은 모양**(가장자리 대비 최소값 하강률)으로
    바꿔 피험자별 PR-AUC 를 낸다. 이 값이 높으면 **크롭에 신호가 있다** — 즉
    실패 원인은 데이터가 아니라 모델이다.
    """
    idx = np.load(os.path.join(data_dir, "index.npz"))
    frames = np.load(os.path.join(data_dir, "frames_m22.npy"), mmap_mode="r")
    h, w = frames.shape[1], frames.shape[2]
    v = idx["e_valid"].astype(bool)
    rows, y, sub = idx["e_rows"][v], idx["e_is_blink"][v], idx["e_subject"][v]

    def proxies(im):
        x = im.astype(np.float32)
        x = (x - x.mean(axis=(1, 2), keepdims=True)) / (x.std(axis=(1, 2), keepdims=True) + 1e-6)
        band = x[:, h // 4: 3 * h // 4, :]
        rp = band.mean(axis=2)
        return np.stack([rp.max(1) - rp.min(1),
                         np.abs(np.diff(band, axis=1)).mean(axis=(1, 2)),
                         (band < -0.7).mean(axis=(1, 2))], 1)

    def drop(v3):
        edge = (v3[:, 0] + v3[:, -1]) / 2.0
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.nan_to_num((edge - v3.min(1)) /
                                 np.where(np.abs(edge) < 1e-6, 1e-6, edge))

    out = {}
    for u in subjects:
        k = np.flatnonzero(sub == u)
        if len(k) > n_events:
            k = k[np.linspace(0, len(k) - 1, n_events).astype(int)]
        r = rows[k]
        ok = r >= 0
        flat = np.where(ok, r, 0).ravel()
        o = np.argsort(flat)                       # memmap 은 정렬 접근이 훨씬 빠르다
        im = np.empty((len(flat), h, w), np.uint8)
        im[o] = frames[flat[o]]
        p = proxies(im).reshape(len(k), 19, 3)
        aps = []
        for j in range(3):
            col = p[..., j]
            col = np.where(ok, col, np.nan)
            col = np.nan_to_num(col, nan=float(np.nanmean(col)))
            aps.append(TH.average_precision(TH.canonical(drop(col), True), y[k]))
        out[int(u)] = {"n": int(len(k)), "img_proxy_ap": float(max(aps)),
                       "per_proxy_ap": [float(a) for a in aps]}
    return out


# ------------------------------------------------------------------ B. 외형 교차표
def appearance_cells(data_dir: str, subjects: list[int], ph: dict,
                     n_frames: int = 600) -> dict:
    idx = np.load(os.path.join(data_dir, "index.npz"))
    frames = np.load(os.path.join(data_dir, "frames_m22.npy"), mmap_mode="r")
    h, w = frames.shape[1], frames.shape[2]
    fs = idx["f_subject"]
    g = {int(r["user"]): int(r["glasses"])
         for r in csv.DictReader(open(GLASSES, encoding="utf-8"))}
    prof = {}
    for u in subjects:
        k = np.flatnonzero(fs == u)
        k = k[np.linspace(0, len(k) - 1, min(n_frames, len(k))).astype(int)]
        im = np.empty((len(k), h, w), np.uint8)
        o = np.argsort(k)
        im[o] = frames[k[o]]
        x = im.astype(np.float32)
        band = x[:, h // 4: 3 * h // 4, :]
        prof[int(u)] = {"brightness": float(x.mean()),
                        "eye_band_contrast": float(band.std(axis=(1, 2)).mean()),
                        "very_dark_frac": float((band < 40).mean()),
                        "glasses": g[u],
                        "ours": ph[str(u)]["ours_pr_auc"],
                        "ear_head": ph[str(u)]["ear_head_pr_auc"]}
    b = np.array([prof[u]["brightness"] for u in subjects])
    thr = float(np.percentile(b, 25))
    cells = {}
    for u in subjects:
        key = ("dark" if prof[u]["brightness"] <= thr else "normal",
               "glasses" if prof[u]["glasses"] else "no_glasses")
        cells.setdefault("·".join(key), []).append(u)
    summary = {}
    for c, us in cells.items():
        o = np.array([prof[u]["ours"] for u in us])
        summary[c] = {"n": len(us), "ours_mean": float(o.mean()),
                      "ours_min": float(o.min()), "worst": int(us[int(np.argmin(o))]),
                      "members": sorted(us)}
    return {"dark_threshold_p25": thr, "cells": summary, "per_subject": prof}


def save_strips(data_dir: str, subjects: list[int], out_png: str) -> str:
    """피험자별 blink / unblink 이벤트 19프레임을 가로로 이어 붙여 저장한다.

    **숫자만 보고 끝내지 않기 위한 것이다.** U1 의 원인은 표를 아무리 봐도 안 보였고,
    크롭을 눈으로 보고서야 드러났다(두꺼운 검은 뿔테 + 어두운 영상).
    대표 이벤트는 EAR 하강폭의 **중앙값**인 것을 고른다 — 가장 잘 보이는 것을
    고르면 그림이 주장을 만든다.
    """
    from PIL import Image, ImageDraw
    idx = np.load(os.path.join(data_dir, "index.npz"))
    frames = np.load(os.path.join(data_dir, "frames_m22.npy"), mmap_mode="r")
    h, w = frames.shape[1], frames.shape[2]
    v = idx["e_valid"].astype(bool)
    rows, y, sub = idx["e_rows"][v], idx["e_is_blink"][v], idx["e_subject"][v]
    ear = idx["f_ear"][:, 2].astype(np.float64)

    panels = []
    for u in subjects:
        for lab, name in ((1, "blink"), (0, "unblink")):
            k = np.flatnonzero((sub == u) & (y == lab))
            if not len(k):
                continue
            r = rows[k]
            e = np.where(r >= 0, ear[np.where(r >= 0, r, 0)], np.nan)
            with np.errstate(invalid="ignore"):
                d = (np.nanmax(e, 1) - np.nanmin(e, 1)) / np.nanmax(e, 1)
            pick = k[np.argsort(d)[len(k) // 2]]
            im = np.stack([frames[x] if x >= 0 else np.zeros((h, w), np.uint8)
                           for x in rows[pick]])
            panels.append((f"U{u} {name}", np.concatenate(list(im), axis=1)))

    pad = 22
    canvas = np.full(((h + pad) * len(panels), w * 19), 255, np.uint8)
    for i, (_, s) in enumerate(panels):
        canvas[i * (h + pad) + pad: i * (h + pad) + pad + h, :s.shape[1]] = s
    img = Image.fromarray(canvas)
    dr = ImageDraw.Draw(img)
    for i, (t, _) in enumerate(panels):
        dr.text((4, i * (h + pad) + 6), t, fill=0)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    img.save(out_png)
    return out_png


# ------------------------------------------------------------------ C. fold 교차 재생
def cross_fold_replay(data_dir: str, target: int, folds: list[int], seeds: list[int],
                      arch: str, latent: int) -> dict:
    """**U1 을 학습에 본 모델도 U1 을 못 맞히는가.**

    이것이 결정적 검사다. U1 은 fold 1 의 test 이므로 fold 1 모델은 U1 을 본 적이 없다.
    그러나 fold 2·3·4 모델은 U1 을 **train 에** 넣고 학습했고(fold 0 은 val),
    그 체크포인트가 전부 남아 있다.

        학습에 본 모델도 U1 에서 낮다   -> U1 의 깜빡임이 픽셀에서 **본질적으로 어렵다**
        학습에 본 모델은 잘 맞힌다      -> **전이 실패**다. 데이터가 아니라 일반화 문제

    ⚠️ train 에 있던 피험자를 평가하는 것이므로 이 숫자는 **성능이 아니다.**
    성능 보고에 쓰면 안 된다. 오직 위 두 갈래를 가르는 용도다.

    per-event 오차 감사는 `src/v2/u1_audit.py` 가 한다(이 파일과 역할이 다르다).
    """
    import torch
    from src.v2.model import encoder as E
    from src.v2.train_encoder import Bundle

    b = Bundle(data_dir)
    assign = splits.load_folds()
    home = assign[target]
    ids = np.flatnonzero(b.subject == target)
    y = b.y[ids]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def role(f: int) -> str:
        rot = splits.fold_rotation(f)
        for r in ("test", "val", "train"):
            if home in rot[r]:
                return r
        return "?"

    out = {"target": target, "home_fold": home, "n_events": int(len(ids)),
           "n_pos": int(y.sum()), "runs": [], "by_role": {}}
    for f in folds:
        for sd in seeds:
            d = os.path.join(MODELS, f"fold{f}_seed{sd}")
            if not os.path.exists(os.path.join(d, "encoder.pt")):
                continue
            front = E.build(arch, latent).to(dev)
            head = E.build_head(latent, 19).to(dev)
            front.load_state_dict(torch.load(os.path.join(d, "encoder.pt"), map_location=dev))
            head.load_state_dict(torch.load(os.path.join(d, "head.pt"), map_location=dev))
            front.eval(); head.eval()
            sc = []
            with torch.no_grad():
                for i in range(0, len(ids), 256):
                    x, mk, _ = b.batch(ids[i:i + 256])
                    nb, t = x.shape[:2]
                    z = front(torch.from_numpy(x.reshape(nb * t, *x.shape[2:])).to(dev))
                    sc.append(head(z.reshape(nb, t, -1),
                                   torch.from_numpy(mk).to(dev)).cpu().numpy())
            sc = np.concatenate(sc)
            out["runs"].append({
                "fold": f, "seed": sd, "role_of_target": role(f),
                "pr_auc": TH.average_precision(TH.canonical(sc, True), y),
                "roc_auc": TH.roc_auc(TH.canonical(sc, True), y),
                "score_mean_pos": float(sc[y == 1].mean()),
                "score_mean_neg": float(sc[y == 0].mean()),
            })
    for r in ("test", "val", "train"):
        v = [x["pr_auc"] for x in out["runs"] if x["role_of_target"] == r]
        if v:
            out["by_role"][r] = {"n_runs": len(v), "pr_auc_mean": float(np.mean(v)),
                                 "pr_auc_min": float(np.min(v)), "pr_auc_max": float(np.max(v))}
    return out


def cross_fold_all(data_dir: str, folds: list[int], seeds: list[int],
                   arch: str, latent: int, scores_out: str | None = None) -> dict:
    """**대조군이 없으면 C 의 숫자는 아무 말도 하지 않는다.**

    "U1 을 학습에 본 모델은 U1 에서 0.9998" 은 그 자체로는 정보가 없다. CNN 은
    train loss 0.02 까지 내려가므로 **아무 피험자나** 학습에 넣으면 1.0 에 가깝다.
    의미가 있는 것은 **격차(train − test)의 분포에서 U1 이 어디 있는가** 다.

    그래서 체크포인트 15개로 **전체 이벤트**를 한 번씩 채점하고, 피험자마다
        test  = 자기 fold 모델 3런 (본 적 없음. 정식 평가)
        train = 자기가 train 에 있던 9런 (성능 아님)
    를 갈라 격차를 낸다.

    부수 효과로 **이번 런에서 저장되지 않은 `ours` 점수를 복원**한다
    (`scores_out` 에 npz 로 남긴다). ear_head 가중치는 없으므로 ours 만이다.
    """
    import torch
    from src.v2.model import encoder as E
    from src.v2.train_encoder import Bundle

    b = Bundle(data_dir)
    assign = splits.load_folds()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n = len(b.y)
    ids = np.arange(n)
    role_of = {}
    for f in folds:
        rot = splits.fold_rotation(f)
        for u, home in assign.items():
            for r in ("test", "val", "train"):
                if home in rot[r]:
                    role_of[(f, u)] = r

    per_run = {}
    for f in folds:
        for sd in seeds:
            d = os.path.join(MODELS, f"fold{f}_seed{sd}")
            if not os.path.exists(os.path.join(d, "encoder.pt")):
                continue
            front = E.build(arch, latent).to(dev)
            head = E.build_head(latent, 19).to(dev)
            front.load_state_dict(torch.load(os.path.join(d, "encoder.pt"), map_location=dev))
            head.load_state_dict(torch.load(os.path.join(d, "head.pt"), map_location=dev))
            front.eval(); head.eval()
            sc = []
            with torch.no_grad():
                for i in range(0, n, 512):
                    x, mk, _ = b.batch(ids[i:i + 512])
                    nb, t = x.shape[:2]
                    z = front(torch.from_numpy(x.reshape(nb * t, *x.shape[2:])).to(dev))
                    sc.append(head(z.reshape(nb, t, -1),
                                   torch.from_numpy(mk).to(dev)).cpu().numpy())
            per_run[(f, sd)] = np.concatenate(sc)
            print(f"      fold{f} seed{sd} 채점 완료", flush=True)

    if scores_out:
        os.makedirs(os.path.dirname(scores_out) or ".", exist_ok=True)
        np.savez_compressed(
            scores_out, y=b.y, subject=b.subject,
            **{f"f{f}_s{sd}": v for (f, sd), v in per_run.items()})

    out = {"per_subject": {}, "n_runs": len(per_run)}
    for u in sorted(assign):
        k = b.subject == u
        if len(np.unique(b.y[k])) < 2:
            continue
        vals = {"test": [], "val": [], "train": []}
        for (f, sd), sc in per_run.items():
            vals[role_of[(f, u)]].append(
                TH.average_precision(TH.canonical(sc[k], True), b.y[k]))
        rec = {r: (float(np.mean(v)) if v else None) for r, v in vals.items()}
        rec["n"] = int(k.sum())
        rec["gap_train_minus_test"] = (
            rec["train"] - rec["test"] if rec["train"] and rec["test"] else None)
        out["per_subject"][int(u)] = rec
    g = np.array([v["gap_train_minus_test"] for v in out["per_subject"].values()
                  if v["gap_train_minus_test"] is not None])
    out["gap_distribution"] = {"median": float(np.median(g)), "p90": float(np.percentile(g, 90)),
                               "max": float(g.max()), "n": int(len(g))}
    return out


# ------------------------------------------------------------------ main
def main() -> int:
    repro.ensure_hashseed()
    repro.seal(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--result", default=RESULT)
    ap.add_argument("--posthoc", default=POSTHOC)
    ap.add_argument("--n-events", type=int, default=400,
                    help="[A] 피험자당 표집 이벤트 수. 줄이면 빨라지지만 AP 가 거칠어진다")
    ap.add_argument("--n-frames", type=int, default=600,
                    help="[B] 피험자당 표집 프레임 수")
    ap.add_argument("--strips", nargs="*", type=int, default=None,
                    help="크롭 스트립을 저장할 피험자 목록. 기본은 대상 + 같은 셀 비교군")
    ap.add_argument("--all-subjects", action="store_true",
                    help="[C] 57명 전원의 train-test 격차를 내 대조군을 만든다. "
                         "체크포인트 15개 × 전체 이벤트라 수 분 걸린다")
    ap.add_argument("--scores-out", default="results/v2/restored_ours_scores.npz",
                    help="복원한 ours 점수를 남길 곳 (--all-subjects 일 때)")
    ap.add_argument("--skip-torch", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    ph = json.load(open(args.posthoc, encoding="utf-8"))["per_subject"]
    subs = sorted(int(k) for k in ph)
    tgt = args.subject
    print(f"대상 U{tgt}: ours {ph[str(tgt)]['ours_pr_auc']:.4f} / "
          f"ear_head {ph[str(tgt)]['ear_head_pr_auc']:.4f}")

    print("\n[A] 크롭에 신호가 있는가 — 학습 없는 이미지 대리값")
    prox = image_proxy_ap(args.data, subs, args.n_events)
    a = np.array([prox[u]["img_proxy_ap"] for u in subs])
    print(f"    U{tgt} {prox[tgt]['img_proxy_ap']:.4f}  "
          f"(57명 중앙 {np.median(a):.4f}, 백분위 {(a < prox[tgt]['img_proxy_ap']).mean()*100:.0f}%)")
    verdict_a = ("크롭에 신호가 있다 -> 원인은 데이터가 아니라 모델"
                 if prox[tgt]["img_proxy_ap"] > np.median(a) else
                 "크롭에도 신호가 약하다 -> 데이터 문제일 수 있다")
    print(f"    => {verdict_a}")

    print("\n[B] 외형 (어두움 × 안경)")
    app = appearance_cells(args.data, subs, ph, args.n_frames)
    for c, s in sorted(app["cells"].items()):
        print(f"    {c:<18} n={s['n']:2d}  ours 평균 {s['ours_mean']:.4f}  "
              f"최소 {s['ours_min']:.4f} (U{s['worst']})")
    p = app["per_subject"][tgt]
    def pctl(key):
        arr = np.array([app["per_subject"][u][key] for u in subs])
        return (arr < p[key]).mean() * 100
    print(f"    U{tgt}: 밝기 {p['brightness']:.1f} ({pctl('brightness'):.0f}%)  "
          f"눈영역 대비 {p['eye_band_contrast']:.1f} ({pctl('eye_band_contrast'):.0f}%)  "
          f"암부 {p['very_dark_frac']:.3f} ({pctl('very_dark_frac'):.0f}%)")

    strips = args.strips if args.strips is not None else [tgt, 11, 51, 24]
    png = save_strips(args.data, [u for u in strips if str(u) in ph],
                      os.path.join(FIGDIR, f"u{tgt}_strips.png"))
    print(f"\n[그림] {png} — 표만 보지 말고 열어서 볼 것")

    res = {"subject": tgt, "env": repro.env_fingerprint(), "figure": png,
           "config": vars(args),
           "A_image_proxy": prox, "A_verdict": verdict_a, "B_appearance": app}

    if not args.skip_torch:
        try:
            cfg = json.load(open(args.result, encoding="utf-8"))["config"]
            print(f"\n[C] fold 교차 재생 — U{tgt} 를 학습에 본 모델도 못 맞히는가")
            cf = cross_fold_replay(args.data, tgt, cfg["folds"], cfg["seeds"],
                                   cfg["arch"], cfg["latent"])
            res["C_cross_fold"] = cf
            print(f"    U{tgt} 의 fold = {cf['home_fold']}  이벤트 {cf['n_events']} "
                  f"(양성 {cf['n_pos']})")
            for r in ("test", "val", "train"):
                if r in cf["by_role"]:
                    b_ = cf["by_role"][r]
                    tag = {"test": "본 적 없음 (정식 평가)", "val": "val 로 봄",
                           "train": "학습에 봄 — 성능 아님, 판별용"}[r]
                    print(f"    {r:<6} {b_['n_runs']:2d}런  PR-AUC 평균 {b_['pr_auc_mean']:.4f} "
                          f"[{b_['pr_auc_min']:.4f}, {b_['pr_auc_max']:.4f}]   {tag}")
            tr = cf["by_role"].get("train")
            te = cf["by_role"].get("test")
            if tr and te:
                gap = tr["pr_auc_mean"] - te["pr_auc_mean"]
                res["C_verdict_provisional"] = (
                    "전이 실패로 보임 (학습에 보면 맞힘)" if gap > 0.05
                    else "본질적 난이도로 보임 (학습에 봐도 못 맞힘)")
                print(f"    => 격차 {gap:+.4f}  →  {res['C_verdict_provisional']}")
                print("    ⚠️ 대조군 없음. 모든 피험자가 학습에 넣으면 ~1.0 이므로 이 격차만으로는"
                      " U1 고유 현상인지 알 수 없다. --all-subjects 로 분포를 볼 것")

            if args.all_subjects:
                print(f"\n[C2] 57명 전원 train−test 격차 (대조군)")
                ca = cross_fold_all(args.data, cfg["folds"], cfg["seeds"],
                                    cfg["arch"], cfg["latent"], args.scores_out)
                res["C2_cross_fold_all"] = ca
                gd = ca["gap_distribution"]
                gt = ca["per_subject"][tgt]["gap_train_minus_test"]
                allg = np.array([v["gap_train_minus_test"] for v in ca["per_subject"].values()
                                 if v["gap_train_minus_test"] is not None])
                pct = (allg < gt).mean() * 100
                print(f"    격차 중앙 {gd['median']:.4f}  p90 {gd['p90']:.4f}  최대 {gd['max']:.4f}")
                print(f"    U{tgt} 격차 {gt:.4f}  → 백분위 {pct:.0f}%")
                worst = sorted(ca["per_subject"].items(),
                               key=lambda kv: -(kv[1]["gap_train_minus_test"] or 0))[:5]
                print("    격차 상위 5: " + "  ".join(
                    f"U{u} {v['gap_train_minus_test']:.3f}(test {v['test']:.3f})"
                    for u, v in worst))
                res["C2_verdict"] = (
                    f"U{tgt} 의 격차가 57명 중 {pct:.0f} 백분위. "
                    + ("고유한 전이 실패" if pct >= 95 else
                       "고유하지 않다 — 전이 격차는 전반적 현상"))
                print(f"    => {res['C2_verdict']}")
        except ImportError as ex:
            print(f"\n[C] 건너뜀 — {ex}")
            res["C_cross_fold"] = {"skipped": str(ex)}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    os.replace(tmp, args.out)
    print(f"\n  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
