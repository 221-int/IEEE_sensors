"""
split_eval.py

Roadmap (4): a protocol that can actually be reported.

THE PROBLEM THIS FIXES
----------------------
Until now the epoch AND the decision threshold were both chosen on the same two
validation subjects that the numbers were then reported on. That is training on
the test set. With only 74 val events, taking the best of 40 epochs x 97
thresholds inflates the result by roughly the amount we were trying to measure,
so neither "we beat EAR" nor "EAR beats us" was defensible.

THE PROTOCOL
------------
Subject-wise 4-fold rotation over the 8 Eyeblink8 clips. Per fold:

    train (4 subjects)  ->  autoencoder + classifier weights
    val   (2 subjects)  ->  pick the epoch AND pick the threshold
    test  (2 subjects)  ->  report. Never looked at before this point.

Every subject serves as test exactly once and as val exactly once, so the
aggregate covers ALL 408 events instead of one arbitrary pair of clips.

The threshold is chosen on val as "the smallest threshold whose val false-alarm
is within budget", then FROZEN and applied to test. Test false-alarm is therefore
NOT pinned to the budget -- it is an outcome, and reporting it is the point.

EAR gets the identical treatment (threshold picked on the same val subjects,
applied to the same test subjects) so the comparison is protocol-matched.
EAR values come from the per-frame dump so MediaPipe runs only once:

    python -m src.eval.ear_baseline --val-subjects eb01 eb02 eb03 eb04 eb08 eb09 eb10 eb11 \
        --dump results/ear_frames.npz

Run:
    python -m src.experiments.split_eval --seeds 0 1 2 --ear-dump results/ear_frames.npz
    python -m src.experiments.split_eval --seeds 0 --epochs-ae 2 --epochs-clf 2   # smoke
"""

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.encoder.train_autoencoder import Autoencoder, Encoder

# Fixed rotation. Order is alphabetical so the folds are reproducible and not
# cherry-picked; each pair is test once and val once.
SUBJECTS = ["eb01", "eb02", "eb03", "eb04", "eb08", "eb09", "eb10", "eb11"]
PAIRS = [SUBJECTS[i:i + 2] for i in range(0, 8, 2)]   # 4 pairs
FOLDS = [{"test": PAIRS[i], "val": PAIRS[(i + 1) % 4],
          "train": [s for j, p in enumerate(PAIRS) if j not in (i, (i + 1) % 4)
                    for s in p]}
         for i in range(4)]


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def to_tensor(images):
    return torch.from_numpy(images).float().div_(255.0).unsqueeze(1)


@torch.no_grad()
def encode_all(encoder, images, device, batch=512):
    encoder.eval()
    return torch.cat([encoder(to_tensor(images[i:i + batch]).to(device)).cpu()
                      for i in range(0, len(images), batch)])


class MLP(nn.Module):
    def __init__(self, latent, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(True),
                                 nn.Dropout(0.3), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


# -------------------------------------------------------------- calibration
# 왜 필요한가: 4-fold 결과에서 val 에서 고른 확률 임계값이 test 피험자로 옮겨가지
# 못했다(test FA 표준편차 0.103, 최악 38%). EAR 은 눈 종횡비라는 물리량이라 사람이
# 달라도 스케일이 비슷하지만, 신경망 확률 출력은 신원별로 오프셋이 다르다.
#
# 깜빡임은 "확률이 0.8을 넘는 상태"가 아니라 "평소보다 갑자기 튀는 순간"이다.
# 아래는 전부 그 사람 자신의 스트림만 쓰고 **라벨을 쓰지 않는다**. 따라서 test
# 피험자에게 적용해도 정보 누출이 아니다(transductive/online adaptation).
CALIBRATIONS = ["none", "rank", "zscore", "rolling"]


def _causal_rolling(p, W):
    """과거만 보는 이동 median / MAD. 배포 가능한 인과적 형태."""
    n = len(p)
    med = np.empty(n, np.float64)
    mad = np.empty(n, np.float64)
    head = min(W, n)
    for t in range(head):                      # 초반은 확장 윈도우
        w = p[:t + 1]
        med[t] = np.median(w)
        mad[t] = np.median(np.abs(w - med[t]))
    if n > W:
        from numpy.lib.stride_tricks import sliding_window_view
        sw = sliding_window_view(p, W)         # window i = p[i:i+W] -> t = i+W-1
        m = np.median(sw, axis=1)
        med[W - 1:] = m
        mad[W - 1:] = np.median(np.abs(sw - m[:, None]), axis=1)
    return med, mad


def calibrate(prob, subject, frame_id, mode, window=150):
    """확률 스트림 -> 보정된 점수. 항상 '높을수록 깜빡임'.

    none    : 원본 확률 (현재 방식)
    rank    : 피험자 내 분위 순위 [0,1]  — 오프셋·스케일 모두 제거, 비인과적
    zscore  : (p - median) / IQR, 피험자 전체 기준 — 비인과적
    rolling : (p - 이동median) / 이동MAD, 과거만 사용 — **인과적, 배포 가능**
    """
    if mode == "none":
        return prob.astype(np.float64)
    out = np.empty(len(prob), np.float64)
    for u in np.unique(subject):
        idx = np.where(subject == u)[0]
        order = idx[np.argsort(frame_id[idx])]   # 시간순 정렬 필수
        p = prob[order].astype(np.float64)
        if mode == "rank":
            r = np.empty(len(p))
            r[np.argsort(p, kind="stable")] = np.arange(len(p))
            s = r / max(len(p) - 1, 1)
        elif mode == "zscore":
            med = np.median(p)
            q1, q3 = np.percentile(p, [25, 75])
            s = (p - med) / (q3 - q1 + 1e-6)
        elif mode == "rolling":
            med, mad = _causal_rolling(p, min(window, len(p)))
            s = (p - med) / (mad + 1e-6)
        else:
            raise ValueError(mode)
        out[order] = s
    return out


def threshold_grid(score, n=97):
    """보정 방식마다 점수 범위가 다르므로 분위수로 격자를 만든다."""
    qs = np.linspace(0.002, 0.998, n)
    g = np.unique(np.quantile(score[np.isfinite(score)], qs))
    return g


# ------------------------------------------------------------------ metrics
def group_events(score, blink_id, subject):
    """-> (list of per-event score lists, array of non-event scores).

    An event is caught if ANY of its frames fires, so the per-event summary is
    max() for a probability and min() for an EAR (lower = more closed).
    """
    ev = defaultdict(list)
    for i in range(len(score)):
        if blink_id[i] != -1:
            ev[(subject[i], blink_id[i])].append(score[i])
    neg = score[blink_id == -1]
    return list(ev.values()), neg


def curve(score, blink_id, subject, thresholds, higher_fires=True):
    """-> list of (thr, event_recall, n_hit, n_events, frame_false_alarm)."""
    events, neg = group_events(score, blink_id, subject)
    n_ev = len(events)
    out = []
    for thr in thresholds:
        if higher_fires:
            hit = sum(1 for e in events if np.nanmax(e) >= thr)
            far = float(np.nanmean(neg >= thr)) if len(neg) else 0.0
        else:
            hit = sum(1 for e in events if np.nanmin(e) <= thr)
            far = float(np.nanmean(neg <= thr)) if len(neg) else 0.0
        out.append((float(thr), hit / n_ev if n_ev else 0.0, hit, n_ev, far))
    return out


def pick_threshold(rows, fa_budget):
    """Choose a threshold on VAL. Returns (thr, val_recall, val_fa, budget_met).

    Within budget: the highest event recall (ties -> first, i.e. the more
    permissive threshold, which generalizes slightly better than a razor edge).

    If the budget is UNREACHABLE (an undertrained model can be above it at every
    threshold) we do not fail -- we take the lowest-false-alarm operating point
    and set budget_met=False. Returning None here would drop the fold silently,
    which would bias the aggregate toward the folds that happened to train well.
    """
    ok = [r for r in rows if r[4] <= fa_budget]
    if ok:
        best = max(r[1] for r in ok)
        for r in ok:
            if r[1] == best:
                return r[0], r[1], r[4], True
    lo = min(r[4] for r in rows)
    r = max((z for z in rows if z[4] == lo), key=lambda z: z[1])
    return r[0], r[1], r[4], False


def apply_threshold(score, blink_id, subject, thr, higher_fires=True):
    events, neg = group_events(score, blink_id, subject)
    n_ev = len(events)
    if higher_fires:
        hit = sum(1 for e in events if np.nanmax(e) >= thr)
        far = float(np.nanmean(neg >= thr)) if len(neg) else 0.0
    else:
        hit = sum(1 for e in events if np.nanmin(e) <= thr)
        far = float(np.nanmean(neg <= thr)) if len(neg) else 0.0
    return hit / n_ev if n_ev else 0.0, hit, n_ev, far


# ------------------------------------------------------------------ training
def train_fold(images, y, m_tr, m_va, latent, device, ep_ae, ep_clf, batch, lr,
               fa_budget, blink_id, subject, log):
    """Train on train-mask, select epoch on val-mask, return (probs_all, thr_info)."""
    # --- autoencoder (select on val reconstruction)
    loader = DataLoader(TensorDataset(to_tensor(images[m_tr])), batch_size=batch,
                        shuffle=True, drop_last=True)
    val_t = to_tensor(images[m_va])
    ae = Autoencoder(latent).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    mse = nn.MSELoss()
    best, best_state = float("inf"), None
    for ep in range(1, ep_ae + 1):
        ae.train()
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad(); mse(ae(xb), xb).backward(); opt.step()
        ae.eval()
        with torch.no_grad():
            v = sum(mse(ae(val_t[i:i + batch].to(device)),
                        val_t[i:i + batch].to(device)).item() * len(val_t[i:i + batch])
                    for i in range(0, len(val_t), batch)) / len(val_t)
        if v < best:
            best = v
            best_state = {k: t.detach().cpu().clone()
                          for k, t in ae.encoder.state_dict().items()}
    enc = Encoder(latent).to(device)
    enc.load_state_dict(best_state); enc.eval()

    Z = encode_all(enc, images, device)

    # --- classifier (select epoch on val event recall within budget)
    ytr = torch.from_numpy(y[m_tr])
    n_pos, n_neg = int(ytr.sum()), int((ytr == 0).sum())
    clf = MLP(latent).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([n_neg / max(n_pos, 1)], device=device))
    dl = DataLoader(TensorDataset(Z[m_tr], ytr), batch_size=batch, shuffle=True)
    grid = np.linspace(0.02, 0.98, 97)

    # Selection key = (budget met?, val recall). An epoch that meets the budget
    # always beats one that does not, and within each group higher recall wins.
    best_key, best_clf, best_pick = None, None, None
    for ep in range(1, ep_clf + 1):
        clf.train()
        for zb, yb in dl:
            zb, yb = zb.to(device), yb.to(device)
            opt.zero_grad(); lossf(clf(zb), yb).backward(); opt.step()
        clf.eval()
        with torch.no_grad():
            pv = torch.sigmoid(clf(Z[m_va].to(device))).cpu().numpy()
        pick = pick_threshold(curve(pv, blink_id[m_va], subject[m_va], grid), fa_budget)
        key = (1 if pick[3] else 0, pick[1])
        if best_key is None or key > best_key:
            best_key, best_pick = key, pick
            best_clf = {k: t.detach().cpu().clone() for k, t in clf.state_dict().items()}
        if log and (ep % 20 == 0 or ep == 1):
            print(f"      CLF ep {ep:3d}  val_recall {pick[1]:.3f} @FA {pick[2]:.4f}"
                  f"{'' if pick[3] else '  [예산 초과]'}", flush=True)

    clf.load_state_dict(best_clf); clf.to(device).eval()
    with torch.no_grad():
        prob = torch.sigmoid(clf(Z.to(device))).cpu().numpy()
    return prob, best_pick, best


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/eyeblink8_eyes.npz")
    ap.add_argument("--ear-dump", default="results/ear_frames.npz",
                    help="per-frame EAR from ear_baseline.py --dump (optional)")
    ap.add_argument("--ear-variant", default="mean", help="which EAR variant to compare")
    ap.add_argument("--latent", type=int, default=128)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--fa-budget", type=float, default=0.03)
    ap.add_argument("--epochs-ae", type=int, default=30)
    ap.add_argument("--epochs-clf", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--json", default="results/split_eval.json")
    ap.add_argument("--calibrations", nargs="*", default=CALIBRATIONS,
                    help="점수 보정 방식. 같은 학습 모델에 후처리만 다르게 적용하므로 "
                         "여러 개를 줘도 추가 학습 비용이 없다.")
    ap.add_argument("--roll-window", type=int, default=150,
                    help="rolling 보정의 과거 윈도우 프레임 수 (15fps 기준 150 = 10초)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    images = d["images"]
    subject = d["subject"].astype(str)
    blink_id = d["blink_id"].astype(int)
    frame_id = d["frame_id"].astype(int)
    y = d["blink_event"].astype(np.float32)

    ear = None
    if args.ear_dump and os.path.exists(args.ear_dump):
        e = np.load(args.ear_dump, allow_pickle=True)
        # EAR 은 '낮을수록 감김'이므로 부호를 뒤집어 '높을수록 발화'로 통일한다.
        # 그래야 보정 함수와 임계값 로직을 우리 모델과 완전히 동일하게 쓸 수 있다.
        # recall / false-alarm 값 자체는 부호 반전에 불변이다.
        ear = {"subject": e["subject"].astype(str),
               "blink_id": e["blink_id"].astype(int),
               "score": -e[args.ear_variant].astype(np.float64),
               "frame_id": (e["frame_id"].astype(int) if "frame_id" in e
                            else np.arange(len(e["blink_id"])))}
        print(f"EAR dump loaded: {len(ear['score'])} frames, variant='{args.ear_variant}' "
              f"(부호 반전 -> 높을수록 발화)")
    else:
        print(f"(EAR dump not found at {args.ear_dump} -- skipping the EAR column. "
              f"Generate it with: python -m src.eval.ear_baseline "
              f"--val-subjects {' '.join(SUBJECTS)} --dump {args.ear_dump})")

    print(f"data {images.shape} | budget FA <= {args.fa_budget:.0%} | "
          f"latent {args.latent} | seeds {args.seeds} | device {args.device}")
    print("\n4-fold subject rotation (each subject is test once, val once):")
    for i, f in enumerate(FOLDS):
        n_ev = len({(s, b) for s, b in zip(subject, blink_id)
                    if b != -1 and s in f["test"]})
        print(f"  fold {i}: test={f['test']} ({n_ev} ev)  val={f['val']}  train={f['train']}")
    print(flush=True)

    results = []
    t0 = time.perf_counter()

    def eval_calibrated(score_all, bid, subj, fid, m_va, m_te, mode):
        """보정 -> val 에서 임계값 선택 -> 얼려서 test 에 적용."""
        s = calibrate(score_all, subj, fid, mode, args.roll_window)
        g = threshold_grid(s[m_va])
        pk = pick_threshold(curve(s[m_va], bid[m_va], subj[m_va], g), args.fa_budget)
        rec, hit, n_ev, far = apply_threshold(s[m_te], bid[m_te], subj[m_te], pk[0])
        return {"thr": float(pk[0]), "val_recall": pk[1], "val_fa": pk[2],
                "val_budget_met": pk[3], "test_recall": rec, "test_hit": hit,
                "test_events": n_ev, "test_fa": far}

    for i, f in enumerate(FOLDS):
        m_tr = np.isin(subject, f["train"])
        m_va = np.isin(subject, f["val"])
        m_te = np.isin(subject, f["test"])

        # ---- EAR: 동일 프로토콜 + 동일 보정 방식들
        ear_rows = None
        if ear is not None:
            em_va = np.isin(ear["subject"], f["val"])
            em_te = np.isin(ear["subject"], f["test"])
            ear_rows = {c: eval_calibrated(ear["score"], ear["blink_id"],
                                           ear["subject"], ear["frame_id"],
                                           em_va, em_te, c)
                        for c in args.calibrations}

        for seed in args.seeds:
            ts = time.perf_counter()
            print(f"=== fold {i} seed {seed} ===", flush=True)
            set_seed(seed)
            # 학습은 한 번. 보정은 순수 후처리라 모든 방식이 '동일한 모델'을 공유한다.
            # (에폭 선택은 보정 없는 확률 기준으로 고정 -> 보정 방식에 유리하게
            #  모델을 고르지 않으므로 비교가 보수적이다)
            prob, pick, ae_mse = train_fold(
                images, y, m_tr, m_va, args.latent, args.device,
                args.epochs_ae, args.epochs_clf, args.batch, args.lr,
                args.fa_budget, blink_id, subject, not args.quiet)

            cal = {c: eval_calibrated(prob, blink_id, subject, frame_id,
                                      m_va, m_te, c)
                   for c in args.calibrations}

            rec_dict = {
                "fold": i, "seed": seed, "test": f["test"], "val": f["val"],
                "recon_val_mse": ae_mse, "cal": cal, "ear": ear_rows,
                "seconds": round(time.perf_counter() - ts, 1),
            }
            results.append(rec_dict)
            line = "    " + "  ".join(
                f"{c}:{cal[c]['test_recall']:.3f}@{cal[c]['test_fa']:.3f}"
                for c in args.calibrations)
            print(line + f"   {rec_dict['seconds']:.0f}s\n", flush=True)

    if not results:
        raise SystemExit("no fold produced a result")

    # ------------------------------------------------------------- report
    n_runs = len(results)
    print("=" * 88)
    print("  SUBJECT-WISE 4-FOLD  (임계값은 val 에서 선택, test 에서만 보고)")
    print(f"  {n_runs} runs, {time.perf_counter()-t0:.0f}s | "
          f"보정 방식별 비교 (같은 학습 모델, 후처리만 다름)")
    print("=" * 88)

    def agg(rows_getter):
        r = np.array([rows_getter(x)["test_recall"] for x in results])
        f = np.array([rows_getter(x)["test_fa"] for x in results])
        over = sum(1 for x in results if rows_getter(x)["test_fa"] > args.fa_budget)
        return r, f, over

    print(f"  {'방식':>9} | {'OURS recall':>18} {'OURS FA':>18} {'초과':>5} | "
          f"{'EAR recall':>16} {'EAR FA':>16}")
    print("  " + "-" * 86)
    summary = {}
    for c in args.calibrations:
        r, f, over = agg(lambda x, c=c: x["cal"][c])
        row = {"ours": {"recall_mean": float(r.mean()), "recall_std": float(r.std()),
                        "fa_mean": float(f.mean()), "fa_std": float(f.std()),
                        "fa_over_budget": over, "fa_max": float(f.max())}}
        es = " " * 34
        if results[0]["ear"]:
            er, ef, eo = agg(lambda x, c=c: x["ear"][c])
            row["ear"] = {"recall_mean": float(er.mean()), "recall_std": float(er.std()),
                          "fa_mean": float(ef.mean()), "fa_std": float(ef.std()),
                          "fa_over_budget": eo, "fa_max": float(ef.max())}
            es = f"{er.mean():>8.3f}±{er.std():<7.3f}{ef.mean():>8.4f}±{ef.std():<7.4f}"
        summary[c] = row
        print(f"  {c:>9} | {r.mean():>9.3f}±{r.std():<8.3f}{f.mean():>9.4f}±{f.std():<8.4f}"
              f"{over:>4}/{n_runs} | {es}")

    base = summary[args.calibrations[0]]["ours"]
    print("\n  ── 보정 효과 (기준: %s) ──" % args.calibrations[0])
    for c in args.calibrations[1:]:
        o = summary[c]["ours"]
        dr = o["recall_mean"] - base["recall_mean"]
        ratio = base["fa_std"] / (o["fa_std"] + 1e-12)
        print(f"   {c:>9}: recall {dr:+.3f} | FA 표준편차 {base['fa_std']:.4f} -> "
              f"{o['fa_std']:.4f} ({ratio:.1f}배 {'감소' if ratio > 1 else '증가'}) | "
              f"최악 FA {base['fa_max']:.3f} -> {o['fa_max']:.3f}")

    print("\n  fold별 상세 (recall, 시드 평균)")
    hdr = f"  {'fold':>4} {'test':>12} " + "".join(f"{c:>10}" for c in args.calibrations)
    print(hdr)
    for i in sorted({r["fold"] for r in results}):
        rs = [r for r in results if r["fold"] == i]
        line = f"  {i:>4} {'+'.join(rs[0]['test']):>12} "
        for c in args.calibrations:
            line += f"{np.mean([x['cal'][c]['test_recall'] for x in rs]):>10.3f}"
        print(line)

    print("\n  주의: test_FA 는 예산에 고정되지 않는다. val 에서 고른 임계값을 그대로")
    print("        적용한 결과이므로, 예산을 넘는다면 그 자체가 일반화 실패의 증거다.")
    print("  rank/zscore 는 피험자 스트림 전체를 쓰는 비인과적 보정이고,")
    print("  rolling 은 과거만 쓰는 인과적 보정이라 그대로 배포 가능하다.")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        json.dump({"config": vars(args), "folds": FOLDS,
                   "summary": summary, "results": results},
                  open(args.json, "w"), indent=2, default=str)
        print(f"\n  -> {args.json}")


if __name__ == "__main__":
    main()
