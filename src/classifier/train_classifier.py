"""
train_classifier.py

Branch: the blink JUDGMENT model. Freezes the trained autoencoder ENCODER,
turns each eye crop into a 128-d vector, and trains a small MLP:
    vector -> blink (0/1)

This is the learned replacement for the EAR threshold rule. It also directly
tests whether the vector separates blink vs non-blink (Phase-0 utility axis).

Split : SUBJECT-SEPARATED (same held-out clips as the autoencoder).
Target : --target blink_event (default) or eye_closed.
Imbalance handled with BCE pos_weight. Reports precision / recall / F1
(recall is our 1st-priority metric) at threshold 0.5 and at the best-F1 threshold.

Run:
    python -m src.classifier.train_classifier --data data/processed/eyeblink8_eyes.npz \
        --encoder models/autoencoder/encoder.pt --out models/classifier
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.encoder.train_autoencoder import Encoder


class MLP(nn.Module):
    def __init__(self, latent=128, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


@torch.no_grad()
def encode_all(encoder, images, device, batch=512):
    encoder.eval()
    zs = []
    for i in range(0, len(images), batch):
        xb = torch.from_numpy(images[i:i + batch]).float().div_(255.0).unsqueeze(1).to(device)
        zs.append(encoder(xb).cpu())
    return torch.cat(zs)


def event_recall_at_fa(prob, blink_id, subject, fa_budget, n_thr=97):
    """Event-level recall reachable at frame false-alarm <= fa_budget.

    This is the metric the project actually reports (an event counts as caught
    if ANY of its frames fires -- see src/eval/eval_events.py), so it is also
    what model selection should optimize. Selecting on frame-level F1 instead
    trades recall away to protect frame precision, which does not help an
    event-level target: missing 2 of an event's 6 frames costs nothing.
    Returns (best_recall, thr) or (0.0, None) if the budget is unreachable.
    """
    from collections import defaultdict
    events = defaultdict(list)
    for i in range(len(prob)):
        if blink_id[i] != -1:
            events[(subject[i], blink_id[i])].append(prob[i])
    if not events:
        return 0.0, None
    neg = prob[blink_id == -1]
    best, best_thr = 0.0, None
    for thr in np.linspace(0.02, 0.98, n_thr):
        far = float((neg >= thr).mean()) if len(neg) else 0.0
        if far > fa_budget:
            continue
        rec = sum(1 for ps in events.values() if max(ps) >= thr) / len(events)
        if rec > best:
            best, best_thr = rec, float(thr)
    return best, best_thr


def metrics(y_true, y_prob, thr):
    pred = (y_prob >= thr).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(thr=thr, precision=prec, recall=rec, f1=f1,
                tp=tp, fp=fp, fn=fn, tn=tn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/eyeblink8_eyes.npz")
    ap.add_argument("--encoder", default="models/autoencoder/encoder.pt")
    ap.add_argument("--config", default="models/autoencoder/config.json")
    ap.add_argument("--out", default="models/classifier")
    ap.add_argument("--target", choices=["blink_event", "eye_closed"], default="blink_event")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-subjects", nargs="*", default=["eb04", "eb11"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--select", choices=["frame_f1", "event_recall"],
                    default="frame_f1",
                    help="which epoch to keep. 'frame_f1' is the original "
                         "behaviour (kept as default so the shipped model stays "
                         "reproducible). 'event_recall' selects on the metric we "
                         "actually report -- event recall at a false-alarm budget.")
    ap.add_argument("--select-fa", type=float, default=0.03,
                    help="false-alarm budget for --select event_recall")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.out, exist_ok=True)
    latent = json.load(open(args.config))["latent"] if os.path.exists(args.config) else 128

    d = np.load(args.data, allow_pickle=True)
    images, subject = d["images"], d["subject"].astype(str)
    y = d[args.target].astype(np.float32)
    blink_id = d["blink_id"].astype(int)

    encoder = Encoder(latent).to(args.device)
    encoder.load_state_dict(torch.load(args.encoder, map_location=args.device))
    Z = encode_all(encoder, images, args.device)          # [N, latent]

    val_mask = np.isin(subject, args.val_subjects)
    Ztr, ytr = Z[~val_mask], torch.from_numpy(y[~val_mask])
    Zva, yva = Z[val_mask], y[val_mask]
    bid_va, subj_va = blink_id[val_mask], subject[val_mask]
    n_pos, n_neg = int(ytr.sum()), int((ytr == 0).sum())
    print(f"train {len(ytr)} (pos {n_pos}, {100*n_pos/len(ytr):.1f}%) | "
          f"val {len(yva)} | target={args.target} | device={args.device}")

    loader = DataLoader(TensorDataset(Ztr, ytr), batch_size=args.batch, shuffle=True)
    clf = MLP(latent).to(args.device)
    opt = torch.optim.Adam(clf.parameters(), lr=args.lr)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=args.device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    Zva_dev = Zva.to(args.device)
    best_score, best = -1.0, None
    best_event = (0.0, None)
    for epoch in range(1, args.epochs + 1):
        clf.train()
        for zb, yb in loader:
            zb, yb = zb.to(args.device), yb.to(args.device)
            opt.zero_grad()
            loss = loss_fn(clf(zb), yb)
            loss.backward()
            opt.step()

        clf.eval()
        with torch.no_grad():
            prob = torch.sigmoid(clf(Zva_dev)).cpu().numpy()
        # best-F1 threshold on val (always computed, for reporting)
        cand = [metrics(yva, prob, t) for t in np.linspace(0.05, 0.95, 19)]
        m_best = max(cand, key=lambda m: m["f1"])
        ev_rec, ev_thr = event_recall_at_fa(prob, bid_va, subj_va, args.select_fa)

        score = m_best["f1"] if args.select == "frame_f1" else ev_rec
        if score > best_score:
            best_score = score
            best = (m_best, metrics(yva, prob, 0.5))
            best_event = (ev_rec, ev_thr)
            torch.save(clf.state_dict(), os.path.join(args.out, "classifier.pt"))
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  frameF1 {m_best['f1']:.3f} "
                  f"(rec {m_best['recall']:.3f} prec {m_best['precision']:.3f} "
                  f"@thr {m_best['thr']:.2f})  |  event_recall@FA{args.select_fa:.0%} "
                  f"{ev_rec:.3f}")

    m_bestF1, m_half = best
    ev_rec, ev_thr = best_event
    print(f"\n=== kept model (selected by --select {args.select}, "
          f"subject-separated val) ===")
    for name, m in [("thr 0.50", m_half), (f"best-F1 thr {m_bestF1['thr']:.2f}", m_bestF1)]:
        print(f"  frame-level {name}: recall {m['recall']:.3f}  "
              f"precision {m['precision']:.3f}  F1 {m['f1']:.3f}  "
              f"(TP {m['tp']} FP {m['fp']} FN {m['fn']} TN {m['tn']})")
    print(f"  EVENT-level: recall {ev_rec:.3f} at frame false-alarm "
          f"<= {args.select_fa:.0%}" +
          (f" (thr {ev_thr:.2f})" if ev_thr is not None else "  [budget unreachable]"))
    json.dump({"target": args.target, "val_subjects": args.val_subjects,
               "select": args.select, "select_fa": args.select_fa,
               "seed": args.seed,
               "event_recall_at_fa": ev_rec, "event_thr": ev_thr,
               "best_f1": m_bestF1, "at_0.5": m_half},
              open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    print(f"\nSaved classifier + metrics to '{args.out}'.")
    print("NOTE: epoch AND threshold are both chosen on val here. With only 74 "
          "val events that is optimistic -- a held-out TEST split is needed "
          "before this number can be compared to EAR in the paper.")


if __name__ == "__main__":
    main()
