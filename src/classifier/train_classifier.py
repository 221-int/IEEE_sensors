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
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    latent = json.load(open(args.config))["latent"] if os.path.exists(args.config) else 128

    d = np.load(args.data, allow_pickle=True)
    images, subject = d["images"], d["subject"].astype(str)
    y = d[args.target].astype(np.float32)

    encoder = Encoder(latent).to(args.device)
    encoder.load_state_dict(torch.load(args.encoder, map_location=args.device))
    Z = encode_all(encoder, images, args.device)          # [N, latent]

    val_mask = np.isin(subject, args.val_subjects)
    Ztr, ytr = Z[~val_mask], torch.from_numpy(y[~val_mask])
    Zva, yva = Z[val_mask], y[val_mask]
    n_pos, n_neg = int(ytr.sum()), int((ytr == 0).sum())
    print(f"train {len(ytr)} (pos {n_pos}, {100*n_pos/len(ytr):.1f}%) | "
          f"val {len(yva)} | target={args.target} | device={args.device}")

    loader = DataLoader(TensorDataset(Ztr, ytr), batch_size=args.batch, shuffle=True)
    clf = MLP(latent).to(args.device)
    opt = torch.optim.Adam(clf.parameters(), lr=args.lr)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=args.device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    Zva_dev = Zva.to(args.device)
    best_f1, best = 0.0, None
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
        # best-F1 threshold on val
        cand = [metrics(yva, prob, t) for t in np.linspace(0.05, 0.95, 19)]
        m_best = max(cand, key=lambda m: m["f1"])
        if m_best["f1"] > best_f1:
            best_f1 = m_best["f1"]
            best = (m_best, metrics(yva, prob, 0.5))
            torch.save(clf.state_dict(), os.path.join(args.out, "classifier.pt"))
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  val bestF1 {m_best['f1']:.3f} "
                  f"(rec {m_best['recall']:.3f} prec {m_best['precision']:.3f} @thr {m_best['thr']:.2f})")

    m_bestF1, m_half = best
    print("\n=== best model (subject-separated val) ===")
    for name, m in [("thr 0.50", m_half), (f"best-F1 thr {m_bestF1['thr']:.2f}", m_bestF1)]:
        print(f"  {name}: recall {m['recall']:.3f}  precision {m['precision']:.3f}  "
              f"F1 {m['f1']:.3f}  (TP {m['tp']} FP {m['fp']} FN {m['fn']} TN {m['tn']})")
    json.dump({"target": args.target, "val_subjects": args.val_subjects,
               "best_f1": m_bestF1, "at_0.5": m_half},
              open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    print(f"\nSaved classifier + metrics to '{args.out}'. Recall is our 1st-priority metric.")


if __name__ == "__main__":
    main()
