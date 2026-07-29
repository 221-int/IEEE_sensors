"""
train_disentangled.py

Identity-disentanglement PoC. Trains the encoder end-to-end with TWO heads on
the 128-d vector:
  - blink head        : keep blink utility        (minimize blink BCE)
  - identity head     : remove identity           (GRADIENT REVERSAL -> encoder
                        is pushed to make identity UN-predictable while the head
                        itself tries to predict it)

Sweeps the adversary strength lambda and reports, for each lambda:
  - blink utility = ROC-AUC of the blink head on held-out frames (threshold-free)
  - privacy       = identity re-ID accuracy of a fresh attacker on frozen vectors
Produces the utility-vs-privacy PARETO curve (this project's Figure 1 idea).

PoC protocol (single consistent split): stratified by subject, 80/20 frame split
so ALL subjects appear in train and test (needed for a meaningful re-ID number).
This is a mechanism demonstration; the rigorous version (subject-separated,
event-level, many identities) is future work.

Run:
    pip install torch numpy matplotlib
    python -m src.encoder.train_disentangled --data data/processed/eyeblink8_eyes.npz \
        --lams 0.0 0.1 0.3 1.0 3.0 --epochs 25 --out models/disentangled
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.encoder.train_autoencoder import Encoder


# ---- gradient reversal ----
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


def grad_reverse(x, lam):
    return GradReverse.apply(x, lam)


class Disent(nn.Module):
    def __init__(self, latent, n_subj):
        super().__init__()
        self.encoder = Encoder(latent)
        self.blink = nn.Sequential(
            nn.Linear(latent, 64), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(64, 1))
        self.ident = nn.Sequential(
            nn.Linear(latent, 128), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(128, n_subj))

    def forward(self, x, lam):
        z = self.encoder(x)
        b = self.blink(z).squeeze(1)
        i = self.ident(grad_reverse(z, lam))
        return z, b, i


def roc_auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


class SubjMLP(nn.Module):
    def __init__(self, latent, n_cls):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent, 128), nn.ReLU(True),
                                 nn.Dropout(0.3), nn.Linear(128, n_cls))

    def forward(self, x):
        return self.net(x)


def attacker_acc(Ztr, ytr, Zte, yte, n_cls, device, epochs=40):
    clf = SubjMLP(Ztr.shape[1], n_cls).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    lf = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(Ztr, torch.from_numpy(ytr)), batch_size=256, shuffle=True)
    for _ in range(epochs):
        clf.train()
        for zb, yb in loader:
            zb, yb = zb.to(device), yb.to(device)
            opt.zero_grad(); lf(clf(zb), yb).backward(); opt.step()
    clf.eval()
    with torch.no_grad():
        pred = clf(Zte.to(device)).argmax(1).cpu().numpy()
    return float((pred == yte).mean())


@torch.no_grad()
def encode(model, X, device, batch=512):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).float().div_(255.0).unsqueeze(1).to(device)
        out.append(model.encoder(xb).cpu())
    return torch.cat(out)


def train_one(lam, X, yb_all, ys_all, tr, te, n_subj, latent, args):
    dev = args.device
    model = Disent(latent, n_subj).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    pos_w = torch.tensor([(yb_all[tr] == 0).sum() / max((yb_all[tr] == 1).sum(), 1)], device=dev)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    ce = nn.CrossEntropyLoss()

    Xt = torch.from_numpy(X[tr]).float().div_(255.0).unsqueeze(1)
    yb = torch.from_numpy(yb_all[tr])
    ys = torch.from_numpy(ys_all[tr])
    loader = DataLoader(TensorDataset(Xt, yb, ys), batch_size=args.batch, shuffle=True, drop_last=True)

    total_steps = args.epochs * max(len(loader), 1)
    step = 0
    for _ in range(args.epochs):
        model.train()
        for xb, ybb, ysb in loader:
            xb, ybb, ysb = xb.to(dev), ybb.to(dev), ysb.to(dev)
            if args.warmup:
                p = step / max(total_steps - 1, 1)          # 0 -> 1 over training
                lam_eff = lam * (2.0 / (1.0 + np.exp(-args.gamma * p)) - 1.0)
            else:
                lam_eff = lam
            opt.zero_grad()
            _, bpred, ipred = model(xb, lam_eff)
            loss = bce(bpred, ybb) + ce(ipred, ysb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            step += 1

    # utility: blink AUC on test
    model.eval()
    with torch.no_grad():
        Xte = torch.from_numpy(X[te]).float().div_(255.0).unsqueeze(1).to(dev)
        bprob = torch.sigmoid(model.blink(model.encoder(Xte))).squeeze(1).cpu().numpy()
    auc = roc_auc(yb_all[te], bprob)

    # privacy: re-ID attacker on frozen vectors (our split)
    Ztr = encode(model, X[tr], dev)
    Zte = encode(model, X[te], dev)
    reid = attacker_acc(Ztr, ys_all[tr], Zte, ys_all[te], n_subj, dev)

    torch.save(model.encoder.state_dict(), os.path.join(args.out, f"encoder_lam{lam}.pt"))
    return auc, reid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/eyeblink8_eyes.npz")
    ap.add_argument("--out", default="models/disentangled")
    ap.add_argument("--target", choices=["blink_event", "eye_closed"], default="blink_event")
    ap.add_argument("--lams", nargs="*", type=float, default=[0.0, 0.1, 0.3, 1.0, 3.0])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--latent", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True,
                    help="ramp lambda 0->target over training (DANN schedule) for stability")
    ap.add_argument("--gamma", type=float, default=10.0, help="warmup steepness")
    ap.add_argument("--clip", type=float, default=5.0, help="gradient-norm clip")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = np.load(args.data, allow_pickle=True)
    X = d["images"]
    yb_all = d[args.target].astype(np.float32)
    subj = d["subject"].astype(str)
    subs = sorted(set(subj.tolist()))
    sidx = {s: i for i, s in enumerate(subs)}
    ys_all = np.array([sidx[s] for s in subj], dtype=np.int64)
    n_subj = len(subs)

    # stratified 80/20 by subject
    rng = np.random.default_rng(args.seed)
    tr, te = [], []
    for s in range(n_subj):
        idx = np.where(ys_all == s)[0]
        rng.shuffle(idx)
        cut = int(0.8 * len(idx))
        tr += idx[:cut].tolist(); te += idx[cut:].tolist()
    tr, te = np.array(tr), np.array(te)
    print(f"subjects={n_subj} | train {len(tr)} | test {len(te)} | chance re-ID={1/n_subj:.3f} | device={args.device}")

    results = []
    for lam in args.lams:
        auc, reid = train_one(lam, X, yb_all, ys_all, tr, te, n_subj, args.latent, args)
        results.append({"lam": lam, "blink_auc": auc, "reid_acc": reid})
        print(f"  lam={lam:<4}  blink_AUC={auc:.3f}   re-ID={reid:.3f}  (chance {1/n_subj:.3f})")

    json.dump({"chance_reid": 1 / n_subj, "results": results},
              open(os.path.join(args.out, "pareto.json"), "w"), indent=2)

    # Pareto plot: utility (y) vs privacy leak (x)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [r["reid_acc"] for r in results]
        ys = [r["blink_auc"] for r in results]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(xs, ys, "o-", color="#2471a3")
        for r in results:
            ax.annotate(f"λ={r['lam']}", (r["reid_acc"], r["blink_auc"]),
                        textcoords="offset points", xytext=(6, 5), fontsize=9)
        ax.axvline(1 / n_subj, ls="--", color="green", alpha=0.6, label=f"re-ID chance ({1/n_subj:.2f})")
        ax.set_xlabel("Identity re-ID accuracy  (privacy leak -> lower is better)")
        ax.set_ylabel("Blink ROC-AUC  (utility -> higher is better)")
        ax.set_title("Utility vs identity leakage — disentanglement PoC (Eyeblink8)")
        ax.legend(loc="lower right"); ax.grid(alpha=0.3)
        os.makedirs("docs/figures", exist_ok=True)
        fig.tight_layout(); fig.savefig("docs/figures/pareto_disentangle.png", dpi=130)
        print("Saved docs/figures/pareto_disentangle.png")
    except Exception as e:
        print(f"(plot skipped: {e})")

    print("Done. Ideal direction: move LEFT (re-ID -> chance) with little UTILITY drop.")


if __name__ == "__main__":
    main()
