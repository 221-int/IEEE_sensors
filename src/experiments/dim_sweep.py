"""
dim_sweep.py

Roadmap (3): latent dimension sweep (64 / 128 / 256).

One experiment, two questions, per STATUS_AND_DIRECTION §4-(3):

  (a) BLINK UTILITY   -- does a bigger latent recover the accuracy we lost when
      the input went from 64x64 (one eye) to 64x160 (both eyes)? The working
      hypothesis is "the image grew 2.5x but the bottleneck stayed at 128".
  (b) IDENTITY CAPACITY -- how much subject information is LINEARLY decodable
      from the vector at each width? If 128 really is a tight bottleneck, it is
      also a bottleneck for identity, which is the space disentanglement needs.

Both are measured on the SAME trained encoder per width, so the two curves are
directly comparable and can be plotted against each other.

Reproducibility principle (§0)
------------------------------
Everything that depends on the dataset is a parameter: `--val-subjects`,
`--seeds`, `--data`. When mEBAL2 (180 subjects) arrives this script should be a
RE-RUN, not a re-design -- only the loader and the subject list change.

Hyperparameters are copied verbatim from the standalone scripts so a sweep point
at latent=128 is comparable to the models already in models/:
    autoencoder : 30 epochs, batch 256, Adam lr 1e-3, MSE
    classifier  : 40 epochs, batch 256, Adam lr 1e-3, BCEWithLogits(pos_weight)

Metrics reported per width
--------------------------
    recon_val_mse      autoencoder reconstruction on held-out subjects
    event_recall@FA    blink events caught, at matched frame false-alarm rates
                       (the headline utility metric, same definition as
                       eval_events.py: an event counts if ANY frame fires)
    reid_linear        identity accuracy from a LINEAR probe  <- §4-(3) asks for
                       linear specifically; probe capacity must not confound the
                       comparison across widths
    reid_mlp           identity accuracy from the 1-hidden-layer probe used by
                       privacy_metrics.py, kept for continuity with past numbers
    chance             1 / n_subjects

Run:
    python -m src.experiments.dim_sweep --dims 64 128 256
    python -m src.experiments.dim_sweep --dims 64 128 256 --seeds 0 1 2   # slower
    python -m src.experiments.dim_sweep --dims 128 --epochs-ae 2 --epochs-clf 2  # smoke
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


# ----------------------------------------------------------------- utilities
def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def to_tensor(images):
    """uint8 [N,H,W] -> float32 [N,1,H,W] in 0..1 (same as training)."""
    return torch.from_numpy(images).float().div_(255.0).unsqueeze(1)


@torch.no_grad()
def encode_all(encoder, images, device, batch=512):
    encoder.eval()
    zs = []
    for i in range(0, len(images), batch):
        xb = to_tensor(images[i:i + batch]).to(device)
        zs.append(encoder(xb).cpu())
    return torch.cat(zs)


# ----------------------------------------------------------------- stage 1: AE
def train_autoencoder(images, val_mask, latent, device, epochs, batch, lr, log):
    train_x, val_x = images[~val_mask], images[val_mask]
    loader = DataLoader(TensorDataset(to_tensor(train_x)), batch_size=batch,
                        shuffle=True, drop_last=True)
    val_t = to_tensor(val_x)

    model = Autoencoder(latent).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best = float("inf")
    best_state = None
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            opt.step()
            tot += loss.item() * xb.size(0)
        model.eval()
        with torch.no_grad():
            vt = 0.0
            for i in range(0, val_t.size(0), batch):
                xb = val_t[i:i + batch].to(device)
                vt += loss_fn(model(xb), xb).item() * xb.size(0)
            val_loss = vt / val_t.size(0)
        if val_loss < best:
            best = val_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.encoder.state_dict().items()}
        if log and (ep % 10 == 0 or ep == 1):
            print(f"      AE  epoch {ep:3d}  train {tot/len(loader.dataset):.5f}  "
                  f"val {val_loss:.5f}", flush=True)

    enc = Encoder(latent).to(device)
    enc.load_state_dict(best_state)
    enc.eval()
    return enc, best


# --------------------------------------------------------- stage 2: classifier
class MLP(nn.Module):
    """Same architecture as src/classifier/train_classifier.MLP."""

    def __init__(self, latent, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_classifier(Z, y, val_mask, latent, device, epochs, batch, lr, log):
    Ztr = Z[~val_mask]
    ytr = torch.from_numpy(y[~val_mask])
    n_pos, n_neg = int(ytr.sum()), int((ytr == 0).sum())

    clf = MLP(latent).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([n_neg / max(n_pos, 1)], device=device))
    loader = DataLoader(TensorDataset(Ztr, ytr), batch_size=batch, shuffle=True)

    for ep in range(1, epochs + 1):
        clf.train()
        for zb, yb in loader:
            zb, yb = zb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(clf(zb), yb).backward()
            opt.step()
        if log and (ep % 20 == 0 or ep == 1):
            print(f"      CLF epoch {ep:3d}", flush=True)

    clf.eval()
    with torch.no_grad():
        prob = torch.sigmoid(clf(Z[val_mask].to(device))).cpu().numpy()
    return prob


# --------------------------------------------------------- stage 3: utility
def event_curve(prob, blink_id_val, subject_val):
    """Same metric as eval_events.py -> list of (thr, recall, hit, n_events, FA)."""
    events = defaultdict(list)
    for i in range(len(prob)):
        if blink_id_val[i] != -1:
            events[(subject_val[i], blink_id_val[i])].append(prob[i])
    n_events = len(events)
    neg = prob[blink_id_val == -1]
    rows = []
    # dense sweep so we can read off recall at MATCHED false-alarm rates
    for thr in np.linspace(0.02, 0.98, 97):
        hit = sum(1 for ps in events.values() if max(ps) >= thr)
        far = float((neg >= thr).mean()) if len(neg) else 0.0
        rows.append((float(thr), hit / n_events if n_events else 0.0,
                     hit, n_events, far))
    return rows


def recall_at_fa(rows, target_fa):
    """Best event recall achievable at frame false-alarm <= target.

    Returns None when the target false-alarm is UNREACHABLE (the model never
    gets that clean at any threshold). Reporting 0.0 there would read as
    "recall zero", which is a different and much worse statement.
    """
    ok = [r for r in rows if r[4] <= target_fa]
    return max((r[1] for r in ok), default=None)


def fa_at_recall(rows, target_recall):
    """Lowest false-alarm at which event recall >= target (None if unreachable)."""
    ok = [r for r in rows if r[1] >= target_recall]
    return min((r[4] for r in ok), default=None)


# --------------------------------------------------------- stage 4: identity
def probe(Z, subject, device, linear, epochs, seed):
    """Identity probe. linear=True -> single Linear layer (no hidden, no ReLU)."""
    subs = sorted(set(subject.tolist()))
    idx = {s: i for i, s in enumerate(subs)}
    y = np.array([idx[s] for s in subject], dtype=np.int64)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    cut = int(0.7 * len(y))
    tr, te = perm[:cut], perm[cut:]

    d = Z.shape[1]
    net = (nn.Linear(d, len(subs)) if linear else
           nn.Sequential(nn.Linear(d, 128), nn.ReLU(True), nn.Dropout(0.3),
                         nn.Linear(128, len(subs)))).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(Z[tr], torch.from_numpy(y[tr])),
                        batch_size=256, shuffle=True)
    for _ in range(epochs):
        net.train()
        for zb, yb in loader:
            zb, yb = zb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(net(zb), yb).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(Z[te].to(device)).argmax(1).cpu().numpy()
    return float((pred == y[te]).mean()), 1.0 / len(subs), len(subs)


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/eyeblink8_eyes.npz")
    ap.add_argument("--dims", nargs="*", type=int, default=[64, 128, 256])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--val-subjects", nargs="*", default=["eb04", "eb11"])
    ap.add_argument("--target", default="blink_event")
    ap.add_argument("--epochs-ae", type=int, default=30)
    ap.add_argument("--epochs-clf", type=int, default=40)
    ap.add_argument("--epochs-probe", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--fa-points", nargs="*", type=float, default=[0.01, 0.03, 0.05])
    ap.add_argument("--recall-points", nargs="*", type=float, default=[0.95, 0.97])
    ap.add_argument("--save-encoders", default=None,
                    help="directory to save each width's encoder (optional)")
    ap.add_argument("--json", default="results/dim_sweep.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    images = d["images"]
    subject = d["subject"].astype(str)
    blink_id = d["blink_id"].astype(int)
    y = d[args.target].astype(np.float32)

    val_mask = np.isin(subject, args.val_subjects)
    n_sub = len(set(subject.tolist()))
    print(f"data {images.shape} | subjects {n_sub} | "
          f"val={args.val_subjects} ({int(val_mask.sum())} frames) | "
          f"device={args.device}")
    print(f"dims={args.dims} seeds={args.seeds} "
          f"AE {args.epochs_ae}ep / CLF {args.epochs_clf}ep\n", flush=True)

    results = []
    t_start = time.perf_counter()
    for dim in args.dims:
        for seed in args.seeds:
            t0 = time.perf_counter()
            print(f"=== latent {dim}  seed {seed} ===", flush=True)
            set_seed(seed)

            enc, val_mse = train_autoencoder(
                images, val_mask, dim, args.device, args.epochs_ae,
                args.batch, args.lr, not args.quiet)

            Z = encode_all(enc, images, args.device)

            prob = train_classifier(Z, y, val_mask, dim, args.device,
                                    args.epochs_clf, args.batch, args.lr,
                                    not args.quiet)
            rows = event_curve(prob, blink_id[val_mask], subject[val_mask])

            acc_lin, chance, n_s = probe(Z, subject, args.device, True,
                                         args.epochs_probe, seed)
            acc_mlp, _, _ = probe(Z, subject, args.device, False,
                                  args.epochs_probe, seed)

            rec = {
                "latent": dim, "seed": seed,
                "recon_val_mse": val_mse,
                "recall_at_fa": {str(f): recall_at_fa(rows, f) for f in args.fa_points},
                "fa_at_recall": {str(r): fa_at_recall(rows, r) for r in args.recall_points},
                "max_event_recall": max(r[1] for r in rows),
                "n_events": rows[0][3],
                "reid_linear": acc_lin, "reid_mlp": acc_mlp,
                "chance": chance, "n_subjects": n_s,
                "seconds": round(time.perf_counter() - t0, 1),
            }
            results.append(rec)

            if args.save_encoders:
                dd = os.path.join(args.save_encoders, f"latent{dim}_seed{seed}")
                os.makedirs(dd, exist_ok=True)
                torch.save(enc.state_dict(), os.path.join(dd, "encoder.pt"))
                json.dump({"latent": dim, "input": "1x64x160", "norm": "0..1",
                           "val_subjects": args.val_subjects,
                           "best_val_mse": val_mse},
                          open(os.path.join(dd, "config.json"), "w"), indent=2)

            _r = rec["recall_at_fa"][str(args.fa_points[1])]
            print(f"    -> recon {val_mse:.5f} | "
                  f"recall@FA3% {'unreachable' if _r is None else f'{_r:.3f}'} | "
                  f"reID lin {acc_lin:.3f} mlp {acc_mlp:.3f} (chance {chance:.3f}) | "
                  f"{rec['seconds']:.0f}s\n", flush=True)

    # ------------------------------------------------------------ report
    print("=" * 78)
    print(f"  LATENT DIMENSION SWEEP   ({len(results)} runs, "
          f"{time.perf_counter()-t_start:.0f}s total)")
    print("=" * 78)
    hdr = (f"  {'latent':>6} {'seed':>4} {'recon':>8}" +
           "".join(f"{'R@FA'+str(int(f*100))+'%':>9}" for f in args.fa_points) +
           "".join(f"{'FA@R'+str(int(r*100)):>9}" for r in args.recall_points) +
           f"{'maxR':>7}{'reID_lin':>9}{'reID_mlp':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in results:
        line = f"  {r['latent']:>6} {r['seed']:>4} {r['recon_val_mse']:>8.5f}"
        for f in args.fa_points:
            v = r["recall_at_fa"][str(f)]
            line += f"{'      n/a' if v is None else f'{v:>9.3f}'}"
        for rc in args.recall_points:
            v = r["fa_at_recall"][str(rc)]
            line += f"{'  n/a' if v is None else f'{v:>9.4f}'}"
        line += f"{r['max_event_recall']:>7.3f}{r['reid_linear']:>9.3f}{r['reid_mlp']:>9.3f}"
        print(line)
    print(f"\n  chance re-ID = {results[0]['chance']:.3f} "
          f"({results[0]['n_subjects']} subjects) | events = {results[0]['n_events']}")
    print("  R@FAx% = event recall reachable at frame false-alarm <= x%")
    print("  FA@Rxx = lowest frame false-alarm at which event recall >= 0.xx")
    print("\n  읽는 법: 'R@FA'가 latent와 함께 오르면 128이 병목이라는 가설이 지지된다.")
    print("           'reID_lin'이 함께 오르면 그 여유 공간은 신원도 함께 담는다는 뜻이고,")
    print("           그것이 disentanglement가 제거해야 할 대상이다.")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        json.dump({"config": vars(args), "results": results},
                  open(args.json, "w"), indent=2, default=str)
        print(f"\n  -> {args.json}")


if __name__ == "__main__":
    main()
