"""
privacy_metrics.py

Quantify how much the current (reconstruction-trained) vector leaks, on two axes:

  (1) Identity re-identification: train a classifier vector -> subject and measure
      test accuracy vs chance. HIGH accuracy => the vector still carries identity
      (bad for privacy). This motivates identity-disentanglement.
      Protocol: frame-level stratified split (SAME subjects in train/test, DIFFERENT
      frames) -- this is the attacker's task.

  (2) Reconstruction quality: decoder(encoder(x)) vs x, mean PSNR on held-out
      frames. HIGHER PSNR => more eye appearance recoverable (less private).

NOTE: Eyeblink8 has only ~8 clips, so the re-ID number is a proof-of-concept,
not a strong claim. A many-identity dataset is needed for the real Pareto curve.

Run:
    python -m src.eval.privacy_metrics --data data/processed/eyeblink8_eyes.npz \
        --encoder models/autoencoder/encoder.pt --decoder models/autoencoder/decoder.pt
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.encoder.train_autoencoder import Encoder, Decoder
from src.classifier.train_classifier import encode_all


class SubjMLP(nn.Module):
    def __init__(self, latent, n_cls, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(hidden, n_cls))

    def forward(self, x):
        return self.net(x)


def reid(Z, subject, device, epochs=40, seed=0):
    subs = sorted(set(subject.tolist()))
    idx = {s: i for i, s in enumerate(subs)}
    y = np.array([idx[s] for s in subject], dtype=np.int64)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    cut = int(0.7 * len(y))
    tr, te = perm[:cut], perm[cut:]

    clf = SubjMLP(Z.shape[1], len(subs)).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(Z[tr], torch.from_numpy(y[tr])),
                        batch_size=256, shuffle=True)
    for _ in range(epochs):
        clf.train()
        for zb, yb in loader:
            zb, yb = zb.to(device), yb.to(device)
            opt.zero_grad(); loss_fn(clf(zb), yb).backward(); opt.step()
    clf.eval()
    with torch.no_grad():
        pred = clf(Z[te].to(device)).argmax(1).cpu().numpy()
    acc = float((pred == y[te]).mean())
    return acc, 1.0 / len(subs), len(subs)


def recon_psnr(encoder, decoder, images, device, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(images), size=min(n, len(images)), replace=False)
    x = torch.from_numpy(images[sel]).float().div_(255.0).unsqueeze(1).to(device)
    with torch.no_grad():
        rec = decoder(encoder(x))
    mse = ((x - rec) ** 2).mean().item()
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else 99.0
    return psnr, mse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/eyeblink8_eyes.npz")
    ap.add_argument("--encoder", default="models/autoencoder/encoder.pt")
    ap.add_argument("--decoder", default="models/autoencoder/decoder.pt")
    ap.add_argument("--config", default="models/autoencoder/config.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    latent = json.load(open(args.config))["latent"] if os.path.exists(args.config) else 128
    d = np.load(args.data, allow_pickle=True)
    images, subject = d["images"], d["subject"].astype(str)

    encoder = Encoder(latent).to(args.device)
    encoder.load_state_dict(torch.load(args.encoder, map_location=args.device))
    Z = encode_all(encoder, images, args.device)

    acc, chance, n_cls = reid(Z, subject, args.device)
    print("\n=== (1) Identity re-identification (attacker) ===")
    print(f"  vector -> subject accuracy: {acc:.3f}   (chance = {chance:.3f}, {n_cls} subjects)")
    print(f"  leakage ratio vs chance: {acc/chance:.1f}x")
    print("  HIGH => vector still carries identity -> motivates disentanglement.")

    decoder = Decoder(latent).to(args.device)
    decoder.load_state_dict(torch.load(args.decoder, map_location=args.device))
    psnr, mse = recon_psnr(encoder, decoder, images, args.device)
    print("\n=== (2) Reconstruction quality (recoverability) ===")
    print(f"  mean PSNR: {psnr:.2f} dB  (MSE {mse:.5f})")
    print("  HIGHER PSNR => more eye appearance recoverable from the vector.")

    json.dump({"reid_acc": acc, "reid_chance": chance, "n_subjects": n_cls,
               "recon_psnr_db": psnr}, open("models/privacy_metrics.json", "w"), indent=2)
    print("\nSaved -> models/privacy_metrics.json")


if __name__ == "__main__":
    main()
