"""
train_autoencoder.py

Branch B of the encoder: a convolutional AUTOENCODER trained (unsupervised) on
the one-eye 64x64 crops. The ENCODER is what we deploy (eye crop -> vector);
the DECODER is a LOCAL validation tool only (never deployed) used to eyeball
whether the vector actually captured the eye.

Input : data/processed/eyeblink8_eyes.npz  (images uint8 [N,64,64], subject, ...)
Output: <out>/encoder.pt, decoder.pt, autoencoder.pt, config.json, recon_val.png

Split : SUBJECT-SEPARATED (hold out whole clips for validation) so the encoder
        is not evaluated on people it memorized.

Run (desktop, GPU if available):
    pip install torch numpy opencv-python
    python -m src.encoder.train_autoencoder --data data/processed/eyeblink8_eyes.npz \
        --epochs 30 --batch 256 --latent 128 --out models/autoencoder
"""

import argparse
import json
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ----------------------------- data -----------------------------
class EyeDataset(Dataset):
    def __init__(self, images):
        # images: uint8 [N,64,64] -> float [N,1,64,64] in [0,1]
        self.x = torch.from_numpy(images).float().div_(255.0).unsqueeze(1)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i]


# ----------------------------- model -----------------------------
# Canonical input is the both-eyes crop (H x W). 4 stride-2 convs -> H/16 x W/16.
IN_HW = (64, 160)


class Encoder(nn.Module):
    def __init__(self, latent=128, in_hw=IN_HW):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
        )
        self.h16, self.w16 = in_hw[0] // 16, in_hw[1] // 16
        self.fc = nn.Linear(256 * self.h16 * self.w16, latent)

    def forward(self, x):
        h = self.net(x).flatten(1)
        return self.fc(h)


class Decoder(nn.Module):
    def __init__(self, latent=128, in_hw=IN_HW):
        super().__init__()
        self.h16, self.w16 = in_hw[0] // 16, in_hw[1] // 16
        self.fc = nn.Linear(latent, 256 * self.h16 * self.w16)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 256, self.h16, self.w16)
        return self.net(h)


class Autoencoder(nn.Module):
    def __init__(self, latent=128):
        super().__init__()
        self.encoder = Encoder(latent)
        self.decoder = Decoder(latent)

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ----------------------------- helpers -----------------------------
def save_recon_grid(model, val_x, device, path, n=12):
    model.eval()
    with torch.no_grad():
        x = val_x[:n].to(device)
        rec = model(x).cpu().numpy()
    x = x.cpu().numpy()
    rows = []
    for src in (x, rec):  # top row = input, bottom row = reconstruction
        row = [(np.clip(src[i, 0], 0, 1) * 255).astype(np.uint8) for i in range(n)]
        rows.append(np.hstack(row))
    grid = np.vstack(rows)
    cv2.imwrite(path, grid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/eyeblink8_eyes.npz")
    ap.add_argument("--out", default="models/autoencoder")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--latent", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-subjects", nargs="*", default=["eb04", "eb11"],
                    help="clips held out for validation (subject-separated)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    d = np.load(args.data, allow_pickle=True)
    images, subject = d["images"], d["subject"].astype(str)

    val_mask = np.isin(subject, args.val_subjects)
    train_x, val_x = images[~val_mask], images[val_mask]
    print(f"train {train_x.shape[0]} | val {val_x.shape[0]} "
          f"(val subjects={args.val_subjects}) | device={args.device}")

    train_loader = DataLoader(EyeDataset(train_x), batch_size=args.batch,
                              shuffle=True, num_workers=0, drop_last=True)
    val_tensor = EyeDataset(val_x).x

    model = Autoencoder(args.latent).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for xb in train_loader:
            xb = xb.to(args.device)
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            opt.step()
            tot += loss.item() * xb.size(0)
        train_loss = tot / len(train_loader.dataset)

        # validation
        model.eval()
        with torch.no_grad():
            vt = 0.0
            for i in range(0, val_tensor.size(0), args.batch):
                xb = val_tensor[i:i + args.batch].to(args.device)
                vt += loss_fn(model(xb), xb).item() * xb.size(0)
            val_loss = vt / val_tensor.size(0)

        print(f"epoch {epoch:3d}  train {train_loss:.5f}  val {val_loss:.5f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.encoder.state_dict(), os.path.join(args.out, "encoder.pt"))
            torch.save(model.decoder.state_dict(), os.path.join(args.out, "decoder.pt"))
            torch.save(model.state_dict(), os.path.join(args.out, "autoencoder.pt"))
            save_recon_grid(model, val_tensor, args.device,
                            os.path.join(args.out, "recon_val.png"))

    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({"latent": args.latent,
                   "input": f"1x{IN_HW[0]}x{IN_HW[1]}", "norm": "0..1",
                   "val_subjects": args.val_subjects, "best_val_mse": best_val}, f, indent=2)
    print(f"Done. best val MSE={best_val:.5f}. Artifacts in '{args.out}'.")
    print("recon_val.png: top row = input eye, bottom row = reconstruction.")


if __name__ == "__main__":
    main()
