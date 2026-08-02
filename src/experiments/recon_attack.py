"""
recon_attack.py

"디코더를 배포하지 않으므로 원본 복원 불가" 라는 주장에 대한 반론 실험.

위협 모델
---------
공격자는 저장·전송되는 **벡터를 가지고 있다**(그것이 이 시스템이 내보내는 전부다).
공격자는 우리 디코더는 없지만, **자기 손으로 (이미지, 벡터) 쌍을 만들 수 있다** —
배포된 인코더에 자기 이미지를 넣으면 되고, 공개 눈 데이터셋을 쓰면 된다.
따라서 공격자는 **디코더를 새로 학습**할 수 있다. 우리 디코더를 안 배포하는 것은
장애물이지 불가능성이 아니다.

프로토콜
--------
    공격자 학습 : train 피험자들의 (z, x) 쌍으로 디코더를 처음부터 학습
    공격 대상   : victim 피험자 (학습에 전혀 등장하지 않은 신원)
    성공 지표   : victim 프레임 복원 PSNR / MAE

대조군이 핵심이다. PSNR 숫자 하나는 의미가 없다 — 눈 크롭은 저분산 그레이스케일이라
**아무 정보 없이도 PSNR이 높게 나온다**. 그래서 네 가지를 함께 보고한다:

    ours        우리 디코더 (인코더와 함께 학습됨) — 상한
    attacker    공격자가 새로 학습한 디코더 — 실제 공격 성공도
    mean        학습셋 평균 이미지를 그대로 출력 — **정보 0의 하한**
    shuffled    공격자 디코더에 짝이 안 맞는 벡터를 넣음 — 디코더가 벡터를
                실제로 읽는지, 아니면 평균 얼굴을 외운 것인지 판별하는 통제군

`attacker` 가 `shuffled` 나 `mean` 과 비슷하면 복원은 사실상 실패한 것이다.
`attacker` 가 그 둘보다 뚜렷이 높으면 벡터가 외형을 흘리고 있다는 뜻이다.

Run:
    python -m src.experiments.recon_attack
    python -m src.experiments.recon_attack --victims eb01 eb02 --epochs 30
    python -m src.experiments.recon_attack --all-folds        # 4개 victim 조합 전부
"""

import argparse
import json
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.encoder.train_autoencoder import Encoder, Decoder
from src.experiments.split_eval import FOLDS, SUBJECTS, set_seed, to_tensor


@torch.no_grad()
def encode_all(encoder, images, device, batch=512):
    encoder.eval()
    return torch.cat([encoder(to_tensor(images[i:i + batch]).to(device)).cpu()
                      for i in range(0, len(images), batch)])


def psnr_mae(x, rec):
    """x, rec: float arrays in [0,1]. -> (psnr_dB, mae)"""
    mse = float(np.mean((x - rec) ** 2))
    p = 10 * np.log10(1.0 / mse) if mse > 0 else 99.0
    return p, float(np.mean(np.abs(x - rec)))


def train_attacker(Z_tr, X_tr, latent, device, epochs, batch, lr, log):
    """공격자 디코더: (벡터 -> 이미지) 를 처음부터 학습. 우리 디코더는 보지 않는다."""
    dec = Decoder(latent).to(device)
    opt = torch.optim.Adam(dec.parameters(), lr=lr)
    lossf = nn.MSELoss()
    dl = DataLoader(TensorDataset(Z_tr, X_tr), batch_size=batch, shuffle=True)
    for ep in range(1, epochs + 1):
        dec.train()
        tot = 0.0
        for zb, xb in dl:
            zb, xb = zb.to(device), xb.to(device)
            opt.zero_grad()
            loss = lossf(dec(zb), xb)
            loss.backward(); opt.step()
            tot += loss.item() * len(zb)
        if log and (ep % 10 == 0 or ep == 1):
            print(f"      attacker dec ep {ep:3d}  train MSE {tot/len(dl.dataset):.5f}",
                  flush=True)
    dec.eval()
    return dec


@torch.no_grad()
def decode_all(dec, Z, device, batch=256):
    return torch.cat([dec(Z[i:i + batch].to(device)).cpu()
                      for i in range(0, len(Z), batch)]).numpy()[:, 0]


def save_grid(path, rows, labels, n=10):
    """rows: list of arrays [N,H,W] in [0,1]. 한 줄에 하나씩 쌓아 PNG로 저장."""
    H, W = rows[0].shape[1:]
    pad, lab = 2, 16
    canvas = np.full(((H + pad) * len(rows) + lab, (W + pad) * n + pad), 30, np.uint8)
    for r, arr in enumerate(rows):
        y = lab + r * (H + pad)
        for c in range(min(n, len(arr))):
            img = (np.clip(arr[c], 0, 1) * 255).astype(np.uint8)
            canvas[y:y + H, pad + c * (W + pad):pad + c * (W + pad) + W] = img
    bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    for r, t in enumerate(labels):
        cv2.putText(bgr, t, (4, lab + r * (H + pad) - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, bgr)


def run_one(images, subject, victims, enc, our_dec, latent, args):
    dev = args.device
    m_te = np.isin(subject, victims)
    m_tr = ~m_te
    Z = encode_all(enc, images, dev)
    X = to_tensor(images)

    # --- 공격자 디코더 학습 (victim 신원은 전혀 보지 않음)
    dec_a = train_attacker(Z[m_tr], X[m_tr], latent, dev,
                           args.epochs, args.batch, args.lr, not args.quiet)

    x_te = X[m_te].numpy()[:, 0]
    out = {}

    # ours (인코더와 함께 학습된 디코더) — 상한
    if our_dec is not None:
        out["ours"] = psnr_mae(x_te, decode_all(our_dec, Z[m_te], dev))
    # attacker
    rec_a = decode_all(dec_a, Z[m_te], dev)
    out["attacker"] = psnr_mae(x_te, rec_a)
    # mean image (정보 0)
    mean_img = X[m_tr].numpy()[:, 0].mean(0)
    out["mean"] = psnr_mae(x_te, np.broadcast_to(mean_img, x_te.shape))
    # shuffled vectors (디코더가 벡터를 실제로 읽는가)
    rng = np.random.default_rng(0)
    perm = rng.permutation(int(m_te.sum()))
    rec_s = decode_all(dec_a, Z[m_te][perm], dev)
    out["shuffled"] = psnr_mae(x_te, rec_s)

    if args.figure:
        idx = np.linspace(0, len(x_te) - 1, 10).astype(int)
        rows = [x_te[idx]]
        labels = ["original (victim)"]
        if our_dec is not None:
            rows.append(decode_all(our_dec, Z[m_te][idx], dev)); labels.append("ours decoder")
        rows.append(rec_a[idx]); labels.append("ATTACKER decoder (retrained)")
        rows.append(np.broadcast_to(mean_img, (10, *mean_img.shape)))
        labels.append("mean image (no info)")
        rows.append(rec_s[idx]); labels.append("attacker + shuffled vector")
        p = args.figure.replace(".png", f"_{'-'.join(victims)}.png")
        save_grid(p, rows, labels)
        print(f"    figure -> {p}")

    return out, int(m_te.sum()), int(m_tr.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/eyeblink8_eyes.npz")
    ap.add_argument("--encoder", default="models/autoencoder/encoder.pt")
    ap.add_argument("--decoder", default="models/autoencoder/decoder.pt",
                    help="우리 디코더 (상한 비교용). 없으면 생략")
    ap.add_argument("--config", default="models/autoencoder/config.json")
    ap.add_argument("--victims", nargs="*", default=["eb01", "eb02"])
    ap.add_argument("--all-folds", action="store_true",
                    help="split_eval 의 4개 test 조합을 모두 victim 으로")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--figure", default="docs/figures/recon_attack.png")
    ap.add_argument("--json", default="results/recon_attack.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    latent = (json.load(open(args.config))["latent"]
              if os.path.exists(args.config) else 128)
    d = np.load(args.data, allow_pickle=True)
    images, subject = d["images"], d["subject"].astype(str)

    enc = Encoder(latent).to(args.device)
    enc.load_state_dict(torch.load(args.encoder, map_location=args.device))
    enc.eval()

    our_dec = None
    if args.decoder and os.path.exists(args.decoder):
        our_dec = Decoder(latent).to(args.device)
        our_dec.load_state_dict(torch.load(args.decoder, map_location=args.device))
        our_dec.eval()
    else:
        print(f"(우리 디코더 {args.decoder} 없음 -> 상한 비교 생략)")

    victim_sets = [f["test"] for f in FOLDS] if args.all_folds else [args.victims]
    print(f"latent {latent} | device {args.device} | "
          f"공격자 디코더 {args.epochs} epoch | victim sets {victim_sets}\n")

    rows = []
    for vs in victim_sets:
        print(f"=== victim = {vs} (이 신원은 공격자 학습에 없음) ===", flush=True)
        set_seed(args.seed)
        out, n_te, n_tr = run_one(images, subject, vs, enc, our_dec, latent, args)
        rows.append({"victims": vs, "n_victim_frames": n_te,
                     "n_attacker_train_frames": n_tr,
                     **{k: {"psnr_db": round(v[0], 2), "mae": round(v[1], 4)}
                        for k, v in out.items()}})
        s = "  ".join(f"{k} {v[0]:.2f}dB" for k, v in out.items())
        print(f"    {s}\n", flush=True)

    # ------------------------------------------------------------- report
    keys = [k for k in ["ours", "attacker", "mean", "shuffled"] if k in rows[0]]
    print("=" * 76)
    print("  복원 공격 — victim 신원은 공격자 학습에 포함되지 않음")
    print("=" * 76)
    print(f"  {'victim':>14} " + "".join(f"{k:>12}" for k in keys))
    print("  " + "-" * (15 + 12 * len(keys)))
    for r in rows:
        print(f"  {'+'.join(r['victims']):>14} " +
              "".join(f"{r[k]['psnr_db']:>11.2f}dB" for k in keys))
    print("  " + "-" * (15 + 12 * len(keys)))
    avg = {k: np.mean([r[k]["psnr_db"] for r in rows]) for k in keys}
    print(f"  {'평균':>14} " + "".join(f"{avg[k]:>11.2f}dB" for k in keys))

    print("\n  해석")
    gap_mean = avg["attacker"] - avg["mean"]
    gap_shuf = avg["attacker"] - avg["shuffled"]
    print(f"   attacker - mean     = {gap_mean:+.2f} dB  "
          f"(평균 이미지보다 이만큼 더 복원한다)")
    print(f"   attacker - shuffled = {gap_shuf:+.2f} dB  "
          f"(벡터를 실제로 읽는 정도)")
    if "ours" in avg:
        print(f"   attacker / ours     = {avg['attacker']:.2f} vs {avg['ours']:.2f} dB  "
              f"(디코더 미배포로 얻는 방어력)")
    print()
    if gap_shuf < 1.0:
        print("   -> 공격자 디코더가 벡터를 거의 읽지 못한다. 평균 얼굴을 외운 수준.")
    else:
        print("   -> 공격자가 벡터에서 외형 정보를 실제로 복원한다.")
        print("      '디코더 미배포'는 방어가 아니라 지연에 불과하다는 뜻이므로,")
        print("      프라이버시 주장은 표현 자체(disentanglement)에 걸어야 한다.")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        json.dump({"config": vars(args), "latent": latent, "results": rows},
                  open(args.json, "w"), indent=2, default=str)
        print(f"\n  -> {args.json}")


if __name__ == "__main__":
    main()
