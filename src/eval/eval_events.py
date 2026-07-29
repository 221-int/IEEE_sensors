"""
eval_events.py

Event-level evaluation (the Phase-0 metric) for the frozen encoder + classifier.
A blink EVENT = all frames sharing the same blink_id (!= -1) within a clip.
An event is DETECTED if ANY of its frames is predicted positive at the threshold.

Reports, on the subject-separated validation clips:
  - event-level recall  = detected events / total events   (1st-priority metric)
  - frame-level false-alarm rate on non-blink frames (proxy for precision)
across a sweep of thresholds.

Run:
    python -m src.eval.eval_events --data data/processed/eyeblink8_eyes.npz \
        --encoder models/autoencoder/encoder.pt --classifier models/classifier/classifier.pt
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch

from src.encoder.train_autoencoder import Encoder
from src.classifier.train_classifier import MLP, encode_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/eyeblink8_eyes.npz")
    ap.add_argument("--encoder", default="models/autoencoder/encoder.pt")
    ap.add_argument("--classifier", default="models/classifier/classifier.pt")
    ap.add_argument("--config", default="models/autoencoder/config.json")
    ap.add_argument("--val-subjects", nargs="*", default=["eb04", "eb11"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    latent = json.load(open(args.config))["latent"] if os.path.exists(args.config) else 128
    d = np.load(args.data, allow_pickle=True)
    images, subject = d["images"], d["subject"].astype(str)
    blink_id = d["blink_id"].astype(int)

    val = np.isin(subject, args.val_subjects)

    encoder = Encoder(latent).to(args.device)
    encoder.load_state_dict(torch.load(args.encoder, map_location=args.device))
    clf = MLP(latent).to(args.device)
    clf.load_state_dict(torch.load(args.classifier, map_location=args.device))
    clf.eval()

    Z = encode_all(encoder, images[val], args.device).to(args.device)
    with torch.no_grad():
        prob = torch.sigmoid(clf(Z)).cpu().numpy()

    subj_v = subject[val]
    bid_v = blink_id[val]

    # group events by (subject, blink_id) for blink_id != -1
    events = defaultdict(list)
    for i in range(len(prob)):
        if bid_v[i] != -1:
            events[(subj_v[i], bid_v[i])].append(prob[i])
    n_events = len(events)
    neg_mask = bid_v == -1
    n_neg = int(neg_mask.sum())

    print(f"val subjects={args.val_subjects} | events={n_events} | non-blink frames={n_neg}\n")
    print(f"{'thr':>5} {'event_recall':>13} {'events_hit':>11} {'frame_false_alarm':>18}")
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        hit = sum(1 for ps in events.values() if max(ps) >= thr)
        far = float((prob[neg_mask] >= thr).mean()) if n_neg else 0.0
        print(f"{thr:5.2f} {hit / n_events:13.3f} {hit:6d}/{n_events:<4d} {far:18.4f}")

    print("\nevent_recall = fraction of real blinks caught (>=1 frame).")
    print("frame_false_alarm = fraction of non-blink frames wrongly flagged.")


if __name__ == "__main__":
    main()
