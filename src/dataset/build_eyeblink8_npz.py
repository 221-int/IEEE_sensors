"""
build_eyeblink8_npz.py

Process every Eyeblink8 clip under a root folder into ONE training-ready file:
    <out>.npz  with arrays:
        images      uint8  [N, 64, 64]   canonical one-eye crops
        subject     str    [N]           clip id (e.g. eb01)  -> use for subject-separated splits
        frame_id    int32  [N]
        blink_id    int32  [N]           -1 if not a blink frame
        blink_event uint8  [N]           1 if blink_id != -1
        eye_closed  uint8  [N]           1 if image-right eye fully closed (RE_FC == 'C')
    <out>_labels.csv  (same metadata, no images)

Reuses the canonical crop + tag parser (single source of truth).

Usage:
    python -m src.dataset.build_eyeblink8_npz --root data/_legacy_public/eyeblink8/eyeblink8 \
        --out data/processed/eyeblink8_eyes.npz --stride 2
"""

import argparse
import csv
import glob
import os

import cv2
import numpy as np

from src.dataset.eye_preprocess import crop_both_eyes_from_corners
from src.dataset.prepare_eyeblink8 import parse_tag


def process_clip(avi, tag, subject, stride):
    tags = parse_tag(tag)
    cap = cv2.VideoCapture(avi)
    imgs, meta = [], []
    fid = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fid % stride == 0 and fid in tags:
            t = tags[fid]
            crop = crop_both_eyes_from_corners(frame, t["le_a"], t["le_b"],
                                               t["re_a"], t["re_b"])
            if crop is not None:
                imgs.append(crop)
                meta.append((subject, fid, t["blink_id"],
                             1 if t["blink_id"] != -1 else 0, t["eye_closed"]))
        fid += 1
    cap.release()
    return imgs, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="folder containing numbered clip subfolders")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--only", nargs="*", default=None,
                    help="only process these clip subfolder names (e.g. 1 2)")
    args = ap.parse_args()

    clips = sorted(glob.glob(os.path.join(args.root, "*", "*_cam.avi")))
    if args.only:
        only = set(args.only)
        clips = [c for c in clips if os.path.basename(os.path.dirname(c)) in only]
    print(f"Found {len(clips)} clips.", flush=True)

    all_imgs, all_meta = [], []
    for avi in clips:
        tag = avi[:-4] + ".tag"
        if not os.path.exists(tag):
            print(f"  skip (no tag): {avi}", flush=True)
            continue
        subject = "eb" + os.path.basename(os.path.dirname(avi)).zfill(2)
        imgs, meta = process_clip(avi, tag, subject, args.stride)
        all_imgs += imgs
        all_meta += meta
        n_evt = sum(m[3] for m in meta)
        print(f"  {subject}: {len(imgs)} crops (blink_event={n_evt})", flush=True)

    X = np.stack(all_imgs).astype(np.uint8)
    subject = np.array([m[0] for m in all_meta])
    frame_id = np.array([m[1] for m in all_meta], np.int32)
    blink_id = np.array([m[2] for m in all_meta], np.int32)
    blink_event = np.array([m[3] for m in all_meta], np.uint8)
    eye_closed = np.array([m[4] for m in all_meta], np.uint8)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, images=X, subject=subject, frame_id=frame_id,
                        blink_id=blink_id, blink_event=blink_event,
                        eye_closed=eye_closed)

    csvp = os.path.splitext(args.out)[0] + "_labels.csv"
    with open(csvp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "subject", "frame_id", "blink_id", "blink_event", "eye_closed"])
        for i, m in enumerate(all_meta):
            w.writerow([i, m[0], m[1], m[2], m[3], m[4]])

    print(f"TOTAL {X.shape[0]} crops -> {args.out}", flush=True)
    print(f"  subjects: {sorted(set(subject.tolist()))}", flush=True)
    print(f"  blink_event=1: {int(blink_event.sum())} ({100*blink_event.mean():.2f}%)", flush=True)
    print(f"  eye_closed=1:  {int(eye_closed.sum())} ({100*eye_closed.mean():.2f}%)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
