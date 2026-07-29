"""
prepare_eyeblink8.py

Convert an Eyeblink8-style clip (.avi + .tag) into a cropped, labeled eye
dataset that matches our canonical preprocessing (one eye, gray 64x64).

Eyeblink8 .tag line format (colon-separated), per annotated frame:
    frameID : blinkID : NF : LE_FC : LE_NV : RE_FC : RE_NV : F_X : F_Y : ...
    ... : <eye corner coords> ...
- blinkID == -1  -> frame is NOT part of any blink event
- RE_FC == 'C'   -> the image-right eye is FULLY CLOSED in this frame
- the LAST 8 numbers on the line are the 4 eye corners:
      LE_Lx LE_Ly LE_Rx LE_Ry  RE_Lx RE_Ly RE_Rx RE_Ry
  i.e. the image-right eye's two corners are the last 4 numbers.

We crop the IMAGE-RIGHT eye (matches capture_eye_dataset.py) and write:
    <out>/<subject>/<subject>_<frame>.png
    <out>/labels.csv   ->  filename, subject_id, frame_id, blink_id, blink_event, eye_closed

Labels (both provided so we can choose frame-level vs event-level later):
    blink_event = 1 if blinkID != -1        (frame belongs to a blink event)
    eye_closed  = 1 if RE_FC == 'C'          (image-right eye fully closed)

Manual annotations => independent ground truth (NOT EAR). Use as the honest
test set for the EAR-vs-model comparison.

Usage:
    python -m src.dataset.prepare_eyeblink8 --avi clip.avi --tag clip.tag
    python -m src.dataset.prepare_eyeblink8 --avi clip.avi --tag clip.tag --subject eb01 --stride 2
"""

import argparse
import csv
import os

import cv2
import numpy as np

from src.dataset.eye_preprocess import crop_eye_from_corners, OUT_SIZE


def parse_tag(tag_path):
    """Return {frame_id: {blink_id, eye_closed(bool), corner_a, corner_b}}."""
    rows = {}
    started = False
    with open(tag_path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line == "#start":
                started = True
                continue
            if not started or not line or line.startswith("#"):
                continue
            toks = line.split(":")
            if len(toks) < 11:
                continue
            try:
                fid = int(toks[0])
                blink_id = int(toks[1])
            except ValueError:
                continue
            re_fc = toks[5].strip().upper()  # image-right eye fully-closed flag
            # last 8 numbers = 4 eye corners: LE_L, LE_R, RE_L, RE_R
            try:
                c = [int(round(float(t))) for t in toks[-8:]]
            except ValueError:
                continue
            rows[fid] = {
                "blink_id": blink_id,
                "eye_closed": 1 if re_fc == "C" else 0,
                # left (image-left) eye corners
                "le_a": (c[0], c[1]), "le_b": (c[2], c[3]),
                # right (image-right) eye corners
                "re_a": (c[4], c[5]), "re_b": (c[6], c[7]),
                # backward-compat one-eye (= right eye)
                "corner_a": (c[4], c[5]), "corner_b": (c[6], c[7]),
            }
    return rows


def build_montage(samples, cols=10, cell=64, pad=2):
    """samples: list of (img64, label_text). Returns a BGR montage image."""
    if not samples:
        return None
    rows = (len(samples) + cols - 1) // cols
    H = rows * (cell + 14) + pad
    W = cols * (cell + pad) + pad
    canvas = np.full((H, W, 3), 30, np.uint8)
    for i, (img, txt) in enumerate(samples):
        r, c = divmod(i, cols)
        y = pad + r * (cell + 14)
        x = pad + c * (cell + pad)
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
        canvas[y:y + cell, x:x + cell] = bgr
        color = (0, 0, 255) if txt.startswith("B") else (0, 200, 0)
        cv2.putText(canvas, txt, (x, y + cell + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--avi", required=True)
    ap.add_argument("--tag", required=True)
    default_out = os.path.join(os.path.expanduser("~"), "Downloads", "eyeblink8_eyes")
    ap.add_argument("--out", default=default_out)
    ap.add_argument("--subject", default=None, help="subject id (default: file stem)")
    ap.add_argument("--stride", type=int, default=2,
                    help="keep every Nth frame (2 => 30fps -> 15fps)")
    args = ap.parse_args()

    subject = args.subject or os.path.splitext(os.path.basename(args.avi))[0]
    sub_dir = os.path.join(args.out, subject)
    os.makedirs(sub_dir, exist_ok=True)

    tags = parse_tag(args.tag)
    print(f"Parsed {len(tags)} annotated frames from tag.")

    cap = cv2.VideoCapture(args.avi)
    if not cap.isOpened():
        print(f"ERROR: cannot open video {args.avi}")
        return

    csv_path = os.path.join(args.out, "labels.csv")
    new_csv = not os.path.exists(csv_path)
    csv_f = open(csv_path, "a", newline="")
    writer = csv.writer(csv_f)
    if new_csv:
        writer.writerow(["filename", "subject_id", "frame_id",
                         "blink_id", "blink_event", "eye_closed"])

    fid = 0
    saved = 0
    n_event = 0
    montage_blink, montage_open = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fid % args.stride == 0 and fid in tags:
            t = tags[fid]
            crop = crop_eye_from_corners(frame, t["corner_a"], t["corner_b"])
            if crop is not None:
                blink_event = 1 if t["blink_id"] != -1 else 0
                fname = f"{subject}_{fid:06d}.png"
                cv2.imwrite(os.path.join(sub_dir, fname), crop)
                writer.writerow([os.path.join(subject, fname), subject, fid,
                                 t["blink_id"], blink_event, t["eye_closed"]])
                saved += 1
                n_event += blink_event
                if blink_event and len(montage_blink) < 30:
                    montage_blink.append((crop, f"B{fid}"))
                elif not blink_event and len(montage_open) < 30:
                    montage_open.append((crop, f"o{fid}"))
        fid += 1

    cap.release()
    csv_f.close()

    montage = build_montage(montage_blink + montage_open)
    if montage is not None:
        mpath = os.path.join(args.out, f"montage_{subject}.png")
        cv2.imwrite(mpath, montage)
        print(f"Montage -> {mpath}")

    print(f"Saved {saved} eye crops ({OUT_SIZE}x{OUT_SIZE}) to '{sub_dir}'.")
    print(f"  blink-event frames: {n_event}  |  non-blink: {saved - n_event}")
    print(f"Labels -> {csv_path}")


if __name__ == "__main__":
    main()
