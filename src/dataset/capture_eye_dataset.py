"""
capture_eye_dataset.py

Webcam eye-crop dataset collector for the on-device eye-representation project.

What it does
------------
- Captures webcam frames (your desktop/laptop webcam)
- Detects eye landmarks with MediaPipe FaceMesh
- Crops ONE eye -> grayscale -> 64x64  (this is the CANONICAL preprocessing:
  reuse crop_eye() EXACTLY at serve/inference time to avoid train/serve skew)
- Computes EAR and auto-labels blink (0/1) via a threshold  (WEAK label)
- Saves crops + a labels.csv:  filename, subject_id, ear, blink

Usage
-----
    pip install opencv-python mediapipe numpy
    python -m src.dataset.capture_eye_dataset --subject me
    python -m src.dataset.capture_eye_dataset --subject friend

Keys while running
------------------
    q : quit and save

Notes
-----
- EAR labels are WEAK (threshold-based, noisy on partial blinks / head pose).
  Use them for ENCODER training (labels don't matter) and as weak labels for
  the classifier. Do the honest EAR comparison on a MANUALLY labeled test set
  (e.g. Eyeblink8), NOT on this EAR-labeled data.
- --thresh may need tuning per person / lighting (watch the on-screen EAR value:
  it should drop clearly when you close your eye).
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np
import mediapipe as mp

# Canonical crop geometry is shared with dataset prep / serve time.
from src.dataset.eye_preprocess import crop_eye_from_corners

# Some mediapipe installs (esp. Windows / Python 3.12) don't expose the lazy
# `mp.solutions` attribute. Resolve the face_mesh module robustly.
try:
    _face_mesh_mod = mp.solutions.face_mesh
except AttributeError:
    from mediapipe.python.solutions import face_mesh as _face_mesh_mod

# MediaPipe FaceMesh indices for the image-right eye (the subject's left eye).
# 6-point EAR set: two corners + two top/bottom pairs.
EYE = {
    "corner_out": 33,
    "corner_in": 133,
    "top1": 160, "bot1": 144,
    "top2": 158, "bot2": 153,
}
EAR_BLINK_THRESH = 0.21  # default; tune per person/lighting


def _pt(landmarks, idx, w, h):
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def compute_ear(landmarks, w, h):
    """Eye Aspect Ratio (Soukupova & Cech)."""
    p_out = _pt(landmarks, EYE["corner_out"], w, h)
    p_in = _pt(landmarks, EYE["corner_in"], w, h)
    t1 = _pt(landmarks, EYE["top1"], w, h)
    b1 = _pt(landmarks, EYE["bot1"], w, h)
    t2 = _pt(landmarks, EYE["top2"], w, h)
    b2 = _pt(landmarks, EYE["bot2"], w, h)
    horiz = np.linalg.norm(p_out - p_in) + 1e-6
    vert = np.linalg.norm(t1 - b1) + np.linalg.norm(t2 - b2)
    return float(vert / (2.0 * horiz))


def crop_eye(frame_bgr, landmarks):
    """Canonical eye crop from MediaPipe landmarks (delegates to shared geometry)."""
    h, w = frame_bgr.shape[:2]
    p_out = _pt(landmarks, EYE["corner_out"], w, h)
    p_in = _pt(landmarks, EYE["corner_in"], w, h)
    return crop_eye_from_corners(frame_bgr, p_out, p_in)


def main():
    ap = argparse.ArgumentParser()
    default_out = os.path.join(os.path.expanduser("~"), "Downloads", "eye_dataset")
    ap.add_argument("--subject", default=None, help="subject id, e.g. me / friend")
    ap.add_argument("--out", default=default_out, help="output folder")
    ap.add_argument("--cam", type=int, default=0, help="webcam index")
    ap.add_argument("--width", type=int, default=1280, help="capture width")
    ap.add_argument("--height", type=int, default=720, help="capture height")
    ap.add_argument("--thresh", type=float, default=EAR_BLINK_THRESH,
                    help="EAR blink threshold (blink if EAR < thresh)")
    args = ap.parse_args()

    # If run without --subject (e.g. clicking Run in PyCharm), ask interactively.
    if not args.subject:
        args.subject = input("Subject id (e.g. me / friend): ").strip() or "me"

    sub_dir = os.path.join(args.out, args.subject)
    os.makedirs(sub_dir, exist_ok=True)
    csv_path = os.path.join(args.out, "labels.csv")
    new_csv = not os.path.exists(csv_path)
    csv_f = open(csv_path, "a", newline="")
    writer = csv.writer(csv_f)
    if new_csv:
        writer.writerow(["filename", "subject_id", "ear", "blink"])

    face_mesh = _face_mesh_mod.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Pick capture backends per OS (DirectShow is Windows-only; macOS uses AVFoundation).
    if sys.platform == "darwin":
        backends = (cv2.CAP_AVFOUNDATION, cv2.CAP_ANY)
    elif sys.platform.startswith("win"):
        backends = (cv2.CAP_DSHOW, cv2.CAP_ANY)
    else:
        backends = (cv2.CAP_ANY,)

    def open_cam(index):
        for backend in backends:
            c = cv2.VideoCapture(index, backend)
            if c.isOpened() and c.read()[0]:
                return c
            c.release()
        return None

    cap = open_cam(args.cam)
    if cap is None:
        print(f"Camera index {args.cam} did not open. Scanning indices 0..3 ...")
        for i in range(4):
            cap = open_cam(i)
            if cap is not None:
                print(f"Using camera index {i}")
                break
    if cap is None:
        print("ERROR: no working webcam found. Check the USB connection and "
              "camera permissions (Windows Settings > Privacy > Camera).")
        return

    # Request higher capture resolution so the (small) eye region has more real
    # pixels before it is cropped to 64x64. Bigger source -> crisper crop.
    # MJPG lets most USB webcams deliver higher resolution / fps over DirectShow;
    # the default format (YUY2) often caps at 640x480.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Capture resolution: {aw}x{ah}")
    if (aw, ah) == (640, 480):
        print("NOTE: still 640x480. The webcam may max out here, or another "
              "camera may be higher-res (try --cam 1 / --cam 2). Sitting closer "
              "to the camera helps the most.")

    idx = 0
    saved = 0
    print("Recording. Look at the camera and blink naturally. Press 'q' to stop.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        vis = frame.copy()

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            ear = compute_ear(lm, frame.shape[1], frame.shape[0])
            crop = crop_eye(frame, lm)
            if crop is not None:
                blink = int(ear < args.thresh)
                fname = f"{args.subject}_{idx:06d}.png"
                cv2.imwrite(os.path.join(sub_dir, fname), crop)
                writer.writerow([os.path.join(args.subject, fname),
                                 args.subject, f"{ear:.4f}", blink])
                saved += 1

                preview = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_NEAREST)
                if GRAYSCALE:
                    preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
                vis[0:128, 0:128] = preview
                color = (0, 255, 0) if blink == 0 else (0, 0, 255)
                cv2.putText(vis, f"EAR {ear:.2f}  blink={blink}  saved={saved}",
                            (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("capture (press q to quit)", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        idx += 1

    cap.release()
    cv2.destroyAllWindows()
    csv_f.close()
    print(f"Done. Saved {saved} eye crops to '{sub_dir}'. Labels -> '{csv_path}'.")


if __name__ == "__main__":
    main()
