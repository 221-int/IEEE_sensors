"""
find_cam.py - probe available webcams and their max resolution.

Run:
    python -m src.tools.find_cam

For each camera index it opens the device, requests MJPG @ 1920x1080, reads a
frame, and prints the resolution the camera actually delivers. Use the index
that reports the highest resolution as your webcam.
"""

import sys
import cv2

if sys.platform == "darwin":
    BACKEND = cv2.CAP_AVFOUNDATION
elif sys.platform.startswith("win"):
    BACKEND = cv2.CAP_DSHOW
else:
    BACKEND = cv2.CAP_ANY

for i in range(5):
    cap = cv2.VideoCapture(i, BACKEND)
    if not cap.isOpened():
        print(f"index {i}: not available")
        cap.release()
        continue
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    ok, frame = cap.read()
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"index {i}: opened, frame={'ok' if ok else 'FAIL'}, delivers ~{w}x{h}")
    cap.release()

print("\nUse the index with the highest resolution, e.g.:")
print("    python capture_eye_dataset.py --subject me --cam <index> --width 1280 --height 720")
