"""
ear_baseline.py

Classic EAR (Eye Aspect Ratio) blink baseline, evaluated on the SAME validation
clips and the SAME event-level metric as the vector model -> honest head-to-head
for Phase-0's "vector model >= EAR baseline" claim.

EAR needs 6 eyelid landmarks, which the Eyeblink8 .tag does NOT contain, so we
run MediaPipe FaceMesh on the actual frames. To be apples-to-apples with our
one-eye vector model, we use the ONE (image-right) eye's EAR. A frame is a blink
prediction if EAR < threshold. Blink events are grouped by the tag's blink_id.

Run (desktop/laptop with mediapipe working):
    python -m src.eval.ear_baseline --root data/_legacy_public/eyeblink8/eyeblink8 \
        --val-subjects eb04 eb11 --stride 2
"""

import argparse
import os
from collections import defaultdict

import cv2

from src.dataset.prepare_eyeblink8 import parse_tag
from src.dataset.capture_eye_dataset import compute_ear, _face_mesh_mod


def subj_to_folder(subj):
    # "eb04" -> "4", "eb11" -> "11"
    return str(int(subj[2:]))


def collect_clip(avi, tag, face_mesh, stride):
    """Return list of (blink_id, ear_or_None) for the stride-sampled frames."""
    tags = parse_tag(tag)
    cap = cv2.VideoCapture(avi)
    out = []
    fid = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fid % stride == 0 and fid in tags:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                ear = compute_ear(lm, frame.shape[1], frame.shape[0])
            else:
                ear = None  # no detection -> cannot flag a blink
            out.append((tags[fid]["blink_id"], ear))
        fid += 1
    cap.release()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/_legacy_public/eyeblink8/eyeblink8")
    ap.add_argument("--val-subjects", nargs="*", default=["eb04", "eb11"])
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()

    face_mesh = _face_mesh_mod.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    rows = []  # (clip, blink_id, ear)
    n_missing = 0
    for subj in args.val_subjects:
        folder = subj_to_folder(subj)
        avis = [p for p in
                [os.path.join(args.root, folder, f) for f in os.listdir(os.path.join(args.root, folder))]
                if p.endswith("_cam.avi")]
        if not avis:
            print(f"  {subj}: no clip found in {folder}")
            continue
        avi = avis[0]
        tag = avi[:-4] + ".tag"
        data = collect_clip(avi, tag, face_mesh, args.stride)
        for bid, ear in data:
            rows.append((subj, bid, ear))
            if ear is None:
                n_missing += 1
        print(f"  {subj}: {len(data)} frames processed")

    # group events by (clip, blink_id)
    events = defaultdict(list)
    neg_ears = []
    for subj, bid, ear in rows:
        if bid != -1:
            events[(subj, bid)].append(ear)
        else:
            neg_ears.append(ear)
    n_events = len(events)
    n_neg = len(neg_ears)
    print(f"\nval subjects={args.val_subjects} | events={n_events} | non-blink frames={n_neg}")
    print(f"MediaPipe no-detection frames: {n_missing}\n")

    def below(ear, thr):
        return ear is not None and ear < thr

    print(f"{'EAR_thr':>7} {'event_recall':>13} {'events_hit':>11} {'frame_false_alarm':>18}")
    for thr in [0.15, 0.18, 0.20, 0.22, 0.25]:
        hit = sum(1 for ears in events.values() if any(below(e, thr) for e in ears))
        far = (sum(below(e, thr) for e in neg_ears) / n_neg) if n_neg else 0.0
        print(f"{thr:7.2f} {hit / n_events:13.3f} {hit:6d}/{n_events:<4d} {far:18.4f}")

    print("\nCompare event_recall / frame_false_alarm against the vector model "
          "(eval_events.py). Same clips, same metric, one eye each.")


if __name__ == "__main__":
    main()
