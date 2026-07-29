"""
ear_baseline.py

Classic EAR (Eye Aspect Ratio) blink baseline, evaluated on the SAME validation
clips and the SAME event-level metric as the vector model -> honest head-to-head
for Phase-0's "vector model >= EAR baseline" claim.

EAR needs 6 eyelid landmarks, which the Eyeblink8 .tag does NOT contain, so we
run MediaPipe FaceMesh on the actual frames. A frame is a blink prediction if
EAR < threshold. Blink events are grouped by the tag's blink_id.

BOTH EYES (roadmap (2))
-----------------------
The vector model moved to a both-eyes crop (64x160), so the baseline has to move
too or the comparison is not apples-to-apples. Every threshold sweep below is
reported for four variants computed in ONE pass:

    legacy   the single eye the old baseline used (landmark set 33/133/...).
             Kept so the previously reported one-eye numbers stay reproducible.
    left     image-left eye   -- assigned by actual x at runtime, so the labels
    right    image-right eye     are honest regardless of frame mirroring
    mean     (left + right) / 2   <- the direct both-eyes counterpart
    min      min(left, right)     <- fires when EITHER eye closes

    MEASURED (eb04+eb11, 74 events): legacy == left on every frame, i.e. the
    33/133 landmark set is the IMAGE-LEFT eye. The comment in
    capture_eye_dataset.EYE calling it "the image-right eye" is wrong.
    This matters historically: the one-eye vector model was trained on the .tag
    RE corners = the image-RIGHT eye, so the old head-to-head in RESULTS.md
    compared EAR on one eye against the model on the OTHER eye. Both sides are
    both-eyes now, so the current table is matched.

`mean` is the like-for-like both-eyes baseline. `min` is included because
STATUS_AND_DIRECTION §1 argues the value of both eyes is left/right ASYMMETRY
(partial and single-eye blinks); min vs mean is the cheapest way to show that a
scalar cannot represent it -- averaging washes a one-eye closure out, and taking
the min throws away the fact that only one eye moved. Either way a single number
loses the asymmetry, which is the argument for a vector.

Run (desktop/laptop with mediapipe working):
    python -m src.eval.ear_baseline --root data/_legacy_public/eyeblink8/eyeblink8 \
        --val-subjects eb04 eb11 --stride 2
"""

import argparse
import os
from collections import defaultdict

import cv2
import numpy as np

from src.dataset.prepare_eyeblink8 import parse_tag
from src.dataset.capture_eye_dataset import compute_ear, _face_mesh_mod

# 6-point EAR landmark sets, MediaPipe FaceMesh.
# EYE_A is the set the original one-eye baseline used (see capture_eye_dataset.EYE).
# EYE_B is its mirror on the other eye. Which one falls on the image-left depends
# on mirroring, so we resolve that at runtime from the corner x coordinates.
EYE_A = {"corner_out": 33,  "corner_in": 133,
         "top1": 160, "bot1": 144, "top2": 158, "bot2": 153}
EYE_B = {"corner_out": 263, "corner_in": 362,
         "top1": 385, "bot1": 380, "top2": 387, "bot2": 373}


def _pt(landmarks, idx, w, h):
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def ear_from_indices(landmarks, w, h, idx):
    """Eye Aspect Ratio (Soukupova & Cech) for one eye, given its 6 indices.

    Same formula as capture_eye_dataset.compute_ear, parameterized by eye.
    """
    p_out = _pt(landmarks, idx["corner_out"], w, h)
    p_in = _pt(landmarks, idx["corner_in"], w, h)
    t1 = _pt(landmarks, idx["top1"], w, h)
    b1 = _pt(landmarks, idx["bot1"], w, h)
    t2 = _pt(landmarks, idx["top2"], w, h)
    b2 = _pt(landmarks, idx["bot2"], w, h)
    horiz = np.linalg.norm(p_out - p_in) + 1e-6
    vert = np.linalg.norm(t1 - b1) + np.linalg.norm(t2 - b2)
    return float(vert / (2.0 * horiz))


VARIANTS = ["legacy", "left", "right", "mean", "min"]


def ears_from_landmarks(landmarks, w, h):
    """-> dict of the 5 EAR variants for one frame."""
    ear_a = ear_from_indices(landmarks, w, h, EYE_A)
    ear_b = ear_from_indices(landmarks, w, h, EYE_B)
    xa = (_pt(landmarks, EYE_A["corner_out"], w, h)[0]
          + _pt(landmarks, EYE_A["corner_in"], w, h)[0]) / 2.0
    xb = (_pt(landmarks, EYE_B["corner_out"], w, h)[0]
          + _pt(landmarks, EYE_B["corner_in"], w, h)[0]) / 2.0
    left, right = (ear_a, ear_b) if xa <= xb else (ear_b, ear_a)
    return {"legacy": ear_a, "left": left, "right": right,
            "mean": (left + right) / 2.0, "min": min(left, right)}


def subj_to_folder(subj):
    # "eb04" -> "4", "eb11" -> "11"
    return str(int(subj[2:]))


def collect_clip(avi, tag, face_mesh, stride, check_legacy=True):
    """Return list of (blink_id, ears_dict_or_None) for the stride-sampled frames."""
    tags = parse_tag(tag)
    cap = cv2.VideoCapture(avi)
    out = []
    fid = 0
    checked = False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fid % stride == 0 and fid in tags:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                h, w = frame.shape[0], frame.shape[1]
                ears = ears_from_landmarks(lm, w, h)
                # Regression guard: the 'legacy' variant must reproduce the
                # original one-eye implementation exactly, or the previously
                # reported numbers are not comparable.
                if check_legacy and not checked:
                    ref = compute_ear(lm, w, h)
                    assert abs(ref - ears["legacy"]) < 1e-9, (
                        f"legacy EAR drifted: {ref} vs {ears['legacy']}")
                    checked = True
            else:
                ears = None  # no detection -> cannot flag a blink
            out.append((tags[fid]["blink_id"], ears))
        fid += 1
    cap.release()
    return out


def sweep(events, neg, thresholds):
    """events: {key: [ear...]}, neg: [ear...]  -> list of (thr, recall, hit, far)."""
    n_events = len(events)
    n_neg = len(neg)
    rows = []
    for thr in thresholds:
        hit = sum(1 for ears in events.values()
                  if any(e is not None and e < thr for e in ears))
        far = (sum(1 for e in neg if e is not None and e < thr) / n_neg) if n_neg else 0.0
        rows.append((thr, hit / n_events if n_events else 0.0, hit, far))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/_legacy_public/eyeblink8/eyeblink8")
    ap.add_argument("--val-subjects", nargs="*", default=["eb04", "eb11"])
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--thresholds", nargs="*", type=float, default=None,
                    help="EAR thresholds (default: dense 0.10..0.35). A coarse "
                         "grid understates BOTH sides of the comparison -- keep "
                         "the vector model's grid equally dense or the "
                         "head-to-head is not fair.")
    ap.add_argument("--dense", type=int, default=51)
    ap.add_argument("--json", default=None, help="write the sweep to a JSON file")
    args = ap.parse_args()
    if args.thresholds is None:
        args.thresholds = [round(t, 4) for t in np.linspace(0.10, 0.35, args.dense)]

    face_mesh = _face_mesh_mod.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    rows = []  # (clip, blink_id, ears_dict_or_None)
    n_missing = 0
    for subj in args.val_subjects:
        folder = subj_to_folder(subj)
        sub_path = os.path.join(args.root, folder)
        avis = [os.path.join(sub_path, f) for f in os.listdir(sub_path)
                if f.endswith("_cam.avi")]
        if not avis:
            print(f"  {subj}: no clip found in {folder}")
            continue
        avi = avis[0]
        tag = avi[:-4] + ".tag"
        data = collect_clip(avi, tag, face_mesh, args.stride)
        for bid, ears in data:
            rows.append((subj, bid, ears))
            if ears is None:
                n_missing += 1
        print(f"  {subj}: {len(data)} frames processed", flush=True)

    # group events by (clip, blink_id)
    events = defaultdict(list)
    negs = []
    for subj, bid, ears in rows:
        if bid != -1:
            events[(subj, bid)].append(ears)
        else:
            negs.append(ears)
    n_events = len(events)
    n_neg = len(negs)
    print(f"\nval subjects={args.val_subjects} | events={n_events} | "
          f"non-blink frames={n_neg}")
    print(f"MediaPipe no-detection frames: {n_missing}\n")

    out = {"val_subjects": args.val_subjects, "stride": args.stride,
           "n_events": n_events, "n_nonblink_frames": n_neg,
           "n_missing_detection": n_missing, "variants": {}}

    for v in VARIANTS:
        ev = {k: [e[v] if e is not None else None for e in lst]
              for k, lst in events.items()}
        ng = [e[v] if e is not None else None for e in negs]
        res = sweep(ev, ng, args.thresholds)
        label = {"legacy": "legacy (one eye, previous baseline)",
                 "left": "left eye only", "right": "right eye only",
                 "mean": "BOTH eyes, mean  <- like-for-like baseline",
                 "min": "BOTH eyes, min   <- fires if either eye closes"}[v]
        print(f"--- {label} ---")
        print(f"{'EAR_thr':>7} {'event_recall':>13} {'events_hit':>11} {'frame_false_alarm':>18}")
        for thr, rec, hit, far in res:
            print(f"{thr:7.2f} {rec:13.3f} {hit:6d}/{n_events:<4d} {far:18.4f}")
        print()
        out["variants"][v] = [{"thr": t, "event_recall": r, "events_hit": h,
                               "frame_false_alarm": f} for t, r, h, f in res]

    if args.json:
        import json
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"sweep -> {args.json}")

    print("Compare event_recall / frame_false_alarm against the vector model "
          "(eval_events.py). Same clips, same metric.\n"
          "The vector model is both-eyes, so 'mean' is the row that belongs in "
          "the head-to-head table; 'legacy' is kept only to reproduce the older "
          "one-eye numbers.")


if __name__ == "__main__":
    main()
