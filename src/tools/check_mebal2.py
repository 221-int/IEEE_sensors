"""
check_mebal2.py

Re-derives the User 1 measurements in docs/mEBAL2_실측_2026-07-31.md §7 through
the loader, so that any change to mebal2_loader.py / eye_preprocess.py that
silently breaks the mEBAL2 geometry fails here instead of 58 users later.

    python -m src.tools.check_mebal2

Expected (measured 2026-07-31, User 1):
    frames                 37,641   landmark rows == video frames
    no face                 3,301   (8.8%)
    2-face frames           1,435   (3.8%)
    event 1339..1357        EAR minimum at frame 1348 = window centre
    crop span             125-132 px, crop_w 276-290 px
    sharpness              88-159  (Eyeblink8 reference 98)
    brightness           34.8-45.4 (Eyeblink8 reference 102.7 -- the known gap)
"""

import argparse
import os

import cv2
import numpy as np

from src.dataset.eye_preprocess import BOTH_MARGIN
from src.dataset.mebal2_loader import EVENT_LEN, MEBAL2User, OK, read_frames

CHECK_FRAMES = [1348, 1543, 1728, 5000, 20000]
CHECK_EVENT = (1339, 1357)
EB8_BRIGHTNESS, EB8_SHARPNESS = 102.7, 98.0


def line(ok, label, got, want=""):
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label:34s} {got}" + (f"   (expected {want})" if want else ""))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="data/raw/mEBAL2/_probe")
    ap.add_argument("--video", default="data/raw/mEBAL2/User 1/RealSense/"
                                       "Color_Webcam/color.mp4")
    ap.add_argument("--user", type=int, default=1)
    ap.add_argument("--frames", type=int, nargs="*", default=CHECK_FRAMES,
                    help="frames to crop-check (default: the 5 hand-checked ones)")
    ap.add_argument("--skip-csv-stats", action="store_true",
                    help="skip the whole-session coverage scan")
    ap.add_argument("--skip-video", action="store_true",
                    help="CSV checks only (no decoding; the 20,000th frame "
                         "costs a sequential decode)")
    args = ap.parse_args()

    u = MEBAL2User.from_probe(args.probe, args.user, video=args.video)
    passed = []

    # ---- 1. tables ----
    print("\n1. CSV tables")
    n_lm = len(u.landmarks)
    n_box = len(u.boxes)
    passed.append(line(n_lm == 37641, "landmark rows", n_lm, 37641))
    passed.append(line(n_box == n_lm, "box rows == landmark rows", n_box, n_lm))
    fmin, fmax = min(u.landmarks), max(u.landmarks)
    passed.append(line((fmin, fmax) == (0, 37640), "frame range (0-based)",
                       f"{fmin}..{fmax}", "0..37640"))

    # ---- 2. face resolution ----
    print("\n2. face resolution (Phase 0.3 / 0.4)")
    if args.skip_csv_stats:
        print("  [SKIP] whole-session coverage scan (--skip-csv-stats)")
    else:
        cov = u.coverage()
        passed.append(line(cov["no_face"] == 3301, "frames with no face",
                           f'{cov["no_face"]} ({cov["no_face_rate"]:.1%})',
                           "3301 (8.8%)"))
        passed.append(line(cov["multi_face"] == 1435, "frames with 2 faces",
                           cov["multi_face"], 1435))
        # box/mesh face-count disagreement -- why we do NOT assert equality
        dis = sum(1 for f in u.landmarks
                  if u.landmarks[f].shape[0]
                  != u.boxes.get(f, np.zeros((0, 4))).shape[0])
        print(f"  [note] box vs mesh face-count disagreement: {dis} frames "
              f"({dis / n_lm:.1%}) -- reconciled geometrically, not asserted")

    # ---- 3. events ----
    print("\n3. events")
    ev, isb = u.events()
    span = ev[:, 1] - ev[:, 0]
    passed.append(line(len(ev) > 0, "events loaded", len(ev)))
    passed.append(line(bool(np.all(span == EVENT_LEN - 1)),
                       "every window is 19 frames",
                       f"end-start unique={sorted(set(span.tolist()))}", "[18]"))
    n_b, n_u = int(isb.sum()), int((1 - isb).sum())
    passed.append(line(n_b == n_u, "blink count == unblink count",
                       f"{n_b} / {n_u}"))
    all_ev, _ = u.events(include_unconfirmed=True)
    print(f"  [note] Blink=0 (possible blink) excluded: "
          f"{len(all_ev) - len(ev)} events dropped")

    # ---- 4. EAR alignment on the hand-checked event ----
    print("\n4. EAR vs label window (frame-numbering agreement)")
    s, e = CHECK_EVENT
    ears = [(f, u.ear(f)) for f in range(s, e + 1)]
    ears = [(f, v) for f, v in ears if v is not None]
    d = dict(ears)
    fmin_ear = min(ears, key=lambda t: t[1])[0]
    centre = (s + e) // 2
    passed.append(line(abs(fmin_ear - centre) <= 1, f"EAR minimum in {s}..{e}",
                       f"frame {fmin_ear} (EAR {d[fmin_ear]:.3f})",
                       f"within 1 frame of centre {centre}"))
    # the substantive claim: the dip is INSIDE the window, not at its edges
    edge = (d[s] + d[e]) / 2.0
    passed.append(line(d[fmin_ear] < 0.8 * edge, "dip is interior, not at edges",
                       f"min {d[fmin_ear]:.3f} vs edge mean {edge:.3f}"))
    print("        " + "  ".join(f"{f}:{v:.3f}" for f, v in ears
                                 if f in (s, centre - 1, centre, e)))
    print("  [note] docs §7 lists 1339:0.273 1347:0.210 1348:0.214 1357:0.357 -- "
          "uniformly ~13% higher.\n"
          "         Those were computed ad hoc with a different 6-point set; the "
          "repo set (eye_preprocess.EAR_EYE_A/B,\n"
          "         shared with ear_baseline) gives the values above. Same "
          "conclusion, different absolute scale --\n"
          "         do not quote the doc's numbers next to baseline results.")

    # ---- 5. crop geometry ----
    if not args.skip_video:
        print("\n5. crop geometry (Phase 1 pre-check)")
        if not os.path.exists(args.video):
            print(f"  [SKIP] video not found: {args.video}")
        else:
            print(f"  {'frame':>6} {'faces':>5} {'span':>7} {'crop_w':>7} "
                  f"{'bright':>7} {'sharp':>7}")
            rows = []
            for f, bgr in read_frames(args.video, args.frames):
                sel = u.face(f)
                if sel.status != OK:
                    print(f"  {f:6d}  no face ({sel.status})")
                    continue
                c = u.eye_corners(f)
                le = (c[0] + c[1]) / 2.0
                re = (c[2] + c[3]) / 2.0
                sp = float(np.linalg.norm(le - re))
                crop = u.crop(bgr, f)
                bright = float(crop.mean())
                sharp = float(cv2.Laplacian(crop, cv2.CV_64F).var())
                rows.append((f, sel.n_mesh, sp, sp * BOTH_MARGIN, bright, sharp))
                print(f"  {f:6d} {sel.n_mesh:5d} {sp:7.1f} {sp * BOTH_MARGIN:7.1f} "
                      f"{bright:7.1f} {sharp:7.1f}")
            if rows:
                a = np.array([[r[2], r[3], r[4], r[5]] for r in rows])
                passed.append(line(bool(np.all((a[:, 0] > 120) & (a[:, 0] < 140))),
                                   "span in 120..140 px",
                                   f"{a[:, 0].min():.1f}..{a[:, 0].max():.1f}"))
                passed.append(line(bool(np.all(a[:, 3] > 60)),
                                   "sharpness > 60 (EB8 ref 98)",
                                   f"{a[:, 3].min():.0f}..{a[:, 3].max():.0f}"))
                print(f"  [note] brightness {a[:, 2].min():.1f}..{a[:, 2].max():.1f} "
                      f"vs Eyeblink8 {EB8_BRIGHTNESS} -- known gap, NOT a gate "
                      f"(mEBAL2 is trained standalone)")

    n_ok = sum(passed)
    print(f"\n{n_ok}/{len(passed)} checks passed")
    return 0 if n_ok == len(passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
