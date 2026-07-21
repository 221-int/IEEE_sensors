"""calibrate — per-subject EAR calibration experiment.

Procedure (per subject, automatic):
  1) "eyes open, relaxed" for OPEN_SEC s   -> collect open-eye EAR
  2) "eyes closed" for CLOSED_SEC s        -> collect closed-eye EAR
  3) fix per-subject thresholds (baseline/floor/STD)
  4) TRIALS times: on "BLINK NOW" cue, check STD counts exactly 1 in the window
  Output CSV (results/):
    calib_<subject>_frames.csv   per-frame EAR (with phase) — for stats
    calib_<subject>_trials.csv   per-trial detected/correct
    calib_summary.csv            per-subject summary (thresholds, EAR stats, accuracy)

Continuous mode: after a subject finishes, it asks for the next subject name
(Enter = quit). Measure several people in one run / one server / one model load,
without retyping the command per person.

Run (from src/):
  python -m scripts.calibrate --subject hanul --source webcam --show
  python -m scripts.calibrate --subject hanul --source url --url http://192.168.0.18:8000/stream.mjpg --show
    -> after hanul, prompts: "Next subject name (Enter = quit):"
"""
import argparse
import os
import time

from eyeblink import config
from eyeblink import landmarks as L
from eyeblink.experiment import (CalibrationRecorder, write_frames_csv,
                                 write_trials_csv, append_summary_csv)
from scripts.run_live import frame_source


def _overlay(cv2, frame, text, color=(0, 220, 0)):
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imshow("calibrate", frame)


def _run_window(gen, cv2, mp, landmarker, seconds, prompt, show, on_ear=None):
    """Pull EAR from frames for `seconds`. Call on_ear(t, ear) if provided."""
    t0 = time.time()
    while True:
        t = time.time()
        remain = seconds - (t - t0)
        if remain <= 0:
            break
        frame = next(gen, None)
        if frame is None:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(t * 1000))
        got_face = bool(res.face_landmarks)
        if got_face and on_ear is not None:
            on_ear(t, L.frame_ear(res.face_landmarks[0], w, h))
        if show:
            tag = prompt if got_face else prompt + "  (no face)"
            _overlay(cv2, frame, f"{tag}  {remain:.1f}s",
                     (0, 220, 0) if got_face else (0, 0, 255))
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                raise KeyboardInterrupt


def run_subject(subject, a, cv2, mp, landmarker, gen):
    """Run one subject's calibration + verification, save CSVs. Returns completed(bool)."""
    rec = CalibrationRecorder(subject)
    ema = {"v": None}

    def feed_smoothed(t, ear):
        ema["v"] = ear if ema["v"] is None else \
            config.EMA_ALPHA * ear + (1 - config.EMA_ALPHA) * ema["v"]
        if rec.feed_trial(t, ema["v"]):
            print("  -> STD counted!")

    print(f"\n[calib] ===== subject = {subject} =====")
    completed = False
    try:
        # 1) open
        _run_window(gen, cv2, mp, landmarker, 3.0, "READY: open-eye measure soon", a.show)
        print("[1/3] Keep your eyes open and relaxed...")
        _run_window(gen, cv2, mp, landmarker, a.open_sec, "OPEN eyes (relax)", a.show, rec.add_open)

        # 2) closed
        _run_window(gen, cv2, mp, landmarker, 3.0, "READY: close eyes soon", a.show)
        print("[2/3] Close your eyes...")
        _run_window(gen, cv2, mp, landmarker, a.closed_sec, "CLOSE eyes", a.show, rec.add_closed)

        # 3) fix thresholds
        c = rec.finalize_thresholds()
        if c["baseline"] is None or c["closed_floor"] is None:
            print("[calib] WARN calibration failed: face/eyes not captured. "
                  "Fix lighting/distance and retry this subject.")
        else:
            print(f"[calib] baseline={c['baseline']:.3f}  floor={c['closed_floor']:.3f}  "
                  f"t_open={c['t_open']:.3f}  t_closed={c['t_closed']:.3f}")

            # 4) blink verification trials
            print(f"[3/3] Blink ONCE on each cue. {a.trials} trials total.")
            for i in range(1, a.trials + 1):
                _run_window(gen, cv2, mp, landmarker, 1.5, f"Trial {i}/{a.trials}: READY", a.show)
                print(f"  Trial {i}: BLINK NOW!")
                ema["v"] = None
                rec.begin_trial(i)
                _run_window(gen, cv2, mp, landmarker, a.trial_sec,
                            f"Trial {i}/{a.trials}: BLINK NOW", a.show, feed_smoothed)
                detected, correct = rec.end_trial(expected=1)
                print(f"  Trial {i}: detected={detected}  {'OK' if correct else 'X'}")
            completed = True
    except KeyboardInterrupt:
        print(f"\n[calib] '{subject}' interrupted (saving what was collected).")

    # save (even if partial)
    os.makedirs(a.outdir, exist_ok=True)
    fp = os.path.join(a.outdir, f"calib_{subject}_frames.csv")
    tp = os.path.join(a.outdir, f"calib_{subject}_trials.csv")
    sp = os.path.join(a.outdir, "calib_summary.csv")
    write_frames_csv(rec, fp)
    write_trials_csv(rec, tp)
    append_summary_csv(rec, sp)

    tr = rec.summary_trials()
    print(f"[calib] {subject} done. accuracy {tr['correct']}/{tr['trials']} "
          f"({tr['accuracy']*100:.0f}%)")
    print(f"  frames: {fp}\n  trials: {tp}\n  summary: {sp}")
    return completed


def _ask_next_subject(done):
    """Prompt for next subject name. Enter/EOF/q -> quit (None)."""
    try:
        s = input(f"\n[calib] done: {', '.join(done) if done else '(none)'}"
                  "\nNext subject name (Enter = quit): ").strip()
    except EOFError:
        return None
    if not s or s.lower() == "q":
        return None
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True, help="first subject name/ID (e.g. hanul)")
    ap.add_argument("--source", choices=["url", "webcam"], default="url")
    ap.add_argument("--url",
                    default=f"http://127.0.0.1:{config.STREAM_PORT}{config.STREAM_PATH}")
    ap.add_argument("--camera", type=int, default=config.CAM_INDEX)
    ap.add_argument("--open-sec", type=float, default=config.CALIB_OPEN_SEC)
    ap.add_argument("--closed-sec", type=float, default=config.CALIB_CLOSED_SEC)
    ap.add_argument("--trials", type=int, default=config.CALIB_TRIALS)
    ap.add_argument("--trial-sec", type=float, default=config.CALIB_TRIAL_SEC)
    ap.add_argument("--outdir", default=config.RESULTS_DIR)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--once", action="store_true",
                    help="do a single subject and exit (no continuous prompt)")
    a = ap.parse_args()

    if not L.mediapipe_available():
        raise SystemExit("mediapipe is required:  pip install mediapipe")
    import cv2
    import mediapipe as mp
    try:
        landmarker = L.build_landmarker()
    except (FileNotFoundError, RuntimeError) as e:
        raise SystemExit(str(e))

    gen = frame_source(a)          # reuse stream/camera for the whole session
    done = []
    subject = a.subject
    try:
        while subject:
            run_subject(subject, a, cv2, mp, landmarker, gen)
            done.append(subject)
            if a.once:
                break
            subject = _ask_next_subject(done)
    finally:
        if a.show:
            cv2.destroyAllWindows()

    print(f"\n[calib] session end. subjects done ({len(done)}): "
          f"{', '.join(done) if done else '(none)'}")
    print(f"  cumulative summary: {os.path.join(a.outdir, 'calib_summary.csv')}")


if __name__ == "__main__":
    main()
