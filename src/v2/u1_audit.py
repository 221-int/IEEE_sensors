"""U1 (or any one subject) error audit for the frozen v2 CNN result.

This is deliberately a *post-hoc diagnostic*, not a new model-selection path.
It replays the saved CNN checkpoints on one subject's held-out events and joins
each event to the flags already present in ``index.npz``.  The result is meant
to answer a narrow question: are this subject's CNN errors concentrated in
missing / off-seat / interpolation / unusual-size events?

The final 15-run result predates persistence of ear-head checkpoints and raw
test scores.  Consequently this script does **not** manufacture an
``EAR-only correct`` category: it reports CNN-correct and CNN-error events
only, and records that limitation in its JSON output.

Example (run in the desktop Python environment with torch installed)::

    python -m src.v2.u1_audit

It writes ``results/v2/u1_audit.json`` and two small contact sheets under
``docs/v2/figures``.  It never edits the training result, folds, or data.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Any

import numpy as np

from src.v2.common import repro, splits
from src.v2.dataset import crop as C
from src.v2.train_encoder import Bundle, EVENT_LEN


DATA = "data/processed/v2"
MODELS = "models/v2/train_encoder_final"   # 2026-08-07: 체크포인트 경로가
# --out 에서 파생되도록 바뀌었다(train_encoder.models_dir). 확정 런의 --out 이
# results/v2/train_encoder_final.json 이므로 여기다.
TRAIN_RESULTS = "results/v2/train_encoder.json"
OUT = "results/v2/u1_audit.json"
FIG_DIR = "docs/v2/figures"


def _atomic_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_state(torch, path: str, device: str):
    """Use the safer PyTorch API when available, while supporting older torch."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # torch < 2.0
        return torch.load(path, map_location=device)


def _predict(bundle: Bundle, ids: np.ndarray, run_specs: list[dict[str, Any]],
             models_root: str, arch: str, latent: int, batch: int,
             device: str) -> tuple[np.ndarray, np.ndarray]:
    """Return per-seed logits and the frozen validation thresholds.

    ``ids`` indexes Bundle's valid-event view, whereas the audit JSON also
    retains original ``index.npz`` event indices for traceability.
    """
    import torch
    from src.v2.model import encoder as E

    logits, thresholds = [], []
    for spec in run_specs:
        fold, seed = int(spec["fold"]), int(spec["seed"])
        model_dir = os.path.join(models_root, f"fold{fold}_seed{seed}")
        enc_path = os.path.join(model_dir, "encoder.pt")
        head_path = os.path.join(model_dir, "head.pt")
        if not (os.path.isfile(enc_path) and os.path.isfile(head_path)):
            raise FileNotFoundError(
                f"saved CNN checkpoint missing for fold={fold}, seed={seed}: {model_dir}")

        front = E.build(arch, latent).to(device)
        head = E.build_head(latent, EVENT_LEN).to(device)
        front.load_state_dict(_load_state(torch, enc_path, device))
        head.load_state_dict(_load_state(torch, head_path, device))
        front.eval(); head.eval()

        out = []
        with torch.no_grad():
            for start in range(0, len(ids), batch):
                x, mask, _ = bundle.batch(ids[start:start + batch])
                b, t = x.shape[:2]
                z = front(torch.from_numpy(x.reshape(b * t, *x.shape[2:])).to(device))
                score = head(z.reshape(b, t, -1), torch.from_numpy(mask).to(device))
                out.append(score.detach().cpu().numpy().reshape(-1))
        logits.append(np.concatenate(out))
        thresholds.append(float(spec["ours"]["thr"]))
    return np.asarray(logits, dtype=np.float64), np.asarray(thresholds, dtype=np.float64)


def _finite_summary(x: np.ndarray) -> dict[str, float | None]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"median": None, "p10": None, "p90": None}
    return {"median": float(np.median(x)), "p10": float(np.quantile(x, .1)),
            "p90": float(np.quantile(x, .9))}


def _field(idx: dict[str, np.ndarray], name: str, default: float = np.nan) -> np.ndarray:
    """A missing diagnostic field must not make the audit silently fail."""
    if name in idx:
        return idx[name]
    n = len(idx["f_subject"]) if name.startswith("f_") else len(idx["e_subject"])
    return np.full(n, default)


def _event_metadata(idx: dict[str, np.ndarray], event_i: int, local_i: int,
                    seed_logits: np.ndarray, thresholds: np.ndarray) -> dict[str, Any]:
    rows = np.asarray(idx["e_rows"][event_i], dtype=np.int64)
    present = rows >= 0
    fr = rows[present]

    def f(name: str, default: float = np.nan) -> np.ndarray:
        return _field(idx, name, default)[fr]

    spans = f("f_span_px")
    brightness = f("f_brightness_m22")
    contrast = f("f_contrast_m22")
    sharpness = f("f_sharpness_m22")
    ears = f("f_ear")
    off = f("f_off_seat", 0).astype(bool)
    cubic = f("f_interp_cubic_m22", 0).astype(bool)
    padded = f("f_padded_m22", 0).astype(bool)

    score_seed = seed_logits[:, local_i]
    pred_seed = score_seed >= thresholds
    vote = bool(pred_seed.mean() >= .5)
    label = int(idx["e_is_blink"][event_i])
    return {
        "event_index": int(event_i),
        "event_id_within_user": int(_field(idx, "e_event_id", -1)[event_i]),
        "start_frame": int(_field(idx, "e_start", -1)[event_i]),
        "end_frame": int(_field(idx, "e_end", -1)[event_i]),
        "label_blink": label,
        "t_rel": float(idx["e_t_rel"][event_i]),
        "n_missing": int(idx["e_n_missing"][event_i]),
        "event_off_seat": bool(_field(idx, "e_off_seat", 0)[event_i]),
        "n_present_frames": int(present.sum()),
        "frame_off_seat_count": int(off.sum()),
        "frame_cubic_count": int(cubic.sum()),
        "frame_padded_count": int(padded.sum()),
        "span_px": _finite_summary(spans),
        "brightness": _finite_summary(brightness),
        "contrast": _finite_summary(contrast),
        "sharpness": _finite_summary(sharpness),
        "ear": _finite_summary(ears),
        "seed_logits": [float(v) for v in score_seed],
        "seed_predictions": [int(v) for v in pred_seed],
        "mean_logit": float(score_seed.mean()),
        "frozen_thresholds": [float(v) for v in thresholds],
        "ensemble_prediction": int(vote),
        "n_seed_disagree": int((pred_seed != label).sum()),
        "ours_correct": bool(vote == label),
        "category": "ours_correct" if vote == label else "ours_error",
    }


def _group_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(events)
    if not n:
        return {"n_events": 0}
    labels = np.array([e["label_blink"] for e in events])
    return {
        "n_events": n,
        "blink": int(labels.sum()),
        "unblink": int(n - labels.sum()),
        "event_off_seat_rate": float(np.mean([e["event_off_seat"] for e in events])),
        "mean_missing_frames": float(np.mean([e["n_missing"] for e in events])),
        "event_with_missing_rate": float(np.mean([e["n_missing"] > 0 for e in events])),
        "mean_frame_off_seat_count": float(np.mean([e["frame_off_seat_count"] for e in events])),
        "mean_cubic_frame_count": float(np.mean([e["frame_cubic_count"] for e in events])),
        "mean_padded_frame_count": float(np.mean([e["frame_padded_count"] for e in events])),
        "mean_seed_disagree": float(np.mean([e["n_seed_disagree"] for e in events])),
        "t_rel": _finite_summary(np.array([e["t_rel"] for e in events])),
        "span_median_px": _finite_summary(np.array([e["span_px"]["median"] for e in events])),
        "brightness_median": _finite_summary(np.array([e["brightness"]["median"] for e in events])),
        "sharpness_median": _finite_summary(np.array([e["sharpness"]["median"] for e in events])),
    }


def _contact_sheet(bundle: Bundle, idx: dict[str, np.ndarray], events: list[dict[str, Any]],
                   path: str, title: str) -> str | None:
    """Show t=0/9/18 for a deterministic, limited review set.

    The sheets are visual review aids only: selected errors are high-confidence
    seed-consistent errors, not a random sample or a new statistical analysis.
    """
    try:
        import cv2
    except ImportError:
        return None
    if not events:
        return None

    tile_h, tile_w, text_h = 64 * 2, 160 * 2, 34
    canvas = np.full((text_h + len(events) * (tile_h + text_h), tile_w * 3, 3), 245, np.uint8)
    cv2.putText(canvas, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 1, cv2.LINE_AA)
    for r, event in enumerate(events):
        y0 = text_h + r * (tile_h + text_h)
        label = "blink" if event["label_blink"] else "unblink"
        pred = "blink" if event["ensemble_prediction"] else "unblink"
        info = (f"e{event['event_index']}  y={label} pred={pred}  "
                f"off={int(event['event_off_seat'])} miss={event['n_missing']}  "
                f"seed_err={event['n_seed_disagree']}/3")
        cv2.putText(canvas, info, (3, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, .40,
                    (0, 0, 0), 1, cv2.LINE_AA)
        rows = idx["e_rows"][event["event_index"]]
        for col, at in enumerate((0, 9, 18)):
            src = int(rows[at])
            if src < 0:
                tile = np.full((tile_h, tile_w, 3), 32, np.uint8)
                cv2.putText(tile, "missing", (100, 65), cv2.FONT_HERSHEY_SIMPLEX, .5,
                            (220, 220, 220), 1, cv2.LINE_AA)
            else:
                im = np.asarray(bundle.frames[src])
                tile = cv2.cvtColor(cv2.resize(im, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST),
                                    cv2.COLOR_GRAY2BGR)
            cv2.putText(tile, f"t={at}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .45,
                        (0, 0, 255), 1, cv2.LINE_AA)
            canvas[y0 + text_h:y0 + text_h + tile_h, col * tile_w:(col + 1) * tile_w] = tile
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, canvas)
    return path.replace("\\", "/")


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay frozen CNN checkpoints for a one-subject audit.")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--models", default=MODELS)
    ap.add_argument("--train-results", default=TRAIN_RESULTS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--fig-dir", default=FIG_DIR)
    ap.add_argument("--user", type=int, default=1)
    ap.add_argument("--tag", default="m22")
    ap.add_argument("--arch", default="vpres")
    ap.add_argument("--latent", type=int, default=16)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-events-per-sheet", type=int, default=10)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    import torch

    run_result = _load_json(args.train_results)
    bundle = Bundle(args.data, args.tag)
    assign = splits.load_folds()
    if args.user not in assign:
        raise SystemExit(f"User {args.user} is not present in the frozen folds.")
    fold = int(assign[args.user])
    local_ids = np.flatnonzero(bundle.subject == args.user)
    if local_ids.size == 0:
        raise SystemExit(f"User {args.user} has no valid (e_valid) events.")

    # Final model output is deliberately tied to the frozen test fold.
    run_specs = sorted([r for r in run_result["runs"] if int(r["fold"]) == fold],
                       key=lambda r: int(r["seed"]))
    if not run_specs:
        raise SystemExit(f"No saved run for U{args.user}'s test fold {fold}.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"U{args.user}: fold {fold}, {len(local_ids)} valid events, replaying {len(run_specs)} saved CNN seeds on {device}.")
    seed_logits, thresholds = _predict(bundle, local_ids, run_specs, args.models, args.arch, args.latent,
                                       args.batch, device)

    idx_file = np.load(os.path.join(args.data, "index.npz"))
    idx = {k: idx_file[k] for k in idx_file.files}
    # Bundle retains exactly e_valid events in source-event order.
    source_valid_events = np.flatnonzero(idx["e_valid"].astype(bool))
    source_event_ids = source_valid_events[local_ids]
    events = [_event_metadata(idx, int(src_i), int(local_i), seed_logits, thresholds)
              for local_i, src_i in enumerate(source_event_ids)]
    correct = [e for e in events if e["ours_correct"]]
    errors = [e for e in events if not e["ours_correct"]]
    # Deterministic, high-confidence review subset.  Keep its selection rule in the output.
    errors_for_sheet = sorted(errors, key=lambda e: (-e["n_seed_disagree"],
                                                      -int(e["event_off_seat"]), e["event_index"]))
    correct_for_sheet = sorted(correct, key=lambda e: (-int(e["event_off_seat"]), e["event_index"]))
    n_sheet = max(0, args.max_events_per_sheet)
    figures: dict[str, str | None] = {}
    if not args.no_figures:
        prefix = os.path.join(args.fig_dir, f"u{args.user}_audit")
        figures["ours_error"] = _contact_sheet(bundle, idx, errors_for_sheet[:n_sheet],
                                                 prefix + "_ours_error.png",
                                                 f"U{args.user}: CNN errors (t=0,9,18)")
        figures["ours_correct_context"] = _contact_sheet(bundle, idx, correct_for_sheet[:n_sheet],
                                                           prefix + "_ours_correct_context.png",
                                                           f"U{args.user}: CNN-correct context (t=0,9,18)")

    face_context = None
    face_path = "results/v2/face_position.json"
    if os.path.isfile(face_path):
        raw_face = _load_json(face_path)
        for key in ("per_user", "users"):
            if isinstance(raw_face.get(key), dict) and str(args.user) in raw_face[key]:
                face_context = raw_face[key][str(args.user)]
                break

    by_label = {"blink": _group_summary([e for e in events if e["label_blink"]]),
                "unblink": _group_summary([e for e in events if not e["label_blink"]])}
    all_user_events = idx["e_subject"] == args.user
    excluded_before_replay = all_user_events & ~idx["e_valid"].astype(bool)
    missing_only = ~_field(idx, "e_valid_missing_only", 1).astype(bool)
    off_seat = _field(idx, "e_off_seat", 0).astype(bool)
    excluded_user = _field(idx, "e_excluded_user", 0).astype(bool)
    payload = {
        "purpose": "Post-hoc U1 root-cause audit; no retraining, threshold selection, or primary-result change.",
        "config": vars(args),
        "input_norm": C.INPUT_NORM,
        "replay": {
            "fold": fold,
            "seeds": [int(r["seed"]) for r in run_specs],
            "device": device,
            "checkpoint_format": "saved frozen CNN encoder.pt + head.pt",
            "frozen_validation_thresholds": [float(v) for v in thresholds],
        },
        "event_set": {
            "replayed": "e_valid events only: exactly the events used by the final training/evaluation protocol.",
            "n_user_events_total": int(all_user_events.sum()),
            "n_replayed": int(len(events)),
            "n_screened_out_before_replay": int(excluded_before_replay.sum()),
            "screened_out_reasons_not_exclusive": {
                "missing_policy": int((all_user_events & missing_only).sum()),
                "off_seat_policy": int((all_user_events & off_seat).sum()),
                "excluded_user_policy": int((all_user_events & excluded_user).sum()),
            },
            "note": "Because e_off_seat events were excluded before the final run, an e_off_seat rate of zero among replayed events is expected, not evidence that the raw session had no face-position issue.",
        },
        "limitations": [
            "The final run did not retain ear_head checkpoints or raw test-score sidecars. "
            "Therefore EAR-only / ours-only event categories are not reconstructed here.",
            "e_off_seat is an existing seat-position flag, not an event-level multi-face flag. "
            "Any face_position entry below is user-level context only.",
            "Contact sheets are deterministic visual review subsets, not random samples or new inferential tests.",
        ],
        "face_position_user_context": face_context,
        "summary": {
            "all": _group_summary(events),
            "ours_correct": _group_summary(correct),
            "ours_error": _group_summary(errors),
            "by_label": by_label,
            "error_label_counts": dict(Counter("blink" if e["label_blink"] else "unblink" for e in errors)),
        },
        "contact_sheet_selection": {
            "error_order": "most seed-consistent errors, then e_off_seat, then source event index",
            "correct_order": "e_off_seat first, then source event index",
            "max_events_per_sheet": n_sheet,
        },
        "figures": figures,
        "events": events,
        "env": repro.env_fingerprint(),
    }
    _atomic_json(args.out, payload)
    print(f"  errors {len(errors)}/{len(events)}; wrote {args.out}")
    for name, path in figures.items():
        print(f"  {name}: {path or 'not written (cv2 unavailable or empty group)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
