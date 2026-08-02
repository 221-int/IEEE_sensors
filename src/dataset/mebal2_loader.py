"""
mebal2_loader.py

mEBAL2 -> canonical both-eyes crops. Parsing + face resolution only; the crop
geometry itself stays in eye_preprocess.py (single source of truth), which this
module calls with coords="pixel".

Everything below was measured on the distributed archives, not read off the
official docs (see docs/mEBAL2_실측_2026-07-31.md). The official description
disagrees with the data in three places and the data wins:

    official                        actual (measured)
    ------------------------------  ----------------------------------------
    68 facial landmarks             MediaPipe FaceMesh 468 per face
    box = xywh                      box = xyxy
    (unspecified) coords            absolute PIXELS, not normalized [0, 1]

Other facts this module encodes:

  * All CSVs are WHITESPACE separated, not comma. pd.read_csv with defaults
    silently yields a single column.
  * landmarks.csv stores a FLAT list of [x, y, z]; faces are not nested. Split
    it into chunks of 468.
  * Frame numbering: landmarks.csv, box.csv, blink/unblink CSVs and the video
    are all 0-based. Intel_Time/Time.csv is 1-BASED -- mixing them shifts
    everything by one frame. Use time_row_for_frame().
  * Events are fixed 19-frame windows (End - Start == 18, no exceptions in
    15,687 events) centred on the blink. There is no duration label.
  * box.csv and landmarks.csv DISAGREE on the face count in 6.0% of frames
    (User 1: 2,154 frames with a box but no mesh, 104 with 2 boxes and 1 mesh).
    So "assert n_landmark_faces == n_box_faces" is not a valid invariant --
    the two are reconciled geometrically instead (mesh centroid inside box).

Face selection (Phase 0.3): among faces that HAVE a mesh, take the one with the
largest box area; a mesh with no matching box falls back to its own bounding
box area. Verified against 5 hand-checked User 1 frames. The rule must not
wobble frame to frame or the crop jumps.

Dependencies: numpy + cv2 only. MediaPipe is deliberately not imported (same
reason as eye_preprocess: the Pi/ONNX path must be able to reuse this geometry).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import cv2
import numpy as np

from src.dataset.eye_preprocess import (
    COORDS_PIXEL,
    EAR_EYE_A,
    EAR_EYE_B,
    crop_both_eyes_from_landmarks,
    ear_from_indices_xy,
    eye_corners_from_landmarks,
)

N_MESH = 468          # MediaPipe FaceMesh points per face
EVENT_LEN = 19        # every mEBAL2 event is exactly 19 frames
FPS = 30.0            # confirmed on User 1 (37,641 frames / 1254.7 s)

# Frame numbering base per file, for the record. 0 = 0-based, 1 = 1-based.
FRAME_BASE = {"video": 0, "landmarks": 0, "box": 0,
              "blink": 0, "unblink": 0, "time": 1}


def time_row_for_frame(frame_idx: int) -> int:
    """0-based video/landmark frame -> 0-based ROW index in Intel_Time/Time.csv.

    Time.csv's own Frame column is 1-based, so row i holds Frame i+1, i.e. the
    row for 0-based frame f is simply f. This function exists so the off-by-one
    lives in exactly one place and is greppable.
    """
    return frame_idx


# --------------------------- CSV parsing ---------------------------

def _parse_bracket_rows(path, item_len, name):
    """'<frame> [[a,b,..],[a,b,..]]' whitespace rows -> dict[int, (n, item_len)].

    An empty list '[]' yields a (0, item_len) array (detection failure).
    """
    out = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline()
        if "[" in header:  # no header line -- rewind and treat as data
            f.seek(0)
        for lineno, line in enumerate(f, start=2):
            line = line.strip()
            if not line:
                continue
            try:
                frame_s, rest = line.split(" ", 1)
                frame = int(frame_s)
            except ValueError as e:
                raise ValueError(f"{path}:{lineno}: bad row start: {line[:60]!r}") from e
            rest = rest.strip()
            if rest in ("[]", ""):
                out[frame] = np.zeros((0, item_len), np.float32)
                continue
            flat = np.fromstring(rest.replace("[", " ").replace("]", " "),
                                 sep=",", dtype=np.float32)
            if flat.size % item_len:
                raise ValueError(
                    f"{path}:{lineno}: {name} count {flat.size} is not a multiple "
                    f"of {item_len}")
            out[frame] = flat.reshape(-1, item_len)
    return out


def load_landmarks(path):
    """landmarks.csv -> dict[frame] = (n_faces, 468, 3) float32, PIXEL coords."""
    raw = _parse_bracket_rows(path, 3, "landmark")
    out = {}
    for frame, a in raw.items():
        if a.shape[0] % N_MESH:
            raise ValueError(f"{path}: frame {frame} has {a.shape[0]} landmarks, "
                             f"not a multiple of {N_MESH}")
        out[frame] = a.reshape(-1, N_MESH, 3)
    return out


def load_boxes(path):
    """box.csv -> dict[frame] = (n_faces, 4) float32 in xyxy."""
    boxes = _parse_bracket_rows(path, 4, "box coordinate")
    for frame, b in boxes.items():
        if b.size and not np.all((b[:, 2] >= b[:, 0]) & (b[:, 3] >= b[:, 1])):
            raise ValueError(
                f"{path}: frame {frame} box is not xyxy (x1<x0 or y1<y0): {b}")
    return boxes


def load_events(path, has_flag):
    """Blink/Unblink CSV -> (n, 3) int array [start, end, confirmed].

    `confirmed` is the Blink column for Right_Blink.csv (1 = blink, 0 = possible
    blink) and is set to 1 for Unblink.csv, which has no such column.
    Rows whose window is not EVENT_LEN frames are returned as-is; the caller
    decides. (None were found in 15,687 events, but do not assume.)
    """
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, start=1):
            parts = line.split()
            if len(parts) < 2 or not parts[0].lstrip("-").isdigit():
                continue  # header or blank
            start, end = int(parts[0]), int(parts[1])
            flag = int(parts[2]) if (has_flag and len(parts) > 2) else 1
            rows.append((start, end, flag))
    return np.array(rows, dtype=np.int64).reshape(-1, 3)


_USER_RE = re.compile(r"User[ _]?(\d+)")


def probe_event_paths(probe_dir, user):
    """_probe/ flat layout -> (blink_csv, unblink_csv) for a user number."""
    b = os.path.join(probe_dir, f"EyeBlinks_User {user}_Blink_Right_Blink.csv")
    u = os.path.join(probe_dir, f"EyeUnblinks_User {user}_Unblink_Unblink.csv")
    return b, u


# --------------------------- face resolution ---------------------------

def box_areas(boxes):
    if boxes.size == 0:
        return np.zeros((0,), np.float32)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def match_landmarks_to_boxes(faces, boxes):
    """-> int array (n_faces,) mapping each mesh to a box index, -1 if none.

    Matched by mesh centroid containment; falls back to the nearest box centre
    when no box contains the centroid. Index-order matching is NOT assumed --
    it happens to hold on User 1 (400/400 checks on 2-face frames) but the face
    counts disagree on 6% of frames, so order cannot be relied on in general.
    """
    n = faces.shape[0]
    out = np.full(n, -1, np.int64)
    if n == 0 or boxes.shape[0] == 0:
        return out
    centres = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2.0,
                        (boxes[:, 1] + boxes[:, 3]) / 2.0], axis=1)
    used = set()
    for i in range(n):
        c = faces[i, :, :2].mean(axis=0)
        inside = [j for j in range(boxes.shape[0])
                  if j not in used
                  and boxes[j, 0] <= c[0] <= boxes[j, 2]
                  and boxes[j, 1] <= c[1] <= boxes[j, 3]]
        if inside:
            j = min(inside, key=lambda k: float(np.linalg.norm(centres[k] - c)))
        else:
            free = [k for k in range(boxes.shape[0]) if k not in used]
            if not free:
                continue
            j = min(free, key=lambda k: float(np.linalg.norm(centres[k] - c)))
        out[i] = j
        used.add(j)
    return out


# Reasons a frame yields no usable face -- counted so Phase 0.4 can report them.
NO_MESH = "no_mesh"          # landmarks.csv empty for this frame
NO_ROW = "no_row"            # frame absent from landmarks.csv entirely
OK = "ok"


@dataclass
class FaceSelection:
    frame: int
    status: str
    landmarks: np.ndarray | None = None   # (468, 3) pixel coords
    box: np.ndarray | None = None         # (4,) xyxy, or None if unmatched
    n_mesh: int = 0
    n_box: int = 0


def select_face(frame, faces, boxes):
    """Pick ONE face for a frame. Rule: largest matched box area.

    A mesh with no matching box uses its own bounding-box area, so a frame is
    never discarded merely because box.csv is missing an entry.
    """
    if faces is None:
        return FaceSelection(frame, NO_ROW)
    n_mesh, n_box = faces.shape[0], (0 if boxes is None else boxes.shape[0])
    if n_mesh == 0:
        return FaceSelection(frame, NO_MESH, n_mesh=0, n_box=n_box)
    if boxes is None:
        boxes = np.zeros((0, 4), np.float32)
    m = match_landmarks_to_boxes(faces, boxes)
    areas = box_areas(boxes)
    scores = np.empty(n_mesh, np.float64)
    for i in range(n_mesh):
        if m[i] >= 0:
            scores[i] = areas[m[i]]
        else:
            xy = faces[i, :, :2]
            scores[i] = float((xy[:, 0].max() - xy[:, 0].min())
                              * (xy[:, 1].max() - xy[:, 1].min()))
    i = int(np.argmax(scores))
    return FaceSelection(frame, OK, faces[i],
                         None if m[i] < 0 else boxes[m[i]],
                         n_mesh=n_mesh, n_box=n_box)


# --------------------------- video ---------------------------

def read_frames(video_path, frame_indices):
    """Yield (frame_idx, bgr) for the requested 0-based indices, in order.

    Sequential grab()/retrieve() rather than CAP_PROP_POS_FRAMES: seeking in
    these h264 files lands on the nearest keyframe and silently returns the
    wrong frame, which would misalign every crop.
    """
    want = sorted(set(int(i) for i in frame_indices))
    if not want:
        return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"cannot open {video_path}")
    try:
        pos, k = 0, 0
        while k < len(want):
            target = want[k]
            while pos < target:
                if not cap.grab():
                    return
                pos += 1
            ok, frame = cap.read()
            if not ok:
                return
            yield target, frame
            pos += 1
            k += 1
    finally:
        cap.release()


# --------------------------- user ---------------------------

class MEBAL2User:
    """One mEBAL2 subject = one folder = one session.

    Paths are passed explicitly because the probe extraction and a full
    Processed_Data extraction have different layouts.
    """

    def __init__(self, user, landmarks_csv, box_csv,
                 blink_csv=None, unblink_csv=None, video=None):
        self.user = int(user)
        self.subject = f"m{int(user):02d}"
        self.landmarks_csv = landmarks_csv
        self.box_csv = box_csv
        self.blink_csv = blink_csv
        self.unblink_csv = unblink_csv
        self.video = video
        self._lm = None
        self._box = None

    @classmethod
    def from_probe(cls, probe_dir, user, landmarks_csv=None, box_csv=None,
                   video=None):
        b, u = probe_event_paths(probe_dir, user)
        return cls(user,
                   landmarks_csv or os.path.join(probe_dir, "PD_landmarks.csv"),
                   box_csv or os.path.join(probe_dir, "PD_box.csv"),
                   b, u, video)

    @classmethod
    def from_processed_dir(cls, processed_root, raw_root, user):
        d = os.path.join(processed_root, f"User {user}")
        return cls(user,
                   os.path.join(d, "landmarks.csv"),
                   os.path.join(d, "box.csv"),
                   video=os.path.join(raw_root, f"User {user}", "RealSense",
                                      "Color_Webcam", "color.mp4"))

    # -- lazy tables --
    @property
    def landmarks(self):
        if self._lm is None:
            self._lm = load_landmarks(self.landmarks_csv)
        return self._lm

    @property
    def boxes(self):
        if self._box is None:
            self._box = load_boxes(self.box_csv)
        return self._box

    def events(self, include_unconfirmed=False):
        """-> (events, is_blink) with events = (n, 3) [start, end, confirmed].

        `Blink=0` ("possible blink") is EXCLUDED by default: the distributors
        balanced the set 1:1 against Unblink using Blink=1 only, and that holds
        for 58/58 users. Include it only for a sensitivity analysis.
        """
        ev, isb = [], []
        if self.blink_csv and os.path.exists(self.blink_csv):
            b = load_events(self.blink_csv, has_flag=True)
            if not include_unconfirmed:
                b = b[b[:, 2] == 1]
            ev.append(b); isb.append(np.ones(len(b), np.int64))
        if self.unblink_csv and os.path.exists(self.unblink_csv):
            u = load_events(self.unblink_csv, has_flag=False)
            ev.append(u); isb.append(np.zeros(len(u), np.int64))
        if not ev:
            return np.zeros((0, 3), np.int64), np.zeros((0,), np.int64)
        return np.concatenate(ev), np.concatenate(isb)

    # -- per-frame --
    def face(self, frame):
        return select_face(frame,
                           self.landmarks.get(frame),
                           self.boxes.get(frame))

    def eye_corners(self, frame):
        """-> (le_a, le_b, re_a, re_b) in pixels, or None if no face."""
        sel = self.face(frame)
        if sel.status != OK:
            return None
        return eye_corners_from_landmarks(sel.landmarks, 0, 0,
                                          coords=COORDS_PIXEL)

    def crop(self, frame_bgr, frame):
        """Canonical 64x160 both-eyes crop for a frame, or None."""
        sel = self.face(frame)
        if sel.status != OK:
            return None
        return crop_both_eyes_from_landmarks(frame_bgr, sel.landmarks,
                                             coords=COORDS_PIXEL)

    def ear(self, frame, variant="mean"):
        """EAR from the provided mesh (pixel coords), or None if no face."""
        sel = self.face(frame)
        if sel.status != OK:
            return None
        xy = sel.landmarks[:, :2]
        a = ear_from_indices_xy(xy, EAR_EYE_A)
        b = ear_from_indices_xy(xy, EAR_EYE_B)
        xa = (xy[EAR_EYE_A["corner_out"], 0] + xy[EAR_EYE_A["corner_in"], 0]) / 2
        xb = (xy[EAR_EYE_B["corner_out"], 0] + xy[EAR_EYE_B["corner_in"], 0]) / 2
        left, right = (a, b) if xa <= xb else (b, a)
        return {"left": left, "right": right,
                "mean": (left + right) / 2.0, "min": min(left, right)}[variant]

    # -- events --
    def event_frames(self, start, end):
        return list(range(int(start), int(end) + 1))

    def frame_ok(self, frame):
        return self.face(frame).status == OK

    def event_missing(self, start, end):
        """How many of an event's frames have no usable face."""
        return sum(0 if self.frame_ok(f) else 1
                   for f in self.event_frames(start, end))

    def coverage(self):
        """-> dict of per-frame face-resolution stats for the whole session."""
        n_ok = n_no_mesh = n_two = 0
        for frame in self.landmarks:
            sel = self.face(frame)
            if sel.status == OK:
                n_ok += 1
                if sel.n_mesh > 1:
                    n_two += 1
            else:
                n_no_mesh += 1
        total = len(self.landmarks)
        return {"frames": total, "ok": n_ok, "no_face": n_no_mesh,
                "multi_face": n_two,
                "no_face_rate": n_no_mesh / total if total else 0.0}
