"""
eye_preprocess.py

CANONICAL eye-crop preprocessing. This is the SINGLE SOURCE OF TRUTH for how an
eye is cropped, everywhere: webcam capture, dataset preparation (Eyeblink8),
encoder training, and Pi/serve-time inference. Import from here; never re-implement.

Spec (locked):
    - one eye
    - grayscale
    - 64 x 64
    - rotation-aligned so the eye-corner line is horizontal (kept upright)
    - square crop with side = CROP_SCALE * eye_width  (margin around the eye)

Dependency note: this module imports ONLY cv2 + numpy on purpose. MediaPipe is
NOT imported here so that an ONNX-only deployment (e.g. the Pi) can reuse the
exact crop geometry without pulling in mediapipe. The landmark helpers below
take an already-detected landmark sequence; the detector lives in
src/deploy/frontend.py.
"""

import cv2
import numpy as np

OUT_SIZE = 64            # output 64x64 (legacy one-eye crop)
OUT_H, OUT_W = 64, 160   # both-eyes crop output (H x W) -- current canonical
CROP_SCALE = 1.8         # square side = CROP_SCALE * eye_width
BOTH_MARGIN = 2.2        # both-eyes crop width = BOTH_MARGIN * inter-eye distance
GRAYSCALE = True
ALIGN_ROTATION = True    # rotate so the eye-corner line is horizontal

# MediaPipe FaceMesh eye-corner landmark indices (468/478-point mesh).
# Each pair = the two corners of ONE eye. Which pair falls on the image-LEFT
# depends on whether the frame was mirrored, so we do NOT hardcode it -- the
# helper below orders the pairs by their actual x coordinate at runtime.
MP_EYE_PAIR_A = (33, 133)     # one eye: outer / inner corner
MP_EYE_PAIR_B = (362, 263)    # the other eye: inner / outer corner


def crop_eye_from_corners(frame_bgr, corner_a, corner_b):
    """Crop one eye given its two corner points (pixel coords).

    corner_a, corner_b: (x, y) pixel coordinates of the two eye corners
    (order does not matter; rotation is normalized to keep the eye upright).
    Returns a 64x64 uint8 image (grayscale if GRAYSCALE), or None on bad input.
    """
    h, w = frame_bgr.shape[:2]
    p_a = np.asarray(corner_a, dtype=np.float32)
    p_b = np.asarray(corner_b, dtype=np.float32)
    center = (p_a + p_b) / 2.0
    eye_w = float(np.linalg.norm(p_a - p_b))
    if eye_w < 3:  # too small / bad detection
        return None
    side = eye_w * CROP_SCALE

    img = frame_bgr
    if ALIGN_ROTATION:
        d = p_b - p_a
        angle = float(np.degrees(np.arctan2(d[1], d[0])))
        # Normalize to a small tilt so the eye stays UPRIGHT (avoid 180 flip).
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
        M = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle, 1.0)
        img = cv2.warpAffine(frame_bgr, M, (w, h), flags=cv2.INTER_LINEAR)

    x0 = int(round(center[0] - side / 2))
    y0 = int(round(center[1] - side / 2))
    x1 = int(round(center[0] + side / 2))
    y1 = int(round(center[1] + side / 2))

    pad = max(0, -x0, -y0, x1 - w, y1 - h)
    if pad > 0:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        x0 += pad; y0 += pad; x1 += pad; y1 += pad

    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    if GRAYSCALE:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    interp = cv2.INTER_AREA if crop.shape[0] >= OUT_SIZE else cv2.INTER_CUBIC
    crop = cv2.resize(crop, (OUT_SIZE, OUT_SIZE), interpolation=interp)
    return crop


def crop_both_eyes_from_corners(frame_bgr, le_a, le_b, re_a, re_b,
                                out_h=OUT_H, out_w=OUT_W):
    """CANONICAL both-eyes crop -> out_h x out_w (grayscale) uint8, or None.

    le_a, le_b : the two corners of the (image-)left eye
    re_a, re_b : the two corners of the (image-)right eye
    Centered on both eyes, rotation-aligned by the line joining the eye centers,
    aspect preserved (crop box has the target out_w:out_h ratio).

    Fast path: when the crop box lies fully inside the frame, only the box is
    warped instead of the whole frame (~10x faster; the crop is the dominant
    per-frame cost on the Pi after MediaPipe).

    NOT byte-identical to the pre-optimization full-frame warp. Folding the box
    origin into the matrix moves where OpenCV's fixed-point rounding happens, so
    interpolation weights can differ by 1 LSB. Measured on 2,400 frames across
    all 8 Eyeblink8 clips:

        crop pixels   mean |d| 0.0042/255, max 5
        blink_prob    mean |d| 3.1e-4, max 7.3e-3
        decision @0.9 1 frame flipped out of 2,400 (0.042%)

    For scale, the MediaPipe-vs-manual-annotation crop difference that already
    exists at inference time is MAE 6.7/255 -- ~1600x larger. Blink scoring is
    event-level (an event counts as detected if ANY of its frames fires), so a
    single frame flip does not move the reported metric.

    If bit-exactness is ever required (e.g. regenerating the training npz),
    the slow path below is the reference; keep dsize=(w, h) and do not fold the
    origin into M.
    """
    h, w = frame_bgr.shape[:2]
    le = (np.asarray(le_a, np.float32) + np.asarray(le_b, np.float32)) / 2.0
    re = (np.asarray(re_a, np.float32) + np.asarray(re_b, np.float32)) / 2.0
    center = (le + re) / 2.0
    span = float(np.linalg.norm(le - re))  # inter-eye-center distance
    if span < 5:
        return None
    crop_w = span * BOTH_MARGIN
    crop_h = crop_w * out_h / out_w        # keep target aspect

    x0 = int(round(center[0] - crop_w / 2)); y0 = int(round(center[1] - crop_h / 2))
    x1 = int(round(center[0] + crop_w / 2)); y1 = int(round(center[1] + crop_h / 2))
    pad = max(0, -x0, -y0, x1 - w, y1 - h)

    M = None
    if ALIGN_ROTATION:
        d = re - le
        angle = float(np.degrees(np.arctan2(d[1], d[0])))
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
        M = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle, 1.0)

    if M is not None and pad == 0:
        # ---- FAST PATH: warp ONLY the crop box ----
        # Fold the box origin into the matrix and set dsize to the box, so the
        # warp produces exactly frame_warped[y0:y1, x0:x1] and nothing else.
        # ~60x fewer destination pixels than warping the full frame.
        M[0, 2] -= x0
        M[1, 2] -= y0
        crop = cv2.warpAffine(frame_bgr, M, (x1 - x0, y1 - y0),
                              flags=cv2.INTER_LINEAR)
    else:
        # ---- SLOW PATH: crop box runs off the frame (or rotation disabled) ----
        # Kept because the out-of-frame margin is BORDER_REFLECT of the WARPED
        # image, which the fast path cannot reproduce (it would give the warp's
        # own constant border instead).
        img = frame_bgr
        if M is not None:
            img = cv2.warpAffine(frame_bgr, M, (w, h), flags=cv2.INTER_LINEAR)
        if pad > 0:
            img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
            x0 += pad; y0 += pad; x1 += pad; y1 += pad
        crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    if GRAYSCALE:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    interp = cv2.INTER_AREA if crop.shape[1] >= out_w else cv2.INTER_CUBIC
    crop = cv2.resize(crop, (out_w, out_h), interpolation=interp)  # cv2 size = (W, H)
    return crop


# --------------------------- landmark adapter ---------------------------
# Bridges a detector (MediaPipe FaceMesh) to the canonical crop above. The
# dataset path feeds corners from the Eyeblink8 .tag file; the inference path
# feeds corners from these helpers. Both then call the SAME crop function, so
# train/serve geometry cannot drift.

def _landmark_xy(landmarks, idx, w, h):
    """One landmark -> (x, y) in pixels.

    `landmarks` may be a MediaPipe NormalizedLandmarkList.landmark sequence
    (objects with .x/.y) or any indexable sequence of (x, y[, z]).
    Coordinates are assumed NORMALIZED to [0, 1] (MediaPipe convention).
    """
    lm = landmarks[idx]
    x = getattr(lm, "x", None)
    if x is None:
        x, y = float(lm[0]), float(lm[1])
    else:
        x, y = float(x), float(lm.y)
    return np.array([x * w, y * h], dtype=np.float32)


def eye_corners_from_landmarks(landmarks, w, h,
                               pair_a=MP_EYE_PAIR_A, pair_b=MP_EYE_PAIR_B):
    """Landmarks -> (le_a, le_b, re_a, re_b), the 4 eye corners in pixels.

    `le_*` is always the IMAGE-LEFT eye and `re_*` the image-right one: the two
    pairs are ordered by their mean x at runtime. This makes the result
    independent of frame mirroring and of MediaPipe's own left/right naming.

    (crop_both_eyes_from_corners is in fact symmetric under swapping the two
    pairs -- centre and span are both symmetric, and the alignment angle is
    normalized to +/-90 deg -- so this ordering is for callers that care, e.g.
    per-eye asymmetry features. The crop itself is safe either way.)
    """
    a1 = _landmark_xy(landmarks, pair_a[0], w, h)
    a2 = _landmark_xy(landmarks, pair_a[1], w, h)
    b1 = _landmark_xy(landmarks, pair_b[0], w, h)
    b2 = _landmark_xy(landmarks, pair_b[1], w, h)
    if (a1[0] + a2[0]) <= (b1[0] + b2[0]):
        return a1, a2, b1, b2
    return b1, b2, a1, a2


def crop_both_eyes_from_landmarks(frame_bgr, landmarks,
                                  out_h=OUT_H, out_w=OUT_W):
    """CANONICAL inference-time crop: frame + landmarks -> out_h x out_w uint8.

    Same output as crop_both_eyes_from_corners on the dataset path.
    Returns None if the eyes are too close together / the crop is degenerate.
    """
    h, w = frame_bgr.shape[:2]
    le_a, le_b, re_a, re_b = eye_corners_from_landmarks(landmarks, w, h)
    return crop_both_eyes_from_corners(frame_bgr, le_a, le_b, re_a, re_b,
                                       out_h=out_h, out_w=out_w)


def to_input_tensor(crop):
    """(H, W) uint8 crop -> (1, 1, H, W) float32 in [0, 1] for the ONNX graph.

    Must match training EXACTLY. Training does
        torch.from_numpy(images).float().div_(255.0).unsqueeze(1)
    (src/encoder/train_autoencoder.py, src/classifier/train_classifier.py),
    i.e. plain /255 with no mean/std normalization.
    """
    x = np.asarray(crop, dtype=np.float32) / 255.0
    if x.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale crop, got shape {x.shape}")
    return x.reshape(1, 1, x.shape[0], x.shape[1])
