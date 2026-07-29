"""
frontend.py

Inference-time FRONT END: raw frame -> face/eye detection -> canonical both-eyes
crop -> ONNX-ready input tensor.

This is the missing link between MediaPipe and the encoder. The dataset path
gets its 4 eye corners from the Eyeblink8 .tag file; this module gets the same
4 corners from MediaPipe FaceMesh. Both then call the SAME canonical crop in
src/dataset/eye_preprocess.py, so train/serve geometry cannot drift.

MediaPipe API choice
--------------------
Uses the legacy `mediapipe.solutions.face_mesh` API (the same one
src/dataset/capture_eye_dataset.py and src/eval/ear_baseline.py already use),
NOT the Tasks API. Reasons: no external face_landmarker.task file to ship
(it is .gitignore'd), and it is the only MediaPipe path still live in this repo.

Perf note for the Pi5 benchmark
-------------------------------
`refine_landmarks=True` adds the iris sub-model (468 -> 478 points). We do NOT
use iris points for the crop, and it costs extra CPU per frame. It defaults to
True here only to stay bit-identical with the existing EAR evaluation path; set
refine_landmarks=False for the latency runs and report which was used.

Usage:
    from src.deploy.frontend import EyeFrontend

    fe = EyeFrontend(refine_landmarks=False)
    x = fe.input_tensor(frame_bgr)          # (1, 1, 64, 160) float32, or None
    if x is not None:
        v = sess.run(None, {"input": x})[0]  # (1, 128)
"""

import numpy as np

from src.dataset.eye_preprocess import (
    OUT_H,
    OUT_W,
    crop_both_eyes_from_corners,
    eye_corners_from_landmarks,
    to_input_tensor,
)

try:  # cv2 is needed only for the BGR->RGB conversion MediaPipe expects
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

# Some mediapipe installs (esp. Windows / Python 3.12) don't expose the lazy
# `mp.solutions` attribute. Resolve the face_mesh module robustly, and keep the
# import soft so that `import src.deploy.frontend` doesn't hard-fail in an
# ONNX-only environment.
_face_mesh_mod = None
_MP_IMPORT_ERROR = None
try:
    import mediapipe as mp

    try:
        _face_mesh_mod = mp.solutions.face_mesh
    except AttributeError:
        from mediapipe.python.solutions import face_mesh as _face_mesh_mod
except Exception as exc:  # pragma: no cover - environment dependent
    _MP_IMPORT_ERROR = exc


class EyeFrontend:
    """Frame -> both-eyes crop -> encoder input, with MediaPipe FaceMesh.

    Every method returns None when no face is detected or the crop is
    degenerate, so callers can count detection misses explicitly rather than
    having them silently disappear.
    """

    def __init__(self, max_num_faces=1, refine_landmarks=True,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5,
                 static_image_mode=False, out_h=OUT_H, out_w=OUT_W):
        if _face_mesh_mod is None:
            raise ImportError(
                "mediapipe is required for EyeFrontend but could not be "
                f"imported: {_MP_IMPORT_ERROR}"
            )
        if cv2 is None:
            raise ImportError("opencv (cv2) is required for EyeFrontend")

        self.out_h = out_h
        self.out_w = out_w
        self.face_mesh = _face_mesh_mod.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        # Last successful corners. Kept so a future "detect every N frames"
        # ablation (roadmap: ROI reuse) can fall back to them without changing
        # this class's interface.
        self.last_corners = None

    # ---------------------------- stages ----------------------------
    def landmarks(self, frame_bgr):
        """Frame -> MediaPipe landmark sequence, or None if no face."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None
        return res.multi_face_landmarks[0].landmark

    def corners(self, frame_bgr):
        """Frame -> (le_a, le_b, re_a, re_b) in pixels, or None.

        le_* is the image-LEFT eye (resolved by x at runtime, so mirroring of
        the input frame does not matter).
        """
        lm = self.landmarks(frame_bgr)
        if lm is None:
            return None
        h, w = frame_bgr.shape[:2]
        c = eye_corners_from_landmarks(lm, w, h)
        self.last_corners = c
        return c

    def crop_from_corners(self, frame_bgr, corners):
        """Corners -> canonical crop. Separated so the benchmark can time the
        detection stage and the crop stage independently (§5 wants them as
        separate rows in the latency table)."""
        le_a, le_b, re_a, re_b = corners
        return crop_both_eyes_from_corners(frame_bgr, le_a, le_b, re_a, re_b,
                                           out_h=self.out_h, out_w=self.out_w)

    def crop(self, frame_bgr):
        """Frame -> (out_h, out_w) uint8 grayscale crop, or None."""
        c = self.corners(frame_bgr)
        if c is None:
            return None
        return self.crop_from_corners(frame_bgr, c)

    def input_tensor(self, frame_bgr):
        """Frame -> (1, 1, out_h, out_w) float32 in [0, 1], or None.

        Feed straight to the ONNX graph: sess.run(None, {"input": x}).
        """
        crop = self.crop(frame_bgr)
        if crop is None:
            return None
        return to_input_tensor(crop)

    # ---------------------------- lifecycle ----------------------------
    def close(self):
        if getattr(self, "face_mesh", None) is not None:
            self.face_mesh.close()
            self.face_mesh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _self_test():
    """Shape/dtype smoke test that does not need a camera or a real face.

    Runs the crop + tensor path on a synthetic frame with hand-placed corners,
    which exercises everything except MediaPipe itself.
    """
    frame = np.random.randint(0, 255, (480, 640, 3), np.uint8)
    le_a, le_b = (250.0, 240.0), (290.0, 240.0)
    re_a, re_b = (350.0, 240.0), (390.0, 240.0)
    crop = crop_both_eyes_from_corners(frame, le_a, le_b, re_a, re_b)
    assert crop is not None and crop.shape == (OUT_H, OUT_W), crop.shape
    assert crop.dtype == np.uint8
    x = to_input_tensor(crop)
    assert x.shape == (1, 1, OUT_H, OUT_W) and x.dtype == np.float32, x.shape
    assert 0.0 <= float(x.min()) and float(x.max()) <= 1.0
    print(f"OK  crop {crop.shape} {crop.dtype} -> tensor {x.shape} {x.dtype} "
          f"range [{x.min():.3f}, {x.max():.3f}]")
    print(f"mediapipe available: {_face_mesh_mod is not None}"
          + ("" if _face_mesh_mod is not None else f"  ({_MP_IMPORT_ERROR})"))


if __name__ == "__main__":
    _self_test()
