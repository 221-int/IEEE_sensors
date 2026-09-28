"""v2 배포 프론트엔드 — 프레임 -> 메시 -> 크롭 -> 입력 텐서.

🔴 v1 (`src/deploy/frontend.py`) 을 **import 하지 않는다. 규격만 가져왔다.**
   v1 의 `to_input_tensor` 는 `/255` 이고 v2 는 `frame_standardize` 다. v1 것을 쓰면
   **지연은 정상으로 나오고 판정만 조용히 망가진다.** 그래서 이 모듈은
   `src/v2/dataset/crop.py` 하나만 참조하고, 기하·정규화를 **직접 구현하지 않는다.**
   전부 위임한다 — 그래야 학습 경로와 갈라질 수가 없다.

메시 공급자를 분리한 이유
------------------------
학습은 mEBAL2 가 제공한 468점 메시를 썼고, Pi 는 MediaPipe FaceMesh 로 얻는다.
**기하 코드는 같지만 랜드마크 출처가 다르다.** 둘을 한 클래스에 묶어 두면
"동치 검증"이 랜드마크 차이까지 섞어서 재게 되어 무엇이 틀렸는지 알 수 없다.

  `crop_from_mesh(frame, mesh_xy)`  메시를 받는다 — **동치 검증은 이 경로로** 한다
                                    (MediaPipe 없이 돌아간다)
  `process(frame)`                  MediaPipe 로 메시를 얻어 위를 부른다 — Pi 실행 경로

→ 동치 검증은 **코드 경로**가 같은지를 재고, 랜드마크 출처 차이는 별개 항목으로
   남긴다(필드에서는 크롭이 비트 동일할 수 없다. 그건 한계이지 버그가 아니다).

v1 에서 가져온 규격
------------------
- `refine_landmarks=False` 로 잰다. iris 서브모델(468->478)은 크롭에 쓰지 않는데
  프레임당 CPU 를 더 먹는다. **무엇을 썼는지 결과에 기록한다.**
- 검출 실패는 None 으로 돌려 **호출자가 세게** 한다. 조용히 사라지면 안 된다.
"""

from __future__ import annotations

import numpy as np

from src.v2.dataset import crop as C

try:
    import cv2
except ImportError:                                   # pragma: no cover
    cv2 = None

_face_mesh_mod = None
_MP_ERR = None
try:
    import mediapipe as mp
    try:
        _face_mesh_mod = mp.solutions.face_mesh
    except AttributeError:
        from mediapipe.python.solutions import face_mesh as _face_mesh_mod
except Exception as exc:                              # pragma: no cover
    _MP_ERR = exc


def mediapipe_available() -> bool:
    return _face_mesh_mod is not None


class EyeFrontend:
    """프레임 -> 양눈 크롭 -> 인코더 입력. 기하·정규화는 전부 `crop.py` 에 위임한다."""

    def __init__(self, refine_landmarks: bool = False, max_num_faces: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 static_image_mode: bool = False,
                 out_h: int = C.OUT_H, out_w: int = C.OUT_W,
                 margin: float = C.MARGIN, lazy: bool = True):
        self.out_h, self.out_w, self.margin = out_h, out_w, margin
        self.refine_landmarks = refine_landmarks
        self._mp_kwargs = dict(
            static_image_mode=static_image_mode, max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence)
        self.face_mesh = None
        if not lazy:
            self._ensure_mesh()

    def _ensure_mesh(self):
        if self.face_mesh is None:
            if _face_mesh_mod is None:
                raise ImportError(f"mediapipe 가 필요합니다: {_MP_ERR}")
            self.face_mesh = _face_mesh_mod.FaceMesh(**self._mp_kwargs)
        return self.face_mesh

    # ------------------------------------------------ 메시 없이 (동치 검증 경로)
    def crop_from_mesh(self, frame_bgr: np.ndarray, mesh_xy: np.ndarray):
        """**픽셀 좌표** 메시 -> (crop uint8, meta) 또는 (None, None).

        학습 경로(`phase2_extract`)가 부르는 것과 **완전히 같은 호출**이다:
            C.crop_both_eyes(frame, xy, out_h, out_w, margin)
        """
        return C.crop_both_eyes(frame_bgr, mesh_xy, self.out_h, self.out_w, self.margin)

    def tensor_from_mesh(self, frame_bgr: np.ndarray, mesh_xy: np.ndarray):
        """메시 -> (1,1,H,W) float32. 정규화는 `crop.INPUT_NORM` 하나만 쓴다."""
        g, meta = self.crop_from_mesh(frame_bgr, mesh_xy)
        if g is None:
            return None, None
        return C.to_input_tensor(g), meta

    # ------------------------------------------------ MediaPipe 경로 (Pi 실행)
    def mesh(self, frame_bgr: np.ndarray):
        """프레임 -> **픽셀 좌표** 메시 (N,2), 얼굴 없으면 None."""
        if cv2 is None:
            raise ImportError("opencv(cv2) 가 필요합니다")
        res = self._ensure_mesh().process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        h, w = frame_bgr.shape[:2]
        m = np.array([[p.x, p.y] for p in lm], np.float32)
        # MediaPipe 는 정규화 좌표를 준다. 픽셀 변환도 crop.py 것을 쓴다.
        return C.mesh_to_pixels(m, w, h, C.COORDS_NORM)

    def crop(self, frame_bgr: np.ndarray):
        m = self.mesh(frame_bgr)
        if m is None:
            return None, None
        return self.crop_from_mesh(frame_bgr, m)

    def input_tensor(self, frame_bgr: np.ndarray):
        m = self.mesh(frame_bgr)
        if m is None:
            return None, None
        return self.tensor_from_mesh(frame_bgr, m)

    def close(self):
        if self.face_mesh is not None:
            self.face_mesh.close()
            self.face_mesh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
