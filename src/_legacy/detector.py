"""
detector.py — 얼굴 랜드마크 추출 및 프레임별 눈 신호.

역할:
  - MediaPipe FaceLandmarker 초기화 (비전 센서 프론트엔드)
  - 6개 랜드마크로 EAR(Eye Aspect Ratio) 계산
  - 측면 각도 보정 계수 추정
  - collect.py / liveness.py 가 쓰는 프레임별 API 제공

카메라 루프 자체는 collect.py / liveness.py 에 있고, 이 모듈은 신호
프론트엔드만 담당한다(그래서 정지 프레임으로 단위 테스트가 가능).
"""
import math
import numpy as np

import config

# MediaPipe 는 라이브 프론트엔드에만 필요. 오프라인 파이프라인
# (features/model/train)이 MediaPipe 없이도 돌도록 지연 import 한다.
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    _MP_AVAILABLE = True
except Exception:                      # pragma: no cover - 플랫폼 의존
    _MP_AVAILABLE = False


def build_landmarker(task_path: str = config.TASK_PATH):
    """MediaPipe FaceLandmarker 를 VIDEO 모드(시계열 트래킹)로 생성."""
    if not _MP_AVAILABLE:
        raise RuntimeError(
            "mediapipe 가 없습니다. 라이브 프론트엔드를 쓰려면 설치하세요."
        )
    base = mp_python.BaseOptions(model_asset_path=task_path)
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=base,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        # VIDEO 모드는 시계열 트래킹을 유지 -> IMAGE 모드보다 운동학이 매끄럽다.
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)


def eye_aspect_ratio(landmarks, eye_indices, img_w, img_h):
    """6개 눈 랜드마크로 EAR 계산 (2D 픽셀 좌표)."""
    pts = [np.array([landmarks[i].x * img_w, landmarks[i].y * img_h])
           for i in eye_indices]
    vert = np.linalg.norm(pts[1] - pts[5]) + np.linalg.norm(pts[2] - pts[4])
    horiz = 2.0 * np.linalg.norm(pts[0] - pts[3]) + 1e-6
    return vert / horiz


def head_pose_correction(landmarks):
    """비정면 얼굴용 가벼운 EAR 보정 계수 (>= 1.0).

    양 눈 사이 x-거리를 정면도(frontal-ness) 근사로 사용: 얼굴이 돌아가면
    값이 줄어드므로, EAR 단축(foreshortening)을 상쇄하도록 보정을 키운다.
    TODO: 검증 후 solvePnP 기반 각도로 교체.
    """
    iod = abs(landmarks[config.LEFT_EYE_L].x - landmarks[config.RIGHT_EYE_R].x)
    return 1.0 + 0.08 * (1.0 - min(iod / 0.4, 1.0))


def frame_ear(landmarks, img_w, img_h):
    """현재 프레임의 보정된 평균 EAR."""
    le = eye_aspect_ratio(landmarks, config.LEFT_EYE,  img_w, img_h)
    re = eye_aspect_ratio(landmarks, config.RIGHT_EYE, img_w, img_h)
    return ((le + re) / 2.0) * head_pose_correction(landmarks)


class Calibrator:
    """첫 N 프레임 동안 눈 뜸 기준값(baseline)을 수집."""

    def __init__(self, n_frames=config.CALIBRATION_FRAMES,
                 pct=config.BASELINE_PERCENTILE):
        self.n_frames = n_frames
        self.pct = pct
        self._buf = []
        self.baseline = None

    @property
    def done(self):
        return self.baseline is not None

    def update(self, ear):
        """EAR 샘플 하나를 넣는다; 진행률 [0, 1] 반환."""
        if self.done:
            return 1.0
        self._buf.append(ear)
        if len(self._buf) >= self.n_frames:
            self.baseline = float(np.percentile(self._buf, self.pct))
        return min(len(self._buf) / self.n_frames, 1.0)

    @property
    def threshold(self):
        if not self.done:
            return None
        return self.baseline * config.THRESHOLD_RATIO
