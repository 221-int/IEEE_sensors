"""landmarks — 얼굴 랜드마크 프론트엔드 + EAR + 2단계 캘리브레이션.

역할:
  - MediaPipe FaceLandmarker 초기화(엣지에서 돌릴 대상)
  - 6점 랜드마크로 EAR(Eye Aspect Ratio) 계산 + 머리자세 보정
  - 개안 baseline + 폐안 floor 를 모두 학습하는 2단계 캘리브레이터
    (완성도/불완전 깜빡임 측정에 폐안 기준이 필요)

카메라 루프는 scripts/run_live.py 에 있고, 이 모듈은 신호 프론트엔드만
담당하므로 정지 프레임으로 단위 테스트가 가능하다.
"""
import numpy as np

from . import config

# MediaPipe 는 라이브 프론트엔드에만 필요. 오프라인(features/model/train/
# benchmark)이 MediaPipe 없이도 돌도록 지연 import.
try:
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    _MP_AVAILABLE = True
except Exception:                       # pragma: no cover - 플랫폼 의존
    _MP_AVAILABLE = False


def mediapipe_available() -> bool:
    return _MP_AVAILABLE


def build_landmarker(task_path: str = config.TASK_PATH):
    """MediaPipe FaceLandmarker 를 VIDEO 모드(시계열 트래킹)로 생성."""
    if not _MP_AVAILABLE:
        raise RuntimeError("mediapipe 가 없습니다. 라이브 프론트엔드에 필요합니다.")
    base = mp_python.BaseOptions(model_asset_path=task_path)
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=base,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
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
    """비정면 얼굴용 가벼운 EAR 보정 계수 (>= 1.0)."""
    iod = abs(landmarks[config.LEFT_EYE_L].x - landmarks[config.RIGHT_EYE_R].x)
    return 1.0 + 0.08 * (1.0 - min(iod / 0.4, 1.0))


def frame_ear(landmarks, img_w, img_h):
    """현재 프레임의 보정된 평균 EAR."""
    le = eye_aspect_ratio(landmarks, config.LEFT_EYE,  img_w, img_h)
    re = eye_aspect_ratio(landmarks, config.RIGHT_EYE, img_w, img_h)
    return ((le + re) / 2.0) * head_pose_correction(landmarks)


class TwoPhaseCalibrator:
    """개안 baseline 과 폐안 floor 를 순차로 수집.

    흐름:
        phase == "open"   : 눈 뜨고 CALIB_OPEN_FRAMES 프레임   -> baseline
        phase == "closed" : 눈 꼭 감고 CALIB_CLOSED_FRAMES 프레임 -> closed_floor
        phase == "done"

    완성도 = (baseline - min_EAR) / (baseline - closed_floor) 에 쓰인다.
    폐안 캘리브를 건너뛰려면 closed_floor=0 으로 두면 되고, 그 경우 완성도는
    "개안 baseline 대비 진폭"으로 퇴화한다(구버전과 동일).
    """

    def __init__(self,
                 open_frames=config.CALIB_OPEN_FRAMES,
                 closed_frames=config.CALIB_CLOSED_FRAMES,
                 open_pct=config.BASELINE_PERCENTILE,
                 closed_pct=config.CLOSED_PERCENTILE):
        self.open_frames = open_frames
        self.closed_frames = closed_frames
        self.open_pct = open_pct
        self.closed_pct = closed_pct
        self._open_buf, self._closed_buf = [], []
        self.baseline = None
        self.closed_floor = None

    @property
    def phase(self):
        if self.baseline is None:
            return "open"
        if self.closed_floor is None:
            return "closed"
        return "done"

    @property
    def done(self):
        return self.phase == "done"

    def update(self, ear):
        """EAR 샘플 하나를 현재 단계에 넣고 진행률 [0,1] 반환."""
        if self.phase == "open":
            self._open_buf.append(ear)
            if len(self._open_buf) >= self.open_frames:
                self.baseline = float(np.percentile(self._open_buf, self.open_pct))
            return min(len(self._open_buf) / self.open_frames, 1.0)
        if self.phase == "closed":
            self._closed_buf.append(ear)
            if len(self._closed_buf) >= self.closed_frames:
                self.closed_floor = float(np.percentile(self._closed_buf, self.closed_pct))
            return min(len(self._closed_buf) / self.closed_frames, 1.0)
        return 1.0

    @property
    def threshold(self):
        if self.baseline is None:
            return None
        return self.baseline * config.THRESHOLD_RATIO

    @property
    def closed_ratio(self):
        """폐안 floor 를 baseline 기준으로 정규화한 값(features 로 넘김)."""
        if not self.done or self.baseline <= 0:
            return 0.0
        return float(np.clip(self.closed_floor / self.baseline, 0.0, 0.95))
