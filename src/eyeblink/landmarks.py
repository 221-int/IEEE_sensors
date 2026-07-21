"""landmarks — 얼굴 랜드마크 프론트엔드 + EAR + 2단계 캘리브레이션.

역할:
  - MediaPipe FaceLandmarker 초기화(엣지에서 돌릴 대상)
  - 6점 랜드마크로 EAR(Eye Aspect Ratio) 계산 + 머리자세 보정
  - 개안 baseline + 폐안 floor 를 모두 학습하는 2단계 캘리브레이터

카메라 루프는 scripts/run_live.py 에 있고, 이 모듈은 신호 프론트엔드만
담당하므로 정지 프레임으로 단위 테스트가 가능하다.
"""
import os

import numpy as np

from . import config

# MediaPipe 는 라이브 프론트엔드에만 필요. 순수 로직(std/metrics/pipeline)이
# MediaPipe 없이도 돌도록 지연 import.
try:
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    _MP_AVAILABLE = True
except Exception:                       # pragma: no cover - 플랫폼 의존
    _MP_AVAILABLE = False

_TASK_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")


def mediapipe_available() -> bool:
    return _MP_AVAILABLE


def build_landmarker(task_path: str = config.TASK_PATH):
    """MediaPipe FaceLandmarker 를 VIDEO 모드(시계열 트래킹)로 생성.

    mediapipe 미설치 또는 모델 파일(.task) 부재 시, 원인과 해결법을 담은
    예외를 던진다(트레이스백만 남기고 죽지 않도록).
    """
    if not _MP_AVAILABLE:
        raise RuntimeError(
            "mediapipe 가 설치되어 있지 않습니다 (라이브 프론트엔드에 필요).\n"
            "  설치:  pip install mediapipe\n"
            "  Pi5/aarch64 에서 실패하면 버전 지정:  pip install 'mediapipe==0.10.14'")
    if not os.path.exists(task_path):
        raise FileNotFoundError(
            "face_landmarker.task 모델 파일이 없습니다.\n"
            f"  기대 경로: {task_path}\n"
            "  (.gitignore 의 *.task 때문에 git clone 에 포함되지 않습니다 — 직접 받아 두세요)\n"
            "  다운로드 (프로젝트 루트에서):\n"
            f"    curl -L -o face_landmarker.task {_TASK_URL}")
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


def eye_width_px(landmarks, img_w, img_h):
    """양안 평균 '눈 가로 픽셀 폭'(w_eye) — 조건부 SR 게이트 신호.

    각 눈의 안/바깥 눈꼬리(EAR 규약의 pts[0], pts[3]) 사이 유클리드 거리(px)를
    구해 좌우 평균. 머리 기울임에 견디도록 대각 거리를 쓴다.

    반환값이 config.SR_W_EYE_MIN 미만이면 '눈이 작다(멀다/저해상도)'로 보고
    frontend 에서 얼굴 ROI SR 을 켠다. (STAGE2: 값 자체는 EAR 과 무관, 게이트 전용)
    """
    def _w(idx):
        a, b = landmarks[idx[0]], landmarks[idx[3]]
        return float(np.hypot((a.x - b.x) * img_w, (a.y - b.y) * img_h))
    return 0.5 * (_w(config.LEFT_EYE) + _w(config.RIGHT_EYE))


def face_bbox(landmarks, img_w, img_h, margin=config.SR_FACE_MARGIN):
    """전체 얼굴 랜드마크의 경계상자(px, 정수) + 여유 마진, 이미지 경계로 클램프.

    two-pass SR 의 pass1 결과로 '얼굴 crop' 을 떠서 SR 에 넣을 때 사용.
    반환: (x0, y0, x1, y1)  — crop = frame[y0:y1, x0:x1].
    """
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x0, x1 = min(xs) * img_w, max(xs) * img_w
    y0, y1 = min(ys) * img_h, max(ys) * img_h
    bw, bh = (x1 - x0), (y1 - y0)
    x0 -= margin * bw; x1 += margin * bw
    y0 -= margin * bh; y1 += margin * bh
    x0 = int(max(0, round(x0)));      y0 = int(max(0, round(y0)))
    x1 = int(min(img_w, round(x1)));  y1 = int(min(img_h, round(y1)))
    return x0, y0, x1, y1


class TwoPhaseCalibrator:
    """개안 baseline 과 폐안 floor 를 순차로 수집.

    흐름:
        phase == "open"   : 눈 뜨고 CALIB_OPEN_FRAMES 프레임    -> baseline
        phase == "closed" : 눈 꼭 감고 CALIB_CLOSED_FRAMES 프레임 -> closed_floor
        phase == "done"
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
        if not self.done or self.baseline <= 0:
            return 0.0
        return float(np.clip(self.closed_floor / self.baseline, 0.0, 0.95))
