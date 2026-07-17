"""config — 상수·경로 (로직 없음).

프로젝트: 라즈베리파이5 실시간 깜빡임 검출(횟수/BPM) + 엣지 강건화(SR/저조도).
파이프라인: (카메라 -> MJPEG) -> 클라이언트 -> [강건화 전처리] -> MediaPipe
FaceLandmarker -> EAR -> STD 4단계 상태머신 -> 깜빡임 카운트.
비교 baseline: Carcellar et al., IEEE IICAIET 2024 (YOLOv5, Pi3B+, 20초 배치, 85%).
"""
import os

# ── MediaPipe FaceLandmarker 눈 6점 (EAR 규약) ──────────────────────────────
LEFT_EYE   = [362, 385, 387, 263, 373, 380]
RIGHT_EYE  = [33,  160, 158, 133, 153, 144]
LEFT_EYE_L = 33
RIGHT_EYE_R = 263

# ── 개인 캘리브 (개안 baseline + 폐안 floor) ────────────────────────────────
CALIB_OPEN_FRAMES   = 60     # 눈 뜬 상태 수집 프레임
CALIB_CLOSED_FRAMES = 30     # 눈 꼭 감은 상태 수집 프레임
BASELINE_PERCENTILE = 80     # 개안 baseline = 개안 분포 백분위수
CLOSED_PERCENTILE   = 50     # 폐안 floor  = 폐안 분포 백분위수(중앙값)
THRESHOLD_RATIO     = 0.75   # (landmarks 단순 임계값, 참고용)

# ── 완성도(선택 지표): (baseline-min_EAR)/(baseline-closed_floor) ───────────
COMPLETE_THRESHOLD = 0.80    # 이 값 미만 = 불완전 깜빡임(선택 지표)

# ── 지표 집계 창 ────────────────────────────────────────────────────────────
METRIC_WINDOW_SEC = 60.0     # 깜빡임률(bpm) 집계 창

# ── EAR 신호 평활 (EMA): s_t = a*ear + (1-a)*s_{t-1} ────────────────────────
EMA_ALPHA = 0.5

# ── STD 4단계 상태머신 임계값 (개안~폐안 밴드 비율) ─────────────────────────
# band = baseline - closed_floor
#   t_closed = closed_floor + STD_CLOSE_FRAC*band  (이 아래 = 폐안 영역)
#   t_open   = closed_floor + STD_OPEN_FRAC*band   (이 위   = 개안 영역)
STD_CLOSE_FRAC     = 0.25
STD_OPEN_FRAC      = 0.55
STD_MAX_CLOSED_SEC = 1.0     # 이보다 오래 감김 = 깜빡임 아님(졸음/내려봄) -> 실격

# ── MJPEG 스트리밍 (서버-클라이언트) ────────────────────────────────────────
STREAM_HOST    = "0.0.0.0"
STREAM_PORT    = 8000
STREAM_PATH    = "/stream.mjpg"
STREAM_QUALITY = 80          # JPEG 품질(1-100)
CAM_INDEX  = 0
CAM_WIDTH  = 1280
CAM_HEIGHT = 720
CAM_FPS    = 30

# ── 경로 (프로젝트 루트 = src/eyeblink 의 2단계 상위) ───────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.dirname(os.path.dirname(_HERE))   # .../IEEE_sensors
TASK_PATH   = os.path.join(_BASE, "face_landmarker.task")  # MediaPipe 모델
DATA_DIR    = os.path.join(_BASE, "data")
MODEL_DIR   = os.path.join(_BASE, "models")
RESULTS_DIR = os.path.join(_BASE, "results")
for _d in (DATA_DIR, MODEL_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)
