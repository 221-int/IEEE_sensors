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

# ── STAGE 2: 실시간 예산 / 조건부 SR / 합성 열화 (2026-07-17 승인) ──────────
# 근거: 참고자료/STAGE2_결정사항.md(A·B·C), 참고자료/SR_STAGE2_설계노트.md
# fps 운영점: no-SR ~30fps / 목표 24fps / 하한 15fps(필수)
TARGET_FPS_MIN = 15          # 필수 하한 (이 아래로 떨어지면 SR 강등/오프)
TARGET_FPS     = 24          # SR-on 목표 운영점

# 조건부 얼굴 ROI two-pass SR (frontend.EarFrontend 가 사용)
SR_ENABLE      = True        # 전역 스위치
SR_W_EYE_MIN   = 24          # 게이트: 눈 가로 px < 이 값이면 SR 켬 (초안 — w_eye 분포로 확정)
SR_W_EYE_HYST  = 30          # (선택) 채터링 억제용 히스테리시스 해제 임계 (>= SR_W_EYE_MIN)
SR_MODEL       = "fsrcnn"    # cv2.dnn_superres 모델키: "fsrcnn" | "espcn"
SR_SCALE       = 2           # 업스케일 배율 (2 우선; 3/4 는 STAGE2 스윕)
SR_FACE_MARGIN = 0.3         # 얼굴 bbox 여유 비율 (crop 시 상하좌우 확장)
# SR 사전학습 .pb 파일은 MODEL_DIR 에 둔다. 파일명 규약(대소문자 주의):
#   FSRCNN: "FSRCNN-small_x{scale}.pb"  /  ESPCN: "ESPCN_x{scale}.pb"
# 실제 경로 해석은 robust.resolve_sr_model_path() 가 담당(config 는 로직 없음).

# 합성 열화 표준 (이번 라운드: 자체 4인 영상에 적용 — scripts/sr_eval.py)
DEGRADE_DOWNSCALE   = (2, 3, 4)         # 원본 대비 축소 배율
DEGRADE_GAMMA       = (1.0, 0.6, 0.4)   # 감마(<1 = 어둡게)
DEGRADE_NOISE_SIGMA = (0, 5, 10)        # 가우시안 노이즈 표준편차(8bit 기준)

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

# ── 피험자별 캘리브레이션 실험 (박사님 프로토콜) ─────────────────────────────
CALIB_OPEN_SEC   = 5.0    # 개안 EAR 수집 시간(초)
CALIB_CLOSED_SEC = 3.0    # 폐안 EAR 수집 시간(초)
CALIB_TRIALS     = 10     # 깜빡임 검증 시행 횟수(피험자당)
CALIB_TRIAL_SEC  = 4.0    # 시행 1회당 창(초) — 이 안에 1회 깜빡이도록 지시
