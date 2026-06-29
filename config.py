"""
config.py — 상수, 랜드마크 인덱스, 경로 모음.

프로젝트: 비접촉 의도-깜빡임 센싱 (Contactless Voluntary-Blink Sensing)
  코어   : 카메라 한 대로, 각 깜빡임을 운동학(kinematics)만 보고
           VOLUNTARY(의도적) vs SPONTANEOUS(무의식)로 분류
  데모   : 깜빡임 챌린지-리스폰스 기반 카메라 liveness / anti-spoofing

이 파일은 설정만 담는다(로직 없음).
"""
import os

# ── MediaPipe FaceLandmarker 랜드마크 인덱스 ────────────────────────────────
# 6점 눈 모델 (EAR 정의와 동일한 규약).
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# 머리 자세 기준점 (solvePnP / EAR 보정용).
NOSE_TIP   = 1
CHIN       = 152
LEFT_EYE_L = 33
RIGHT_EYE_R = 263
LEFT_MOUTH = 61
RIGHT_MOUTH = 291

# ── 캘리브레이션 / 검출 파라미터 ────────────────────────────────────────────
CALIBRATION_FRAMES = 60      # 눈 뜸 기준값 학습에 쓰는 프레임 수
BASELINE_PERCENTILE = 80     # 눈 뜸 EAR = 캘리브 분포의 이 백분위수
THRESHOLD_RATIO = 0.75       # EAR < ratio * baseline 이면 깜빡임 시작
SLOPE_THRESH = 0.005         # 닫힘/열림 프레임으로 볼 최소 |기울기|
BLINK_MIN_FRAMES = 2         # 깜빡임으로 셀 최소 연속 감김 프레임
BLINK_MAX_FRAMES = 60        # 깜빡임이 아닌 장시간 감김 방지용 상한

# ── 깜빡임 종류 라벨 ────────────────────────────────────────────────────────
# 기본은 이진. REFLEX(반사)는 선택(3클래스)이며 수집 전에는 끔.
LABELS = ["spontaneous", "voluntary"]          # 0, 1
LABEL_MAP = {name: i for i, name in enumerate(LABELS)}

# ── 깜빡임별 특징 벡터 스키마 (features.py 참조) ────────────────────────────
FEATURE_NAMES = [
    "dur_total",        # 전체 깜빡임 지속시간 (s)
    "dur_closing",      # 닫힘(onset) 지속시간 (s)
    "dur_valley",       # 최저점 부근 유지시간 (s)
    "dur_opening",      # 열림(offset) 지속시간 (s)
    "amplitude",        # 완성도 = (baseline - min) / baseline
    "min_ratio",        # 최저 EAR / baseline
    "peak_close_vel",   # 피크 닫힘 속도 (|기울기|, 1/s)
    "peak_open_vel",    # 피크 열림 속도 (|기울기|, 1/s)
    "asym_dur",         # dur_closing / dur_opening
    "asym_vel",         # peak_close_vel / peak_open_vel
    "auc_deficit",      # 깜빡임 구간에서 baseline과 EAR 사이 면적
]
FEATURE_LEN = len(FEATURE_NAMES)

# ── 데이터 수집 프로토콜 ────────────────────────────────────────────────────
# VOLUNTARY(신호 깜빡임) 블록과 SPONTANEOUS(자유 시청) 블록을 번갈아 진행.
CUE_INTERVAL_SEC = 2.5       # 의도 깜빡임 신호 간격 (초)
N_VOLUNTARY_CUES = 20        # voluntary 블록당 신호 횟수
SPONTANEOUS_SEC  = 90        # spontaneous 블록당 자유 시청 시간 (초)
N_BLOCKS = 2                 # (voluntary, spontaneous) 쌍 반복 횟수

# ── Liveness 챌린지-리스폰스 ────────────────────────────────────────────────
CHALLENGE_N_BLINKS = 3       # 요청하는 의도 깜빡임 횟수
CHALLENGE_TIMEOUT_SEC = 6.0  # 챌린지 제한 시간 (초)
CHALLENGE_MIN_VOLUNTARY = 3  # 통과에 필요한 voluntary 판정 깜빡임 수

# ── 경로 ────────────────────────────────────────────────────────────────────
_BASE     = os.path.dirname(os.path.abspath(__file__))
TASK_PATH = os.path.join(_BASE, "face_landmarker.task")   # MediaPipe 모델
DATA_DIR  = os.path.join(_BASE, "data")                   # 라벨링된 깜빡임 CSV
MODEL_DIR = os.path.join(_BASE, "models")                 # 학습된 분류기
RESULTS_DIR = os.path.join(_BASE, "results")             # 표/그림

for _d in (DATA_DIR, MODEL_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)

BLINK_CSV = os.path.join(DATA_DIR, "blinks.csv")          # 전체 라벨 깜빡임
MODEL_PATH = os.path.join(MODEL_DIR, "blink_type.npz")
