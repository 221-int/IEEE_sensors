"""eyeblink — 라즈베리파이5 실시간 깜빡임 검출 + 엣지 강건화 파이프라인.

새 방향(척추):
  카메라 -> (MJPEG 서버) -> 클라이언트 -> [강건화 전처리] -> MediaPipe
  FaceLandmarker -> EAR -> STD 4단계 상태머신 -> 깜빡임 카운트/BPM.
비교 baseline: Carcellar et al., IEEE IICAIET 2024 (Pi3B+, 20초 배치, 85%).

모듈:
  config      상수/경로 (STD 임계값/스트리밍/캘리브)
  landmarks   MediaPipe 프론트엔드 + EAR + 2단계(개안/폐안) 캘리브
  std         STD 4단계 상태머신(A->B->C->D->A) -> 깜빡임 이벤트
  metrics     카운트 + 분당 깜빡임률(BPM) 롤링 집계 (+ 불완전 비율)
  robust      강건화 전처리 훅(SR/저조도) - 향후 경량 모델 자리
  profiling   fps / 단계별 처리시간 계측 유틸
  pipeline    EAR 스트림 -> 캘리브 -> EMA -> STD -> 카운트 글루
  streaming/  MJPEG 서버(카메라) + 클라이언트(프레임 수신)
"""
from . import config  # noqa: F401
