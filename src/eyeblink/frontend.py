"""frontend — 프레임 → EAR '프론트엔드' (랜드마커 + 조건부 얼굴 ROI two-pass SR).

【위치/역할】 파이프라인 계층. 프레임 -> FaceLandmarker -> frame_ear 사이에 조건부
SR(two-pass)을 끼운다. robust.SuperResolution 은 순수 SR 유틸(frame→frame)로 두고,
여기서 landmarker 핸들을 쥐고 오케스트레이션한다.

【데이터 흐름】
    frame ──▶ pass1: FaceLandmarker(VIDEO, 원본) ──▶ 얼굴 없음? → EAR=None
                     │ 얼굴 있음
                     ▼
              w_eye = eye_width_px(landmarks)
                     │
          Gate: SR 이고 w_eye < SR_W_EYE_MIN ?
             │ 아니오                         │ 예
             ▼                                ▼
       EAR = frame_ear(pass1)      face_crop = crop(face_bbox, margin)
       used_sr = False             face_hr   = SR.upsample(face_crop)   [robust]
                                   pass2: FaceLandmarker(IMAGE, face_hr)
                                   EAR = frame_ear(pass2) | 폴백(pass1)
                                   used_sr = True
    반환: FrameResult(ear, has_face, w_eye, used_sr, lm_ms, sr_ms)

【★1 결정(2026-07-20)】 pass2 는 **별도 IMAGE 모드 landmarker(stateless detect)**. 이유:
  - VIDEO 트래커에 업스케일 crop 을 섞으면 트래킹 상태가 오염됨(다음 프레임 품질 저하).
  - detect_for_video 의 타임스탬프 단조성 제약을 우회.
pass2 landmarker 는 SR 이 처음 필요할 때 지연 생성(메모리 절약).

【★2 미검증】 MediaPipe 가 얼굴 crop 을 내부 고정크기로 리사이즈하면 업스케일 이득이
상쇄될 수 있음 → go/no-go 파일럿(scripts.sr_eval)에서 EAR/검출 델타로 확인.
【★3】 EAR 은 스케일 불변 비율이라 pass2 crop 좌표계에서 계산해도 역매핑 불필요.

Gate 로직/FrameResult 는 MediaPipe 없이 단위 테스트 가능(순수).
"""
from dataclasses import dataclass
from typing import Optional

from . import config
from . import landmarks as L


@dataclass
class FrameResult:
    """한 프레임 처리 결과(프론트엔드 산출 + 텔레메트리)."""
    ear: Optional[float]          # EAR (얼굴 없으면 None)
    has_face: bool
    w_eye: Optional[float] = None # 눈 가로 px (게이트 신호)
    used_sr: bool = False         # 이 프레임에 SR two-pass 적용?
    lm_ms: float = 0.0            # 랜드마크 검출 소요(pass1[+pass2]) ms
    sr_ms: float = 0.0            # SR upsample 소요 ms (SR-off 면 0)
    landmarks: object = None      # 하위 필요 시(선택). 기본 미보관


class SrGate:
    """w_eye 기반 조건부 SR 스위치 (히스테리시스 포함) — 순수 로직, 단위 테스트 가능.

    on_below  : 이 픽셀폭 미만이면 SR 켬 (config.SR_W_EYE_MIN)
    off_above : 켜진 상태에서 이 픽셀폭 이상이면 끔 (config.SR_W_EYE_HYST)
    on_below <= off_above 로 채터링 억제.
    """

    def __init__(self, on_below=config.SR_W_EYE_MIN, off_above=config.SR_W_EYE_HYST,
                 enable=config.SR_ENABLE):
        self.on_below = float(on_below)
        self.off_above = float(max(off_above, on_below))
        self.enable = bool(enable)
        self._on = False

    @property
    def is_on(self):
        return self._on

    def decide(self, w_eye):
        """이번 프레임 SR 사용 여부 갱신·반환. w_eye None(얼굴없음)이면 상태 유지·False."""
        if not self.enable or w_eye is None:
            return False
        if self._on:
            if w_eye >= self.off_above:
                self._on = False
        else:
            if w_eye < self.on_below:
                self._on = True
        return self._on


class EarFrontend:
    """프레임 -> EAR 프론트엔드. two-pass SR 오케스트레이션의 소유자.

    landmarker : L.build_landmarker() (VIDEO 모드) — pass1. 스캐폴딩용 None 허용.
    sr         : robust.SuperResolution 인스턴스 또는 None(=SR 비활성).
    gate       : SrGate 또는 None(=기본 생성).
    task_path  : pass2(IMAGE 모드) landmarker 지연 생성용 .task 경로.
    """

    def __init__(self, landmarker=None, sr=None, gate=None, task_path=config.TASK_PATH):
        self.landmarker = landmarker
        self.sr = sr
        self.task_path = task_path
        self.gate = gate or SrGate(enable=(config.SR_ENABLE and sr is not None))
        self._pass2 = None            # 지연 생성되는 IMAGE 모드 landmarker

    # ── pass1: VIDEO 모드(시계열 트래킹) ───────────────────────────────────────
    def _detect_video(self, frame_bgr, ts_ms):
        import cv2
        import mediapipe as mp
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(ts_ms))
        return res.face_landmarks[0] if res.face_landmarks else None

    # ── pass2: IMAGE 모드(stateless) — 업스케일 crop 전용 ──────────────────────
    def _detect_image(self, img_bgr):
        import cv2
        import mediapipe as mp
        if self._pass2 is None:
            self._pass2 = L.build_landmarker(self.task_path, mode="image")
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = self._pass2.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        return res.face_landmarks[0] if res.face_landmarks else None

    def process(self, frame, ts_ms) -> FrameResult:
        """프레임 하나 -> FrameResult."""
        import time
        if self.landmarker is None:
            raise RuntimeError("EarFrontend.process: landmarker 미주입")

        h, w = frame.shape[:2]

        # pass1 (원본, VIDEO)
        t0 = time.perf_counter()
        lms = self._detect_video(frame, ts_ms)
        lm_ms = (time.perf_counter() - t0) * 1e3
        if lms is None:
            return FrameResult(ear=None, has_face=False, lm_ms=lm_ms)

        w_eye = L.eye_width_px(lms, w, h)
        want_sr = self.gate.decide(w_eye) and (self.sr is not None)
        if not want_sr:
            return FrameResult(ear=L.frame_ear(lms, w, h), has_face=True,
                               w_eye=w_eye, used_sr=False, lm_ms=lm_ms)

        # ── SR two-pass ────────────────────────────────────────────────────
        x0, y0, x1, y1 = L.face_bbox(lms, w, h, margin=config.SR_FACE_MARGIN)
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:                       # 퇴화 bbox → SR 폴백
            return FrameResult(ear=L.frame_ear(lms, w, h), has_face=True,
                               w_eye=w_eye, used_sr=False, lm_ms=lm_ms)

        t1 = time.perf_counter()
        crop_hr = self.sr(crop)                  # robust.SuperResolution (frame→frame)
        sr_ms = (time.perf_counter() - t1) * 1e3

        t2 = time.perf_counter()
        lms2 = self._detect_image(crop_hr)
        lm_ms += (time.perf_counter() - t2) * 1e3

        if lms2 is not None:
            ch, cw = crop_hr.shape[:2]
            ear = L.frame_ear(lms2, cw, ch)      # EAR 스케일 불변 → 역매핑 불필요
            used = True
        else:
            ear = L.frame_ear(lms, w, h)         # pass2 얼굴 못찾으면 pass1 폴백
            used = False
        return FrameResult(ear=ear, has_face=True, w_eye=w_eye,
                           used_sr=used, lm_ms=lm_ms, sr_ms=sr_ms)
