"""frontend — 프레임 → EAR '프론트엔드' (랜드마커 + 조건부 얼굴 ROI two-pass SR).

【위치/역할】 파이프라인 계층. 지금까지 scripts/run_live.py 와 scripts/fps_sweep.py 에
인라인 중복돼 있던 "프레임 -> FaceLandmarker -> frame_ear" 로직을 한 곳으로 모으고,
그 사이에 조건부 SR(two-pass)을 끼운다. robust.SuperResolution 은 순수 SR 유틸로
두고(관심사 분리), 여기서 landmarker 핸들을 쥐고 오케스트레이션한다.

【데이터 흐름 (설계)】
    frame ──▶ pass1: FaceLandmarker(원본) ──▶ 얼굴 없음? → EAR=None 반환
                     │ 얼굴 있음
                     ▼
              w_eye = eye_width_px(landmarks)
                     │
          Gate: SR_ENABLE 이고 w_eye < SR_W_EYE_MIN ?
             │ 아니오                         │ 예
             ▼                                ▼
       EAR = frame_ear(pass1)      face_crop = crop(face_bbox, margin)
       used_sr = False             face_hr   = SR.upsample(face_crop)   [robust]
                                   pass2: FaceLandmarker(face_hr)
                                   EAR = frame_ear(pass2 or pass1 폴백)
                                   used_sr = True
    반환: FrameResult(ear, w_eye, used_sr, lm_ms, sr_ms, has_face)

【스캐폴딩 상태 2026-07-20 — 다음 라운드에 반드시 해결할 미결(★)】
  ★1 VIDEO 모드 타임스탬프: detect_for_video 는 단조 증가 ts 를 요구한다. pass1/pass2 를
     같은 ts 로 부르면 규약 위반. 해결안 후보: (a) pass2 는 별도 IMAGE 모드 landmarker,
     (b) pass2 용 landmarker 인스턴스 분리, (c) ts 를 +1ms. → 결정 후 _detect 확정.
  ★2 MediaPipe 내부 리사이즈: FaceLandmarker 가 얼굴 crop 을 자체 입력크기로 리사이즈하면
     업스케일 이득이 상쇄될 수 있음(STAGE1 열린질문). go/no-go 파일럿에서 먼저 확인.
  ★3 EAR 스케일 불변(STAGE1 F3): pass2 EAR 은 정규화좌표 비율이라 역매핑 불필요. 단
     pass2 가 crop 좌표계라 얼굴 외 다른 지표를 쓸 땐 offset/scale 역매핑 필요 — 지금은
     EAR 만 쓰므로 불필요(확인).

Gate 로직/FrameResult 는 MediaPipe 없이 단위 테스트 가능(순수). 프레임 처리부(process)는
landmarker 가 주입돼야 동작하며, ★1 결정 전까지 SR 경로는 명시적으로 막아둔다.
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

    landmarker : L.build_landmarker() 결과 (VIDEO 모드). 스캐폴딩 단계에선 None 허용.
    sr         : robust.SuperResolution 인스턴스 또는 None(=SR 비활성).
    gate       : SrGate 또는 None(=기본 생성).
    """

    def __init__(self, landmarker=None, sr=None, gate=None):
        self.landmarker = landmarker
        self.sr = sr
        self.gate = gate or SrGate(enable=(config.SR_ENABLE and sr is not None))

    # ── 내부: 단일 landmarker 호출 (★1 타임스탬프 정책 확정 지점) ──────────────
    def _detect(self, image_bgr, ts_ms):
        """BGR 이미지 -> MediaPipe FaceLandmarker 결과. 랜드마크 리스트(or None) 반환.

        TODO(★1): pass1/pass2 타임스탬프 단조성 정책 확정 후 이 메서드로 일원화.
        """
        import cv2
        import mediapipe as mp
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        res = self.landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(ts_ms))
        return res.face_landmarks[0] if res.face_landmarks else None

    def process(self, frame, ts_ms) -> FrameResult:
        """프레임 하나 -> FrameResult. (스캐폴딩: SR 경로는 ★1 결정 전까지 폴백)"""
        import time
        if self.landmarker is None:
            raise RuntimeError("EarFrontend.process: landmarker 미주입 (스캐폴딩 상태)")

        h, w = frame.shape[:2]

        # pass1 (원본)
        t0 = time.perf_counter()
        lms = self._detect(frame, ts_ms)
        lm_ms = (time.perf_counter() - t0) * 1e3
        if lms is None:
            return FrameResult(ear=None, has_face=False, lm_ms=lm_ms)

        w_eye = L.eye_width_px(lms, w, h)
        want_sr = self.gate.decide(w_eye) and (self.sr is not None)

        if not want_sr:
            ear = L.frame_ear(lms, w, h)
            return FrameResult(ear=ear, has_face=True, w_eye=w_eye,
                               used_sr=False, lm_ms=lm_ms)

        # ── SR two-pass 경로 (스캐폴딩) ─────────────────────────────────────
        # TODO(★1): pass2 타임스탬프 정책 확정 전까지 실제 pass2 를 켜지 않는다.
        #           확정 후 아래 블록을 활성화하고 폴백 제거.
        raise NotImplementedError(
            "SR two-pass 경로는 스캐폴딩 상태 — 타임스탬프 정책(★1)·"
            "MediaPipe 내부 리사이즈(★2) 확인 후 활성화. 설계노트 참조.")
        # x0, y0, x1, y1 = L.face_bbox(lms, w, h, margin=config.SR_FACE_MARGIN)
        # crop = frame[y0:y1, x0:x1]
        # t1 = time.perf_counter()
        # crop_hr = self.sr(crop)                 # robust.SuperResolution (frame→frame)
        # sr_ms = (time.perf_counter() - t1) * 1e3
        # t2 = time.perf_counter()
        # lms2 = self._detect(crop_hr, ts_ms + 1) # ★1: 임시 +1ms
        # lm_ms += (time.perf_counter() - t2) * 1e3
        # ch, cw = crop_hr.shape[:2]
        # ear = L.frame_ear(lms2, cw, ch) if lms2 is not None else L.frame_ear(lms, w, h)
        # return FrameResult(ear=ear, has_face=True, w_eye=w_eye,
        #                    used_sr=lms2 is not None, lm_ms=lm_ms, sr_ms=sr_ms)
