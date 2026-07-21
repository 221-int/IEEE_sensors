"""robust — 강건화 전처리 훅 + 순수 SR 유틸 (저조도·저해상도 대응).

역할(STAGE2 이후):
  - Preprocessor 규약은 그대로 **BGR ndarray in → BGR ndarray out** (프레임/크롭 무관).
  - SuperResolution 은 이제 '순수 SR 유틸'이다: 주어진 이미지를 배율만큼 업스케일할
    뿐, MediaPipe 나 게이트 로직은 모른다.
  - **얼굴 ROI two-pass 오케스트레이션(pass1→게이트→crop SR→pass2)은 여기 두지 않는다.**
    그건 landmarker 핸들을 가진 파이프라인 계층(frontend.EarFrontend)이 담당한다.
    (근거: 참고자료/SR_STAGE2_설계노트.md — 관심사 분리, frame→frame 규약 유지)

스캐폴딩 상태(2026-07-20):
  - SuperResolution 은 cv2.dnn_superres 래퍼로 골격을 잡았다. 실제 .pb 모델과
    Pi5 실측 검증은 다음 라운드. 모델/런타임 부재 시 조용히 통과하지 않고 명확히 예외.
  - LowLight(Zero-DCE 등)는 여전히 스텁.
"""
import os

from . import config


class Preprocessor:
    name = "base"

    def process(self, frame):
        raise NotImplementedError

    def __call__(self, frame):
        return self.process(frame)


class Passthrough(Preprocessor):
    name = "none"

    def process(self, frame):
        return frame


# ── SR 사전학습 모델(.pb) 경로 규약 ─────────────────────────────────────────
# cv2.dnn_superres setModel 키는 소문자("fsrcnn"/"espcn"). FSRCNN-small 도 키는 "fsrcnn".
_SR_FILENAME = {
    "fsrcnn": "FSRCNN-small_x{scale}.pb",   # 1순위(경량). 표준 FSRCNN 쓰면 "FSRCNN_x{scale}.pb"
    "espcn":  "ESPCN_x{scale}.pb",          # 대안
}


def resolve_sr_model_path(model=config.SR_MODEL, scale=config.SR_SCALE):
    """모델키+배율 -> config.MODEL_DIR 내 .pb 경로. 파일 존재 여부는 검사하지 않음."""
    key = (model or "").lower()
    if key not in _SR_FILENAME:
        raise ValueError(f"알 수 없는 SR 모델: {model} (가능: {list(_SR_FILENAME)})")
    return os.path.join(config.MODEL_DIR, _SR_FILENAME[key].format(scale=int(scale)))


class SuperResolution(Preprocessor):
    """경량 SR 유틸 (cv2.dnn_superres). 입력 이미지를 scale 배 업스케일.

    frame→frame 순수 함수. 게이트/크롭/pass2 없음(→ frontend 담당).
    모델은 첫 호출 시 1회 지연 로드.

    TODO(다음 라운드): .pb 다운로드·배치 후 Pi5 실측(latency/fps). 배율 2×/3× 스윕.
    """
    name = "sr"

    def __init__(self, model=config.SR_MODEL, scale=config.SR_SCALE, model_path=None):
        self.model = (model or "fsrcnn").lower()
        self.scale = int(scale)
        self.model_path = model_path or resolve_sr_model_path(self.model, self.scale)
        self._sr = None                      # 지연 로드된 DnnSuperResImpl

    def _ensure_loaded(self):
        if self._sr is not None:
            return
        import cv2
        if not hasattr(cv2, "dnn_superres"):
            raise RuntimeError(
                "cv2.dnn_superres 가 없습니다 — opencv-contrib-python 필요.\n"
                "  설치:  pip install opencv-contrib-python")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                "SR 모델(.pb)이 없습니다.\n"
                f"  기대 경로: {self.model_path}\n"
                "  받기: FSRCNN https://github.com/Saafke/FSRCNN_Tensorflow/tree/master/models\n"
                "        ESPCN  https://github.com/fannymonori/TF-ESPCN/tree/master/export")
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(self.model_path)
        sr.setModel(self.model, self.scale)   # 예: ("fsrcnn", 2)
        self._sr = sr

    def process(self, frame):
        """BGR ndarray -> scale× 업스케일된 BGR ndarray."""
        self._ensure_loaded()
        return self._sr.upsample(frame)


class LowLight(Preprocessor):
    """저조도 개선. TODO: Zero-DCE 등 경량 모델 탑재."""
    name = "lowlight"

    def process(self, frame):
        raise NotImplementedError(
            "저조도 개선 미구현 — Zero-DCE 등 경량 모델을 연결할 자리")


_REGISTRY = {"none": Passthrough, "sr": SuperResolution, "lowlight": LowLight}


def get_preprocessor(name="none", **kw):
    """이름으로 전처리기 생성. name in {none, sr, lowlight}."""
    key = (name or "none").lower()
    if key not in _REGISTRY:
        raise ValueError(f"알 수 없는 전처리: {name} (가능: {list(_REGISTRY)})")
    return _REGISTRY[key](**kw)
