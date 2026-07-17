"""robust — 강건화 전처리 훅 (저조도·저해상도 대응).

파이프라인의 [강건화] 단계 인터페이스. 지금은 Passthrough(무동작)만 실동작하고,
SuperResolution(해상도)·LowLight(조도)는 향후 경량 모델을 얹을 자리(스텁)다.

  - SuperResolution : ESPCN / FSRCNN / edge-SR 등 경량 SR (저해상도 → 고해상도)
  - LowLight        : Zero-DCE 등 경량 저조도 개선

모두 BGR ndarray in → BGR ndarray out. 실제 검출을 돕는지(순차 SR 손해 사례)
반드시 검증 후 탑재. get_preprocessor(name) 으로 파이프라인에 주입.
"""


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


class SuperResolution(Preprocessor):
    """저해상도 → 고해상도 (경량 SR). TODO: ESPCN/FSRCNN/edge-SR 탑재."""
    name = "sr"

    def __init__(self, scale=2):
        self.scale = scale

    def process(self, frame):
        raise NotImplementedError(
            "SR 미구현 — 경량 SR 모델(ESPCN/FSRCNN/edge-SR)을 연결할 자리")


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
