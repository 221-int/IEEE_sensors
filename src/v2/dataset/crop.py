"""크롭 기하 — v2 단일 진리원.

이 파일 밖에서 눈을 자르는 코드를 만들지 마십시오. 학습·평가·배포(Pi5)가 전부
여기 있는 `crop_both_eyes()` 하나만 부릅니다. 전처리가 두 벌이 되면 train/serve
기하가 조용히 갈라지고, 그 차이는 나중에 성능 차이로만 보여서 원인을 못 찾습니다.

의존성은 cv2 + numpy 뿐입니다. MediaPipe 를 여기서 import 하지 않는 것은 의도이며,
ONNX 만 있는 배포 환경(Pi)에서도 같은 기하를 그대로 재사용하기 위해서입니다.
검출기는 호출자가 알아서 주고, 이 모듈은 **이미 구해진 랜드마크**만 받습니다.

좌표계
------
mEBAL2 의 landmarks.csv 는 **픽셀** 좌표입니다(정규화 [0,1] 이 아님). MediaPipe 를
직접 돌리면 정규화 좌표가 나옵니다. 그래서 좌표계를 **명시적 인자**로 받습니다.
기본값을 두지 않은 이유는, 기본값이 있으면 언젠가 틀린 쪽으로 조용히 흘러가기
때문입니다. 호출자가 매번 밝히도록 강제합니다.

출력 규격 — **미확정**
--------------------
OUT_H=64, OUT_W=160, MARGIN=2.2 는 기존 파이프라인이 쓰던 값을 **출발점으로만**
가져온 것입니다. v2 에서 확정된 값이 아니며 Phase 1/2 의 기하·광학 측정 결과로
정합니다. 그때까지 이 상수를 '정해진 규격'처럼 인용하지 마십시오.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np

# --- 출력 규격 (미확정, Phase 1/2 에서 확정) ---
OUT_H, OUT_W = 64, 160
MARGIN = 2.2          # 크롭 폭 = 두 눈 중심 거리 x MARGIN
MIN_SPAN_PX = 5.0     # 두 눈 중심 거리가 이보다 작으면 검출 실패로 본다

# --- MediaPipe FaceMesh 눈꼬리 인덱스 (468점 메시) ---
# 각 쌍이 한쪽 눈의 두 눈꼬리. 어느 쪽이 이미지 왼쪽인지는 미러링 여부에 달렸으므로
# 하드코딩하지 않고 런타임에 x 좌표로 정렬합니다.
EYE_PAIR_A = (33, 133)
EYE_PAIR_B = (362, 263)

# --- EAR 6점 세트 (Soukupova & Cech). 베이스라인이 우리와 같은 메시를 쓰도록
#     여기 함께 둡니다. 같은 상수가 두 파일에 살면 언젠가 갈라집니다. ---
EAR_EYE_A = {"out": 33, "in": 133, "top1": 160, "bot1": 144, "top2": 158, "bot2": 153}
EAR_EYE_B = {"out": 263, "in": 362, "top1": 385, "bot1": 380, "top2": 387, "bot2": 373}

COORDS_PIXEL = "pixel"
COORDS_NORM = "norm"


@dataclass
class CropMeta:
    """크롭 1장의 기하 감사 정보. index.npz 의 f_* 필드로 그대로 들어갑니다.

    이걸 남기는 이유: 나중에 재식별이 높게 나왔을 때 '신원인가 전처리 인공물인가'를
    **재추출 없이** 확인하기 위해서입니다. 예를 들어 사용자마다 보간 커널이 다르면
    (`interp_cubic`) 그 자체가 프로브가 주울 수 있는 신호입니다.
    """

    span_px: float          # 두 눈 중심 사이 거리
    tilt_deg: float         # 회전 정렬 각도
    crop_w_px: float        # 원본에서 잘라낸 상자 폭
    crop_h_px: float
    interp_cubic: bool      # True = 업스케일(INTER_CUBIC), False = 다운스케일(INTER_AREA)
    padded: bool            # 상자가 화면 밖으로 나가 반사 패딩을 썼는가
    center_x: float
    center_y: float

    def as_dict(self) -> dict:
        return asdict(self)


def mesh_to_pixels(mesh: np.ndarray, w: int, h: int, coords: str) -> np.ndarray:
    """(N, 2|3) 메시 -> (N, 2) 픽셀 좌표."""
    if coords not in (COORDS_PIXEL, COORDS_NORM):
        raise ValueError(f"coords must be {COORDS_PIXEL!r} or {COORDS_NORM!r}, got {coords!r}")
    xy = np.asarray(mesh, np.float32)[:, :2]
    if coords == COORDS_NORM:
        xy = xy * np.array([w, h], np.float32)
    return xy


def eye_corners(mesh_xy: np.ndarray) -> tuple[np.ndarray, ...]:
    """픽셀 메시 -> (le_a, le_b, re_a, re_b). le_* 가 항상 **이미지 왼쪽** 눈입니다.

    두 쌍을 평균 x 로 정렬하므로 미러링 여부와 MediaPipe 의 좌/우 명명에 무관합니다.
    """
    a1, a2 = mesh_xy[EYE_PAIR_A[0]], mesh_xy[EYE_PAIR_A[1]]
    b1, b2 = mesh_xy[EYE_PAIR_B[0]], mesh_xy[EYE_PAIR_B[1]]
    if (a1[0] + a2[0]) <= (b1[0] + b2[0]):
        return a1, a2, b1, b2
    return b1, b2, a1, a2


def crop_both_eyes(frame_bgr: np.ndarray, mesh_xy: np.ndarray,
                   out_h: int = OUT_H, out_w: int = OUT_W,
                   margin: float = MARGIN) -> tuple[np.ndarray | None, CropMeta | None]:
    """양눈 크롭 -> (그레이 uint8 (out_h, out_w), CropMeta) 또는 (None, None).

    절차
      1. 두 눈 중심의 중점을 기준점으로
      2. 두 눈 중심을 잇는 선이 수평이 되도록 회전 (기울기는 ±90도로 정규화해 눈이 뒤집히지 않게)
      3. 크롭 폭 = 두 눈 중심 거리 x margin, 높이는 출력 종횡비에 맞춤
      4. 그레이스케일 -> (out_w, out_h) 리사이즈

    상자가 화면 안에 다 들어오면 **상자만** warp 합니다(전체 프레임을 돌리면 60배 낭비).
    화면을 벗어나면 전체를 warp 한 뒤 반사 패딩합니다 — 빠른 경로로는 그 경계를
    재현할 수 없기 때문입니다.
    """
    h, w = frame_bgr.shape[:2]
    le_a, le_b, re_a, re_b = eye_corners(mesh_xy)
    le = (le_a + le_b) / 2.0
    re = (re_a + re_b) / 2.0
    center = (le + re) / 2.0
    span = float(np.linalg.norm(le - re))
    if not np.isfinite(span) or span < MIN_SPAN_PX:
        return None, None

    crop_w = span * margin
    crop_h = crop_w * out_h / out_w

    x0 = int(round(center[0] - crop_w / 2)); y0 = int(round(center[1] - crop_h / 2))
    x1 = int(round(center[0] + crop_w / 2)); y1 = int(round(center[1] + crop_h / 2))
    pad = max(0, -x0, -y0, x1 - w, y1 - h)

    d = re - le
    angle = float(np.degrees(np.arctan2(d[1], d[0])))
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180
    M = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle, 1.0)

    if pad == 0:
        M = M.copy()
        M[0, 2] -= x0
        M[1, 2] -= y0
        box = cv2.warpAffine(frame_bgr, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
    else:
        img = cv2.warpAffine(frame_bgr, M, (w, h), flags=cv2.INTER_LINEAR)
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        box = img[y0 + pad:y1 + pad, x0 + pad:x1 + pad]

    if box.size == 0:
        return None, None
    gray = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY) if box.ndim == 3 else box
    cubic = gray.shape[1] < out_w
    out = cv2.resize(gray, (out_w, out_h),
                     interpolation=cv2.INTER_CUBIC if cubic else cv2.INTER_AREA)
    meta = CropMeta(span_px=span, tilt_deg=angle, crop_w_px=float(x1 - x0),
                    crop_h_px=float(y1 - y0), interp_cubic=bool(cubic),
                    padded=bool(pad > 0), center_x=float(center[0]),
                    center_y=float(center[1]))
    return out, meta


def ear_one_eye(mesh_xy: np.ndarray, idx: dict) -> float:
    """한쪽 눈 EAR. 베이스라인이 **우리와 같은 메시**를 쓰도록 여기 둡니다."""
    p_out, p_in = mesh_xy[idx["out"]], mesh_xy[idx["in"]]
    t1, b1 = mesh_xy[idx["top1"]], mesh_xy[idx["bot1"]]
    t2, b2 = mesh_xy[idx["top2"]], mesh_xy[idx["bot2"]]
    horiz = float(np.linalg.norm(p_out - p_in)) + 1e-6
    vert = float(np.linalg.norm(t1 - b1)) + float(np.linalg.norm(t2 - b2))
    return float(vert / (2.0 * horiz))


def ear_both(mesh_xy: np.ndarray) -> dict[str, float]:
    """-> {left, right, mean, min}. left/right 는 **이미지 기준**으로 정렬합니다."""
    a = ear_one_eye(mesh_xy, EAR_EYE_A)
    b = ear_one_eye(mesh_xy, EAR_EYE_B)
    xa = (mesh_xy[EAR_EYE_A["out"], 0] + mesh_xy[EAR_EYE_A["in"], 0]) / 2
    xb = (mesh_xy[EAR_EYE_B["out"], 0] + mesh_xy[EAR_EYE_B["in"], 0]) / 2
    left, right = (a, b) if xa <= xb else (b, a)
    return {"left": left, "right": right,
            "mean": (left + right) / 2.0, "min": min(left, right)}


def photometrics(gray: np.ndarray) -> dict[str, float]:
    """크롭 1장의 광학 지표. Phase 1 의 (밝기, 선명도) 프로브 입력입니다."""
    g = np.asarray(gray)
    return {"brightness": float(g.mean()), "contrast": float(g.std()),
            "sharpness": float(cv2.Laplacian(g, cv2.CV_64F).var())}


def to_input_tensor(gray: np.ndarray) -> np.ndarray:
    """(H, W) uint8 -> (1, 1, H, W) float32 [0,1].

    학습과 **정확히** 같아야 합니다. 지금은 /255 만 합니다 — 밝기 정규화 도입 여부는
    Phase 1 결과로 정하는 미결 항목이며, 넣게 되면 이 함수 한 곳만 고칩니다.
    """
    x = np.asarray(gray, np.float32) / 255.0
    if x.ndim != 2:
        raise ValueError(f"2-D 그레이 크롭이어야 합니다. got {x.shape}")
    return x.reshape(1, 1, *x.shape)
