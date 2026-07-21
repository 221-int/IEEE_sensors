"""sr_eval — 자체 4인 영상 기반 SR 효과 평가 (합성 열화 → SR on/off 비교).

【이번 라운드 스코프(STAGE2_결정사항 §2·§6)】 공개셋 `.tag` F1 벤치는 미룬다.
깨끗한 자체 영상을 참조(reference)로 두고, 표준 열화(다운스케일/감마/노이즈)를 씌워
"열화 → SR off(정확도↓) → SR on(회복)" 곡선을 만든다. 정답은 큐 기반(이후 연결).

【열화 표준】 config.DEGRADE_DOWNSCALE / DEGRADE_GAMMA / DEGRADE_NOISE_SIGMA.

【스캐폴딩 상태 2026-07-20】
  - 열화 함수(degrade_*)는 실제 구현(순수 cv2/numpy) — 지금 바로 검증 가능.
  - SR on 경로는 frontend.EarFrontend 의 two-pass(★1 타임스탬프 정책) 확정에 의존.
    그전까지 --sr on 은 NotImplementedError 로 명확히 막힌다(조용한 오동작 금지).
  - 정답 정합(큐 기반 truth)·F1/정밀도 계산은 TODO(다음 라운드).

실행 (src/ 에서):
    python -m scripts.sr_eval --video clean_clip.mp4 --sr off
    python -m scripts.sr_eval --video clean_clip.mp4 --sr on   # ★1 확정 후
"""
import argparse
import itertools

import numpy as np

from eyeblink import config
from eyeblink import landmarks as L
from eyeblink.std import BlinkSTD


# ── 합성 열화 (순수 cv2/numpy, 실제 구현) ────────────────────────────────────
def degrade_downscale(frame, factor):
    """factor 배 실제 축소 → 얼굴/눈 픽셀 수를 줄여 '멀다/저해상도'를 모사.

    dims 가 실제로 작아지므로 w_eye 가 줄고 SR 게이트가 발동한다(핵심).
    factor<=1 이면 원본 그대로.
    """
    import cv2
    if not factor or factor <= 1:
        return frame
    h, w = frame.shape[:2]
    return cv2.resize(frame, (max(1, w // factor), max(1, h // factor)),
                      interpolation=cv2.INTER_AREA)


def degrade_gamma(frame, gamma):
    """감마(<1 = 어둡게). out = 255*(in/255)^(1/gamma). gamma==1 이면 무동작."""
    if not gamma or abs(gamma - 1.0) < 1e-6:
        return frame
    inv = 1.0 / float(gamma)
    lut = (((np.arange(256) / 255.0) ** inv) * 255.0).clip(0, 255).astype(np.uint8)
    import cv2
    return cv2.LUT(frame, lut)


def degrade_noise(frame, sigma):
    """가우시안 노이즈(8bit 표준편차). sigma<=0 이면 무동작."""
    if not sigma or sigma <= 0:
        return frame
    noise = np.random.normal(0.0, float(sigma), frame.shape)
    return (frame.astype(np.float32) + noise).clip(0, 255).astype(np.uint8)


def apply_degrade(frame, downscale=1, gamma=1.0, noise_sigma=0.0):
    """열화 파이프라인: 축소 → 감마 → 노이즈."""
    f = degrade_downscale(frame, downscale)
    f = degrade_gamma(f, gamma)
    f = degrade_noise(f, noise_sigma)
    return f


# ── EAR 추출 (SR off / on) — frontend 경유 ──────────────────────────────────
def extract_ears(video, degrade_params, use_sr):
    """열화 적용 영상에서 (times, ears) 추출.

    use_sr=False : 랜드마커만(현 파이프라인과 동일 경로).
    use_sr=True  : frontend.EarFrontend two-pass — ★1 확정 전엔 NotImplementedError.

    TODO(다음 라운드): frontend 로 완전 일원화(off/on 모두 EarFrontend 로).
    """
    import cv2
    if not L.mediapipe_available():
        raise SystemExit("mediapipe 가 필요합니다:  pip install mediapipe")

    from eyeblink.frontend import EarFrontend
    from eyeblink.robust import SuperResolution

    landmarker = L.build_landmarker()
    sr = SuperResolution() if use_sr else None
    frontend = EarFrontend(landmarker=landmarker, sr=sr)

    cap = cv2.VideoCapture(video)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    times, ears, n_sr = [], [], 0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = apply_degrade(frame, **degrade_params)
        r = frontend.process(frame, int(i * 1000 / native_fps))   # SR-on: ★1 전엔 예외
        if r.has_face and r.ear is not None:
            times.append(i / native_fps)
            ears.append(r.ear)
            n_sr += int(r.used_sr)
        i += 1
    cap.release()
    return np.array(times), np.array(ears), native_fps, n_sr


def auto_calib(ears, open_pct=85, closed_pct=3):
    return float(np.percentile(ears, open_pct)), float(np.percentile(ears, closed_pct))


def count_blinks(times, ears, baseline, floor):
    std = BlinkSTD(baseline, floor)
    return sum(1 for t, e in zip(times, ears)
               if std.update(float(t), float(e)) is not None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="깨끗한 자체 영상(참조)")
    ap.add_argument("--sr", choices=["off", "on"], default="off")
    ap.add_argument("--truth", type=int, default=None,
                    help="큐 기반 정답 깜빡임 수(있으면 오차 계산). TODO: 큐 로그 연동")
    a = ap.parse_args()
    use_sr = (a.sr == "on")

    # 참조(무열화)로 캘리브 기준 확보
    t0, e0, native, _ = extract_ears(a.video, dict(downscale=1, gamma=1.0, noise_sigma=0.0),
                                     use_sr=False)
    baseline, floor = auto_calib(e0)
    ref_blinks = count_blinks(t0, e0, baseline, floor)
    print(f"[sr_eval] native_fps={native:.1f}  ref(무열화) blinks={ref_blinks}"
          + (f"  truth={a.truth}" if a.truth is not None else ""))
    print(f"[sr_eval] calib baseline={baseline:.3f} floor={floor:.3f}  sr={a.sr}\n")

    # 열화 그리드 스윕
    grid = itertools.product(config.DEGRADE_DOWNSCALE, config.DEGRADE_GAMMA,
                             config.DEGRADE_NOISE_SIGMA)
    print(f"{'down':>5}{'gamma':>7}{'noise':>7}{'frames':>8}{'blinks':>8}"
          f"{'sr_used':>9}{'d_ref':>7}")
    for ds, gm, ns in grid:
        params = dict(downscale=ds, gamma=gm, noise_sigma=ns)
        t, e, _, n_sr = extract_ears(a.video, params, use_sr=use_sr)
        if len(e) == 0:
            print(f"{ds:>5}{gm:>7.1f}{ns:>7}{0:>8}{'(무검출)':>8}")
            continue
        n = count_blinks(t, e, baseline, floor)     # 참조 캘리브 재사용(공정 비교)
        d_ref = n - ref_blinks
        print(f"{ds:>5}{gm:>7.1f}{ns:>7}{len(e):>8}{n:>8}{n_sr:>9}{d_ref:>+7d}")

    # TODO(다음 라운드): 큐 기반 truth 정합 → 프레임단위 TP/FP/FN → F1/정밀도/재현율,
    #   그리고 SR off vs on 을 같은 표에 나란히(회복 곡선). fps/전력은 run_live --log 로.


if __name__ == "__main__":
    main()
