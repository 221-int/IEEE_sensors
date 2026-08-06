"""train/serve 동치 게이트 — **이 검사를 통과하기 전에는 Pi 를 재지 않는다.**

    python -m src.v2.deploy.check_equivalence

왜 이 게이트가 있는가
--------------------
배포 경로가 학습과 어긋나면 **지연은 정상으로 나오고 판정만 조용히 망가진다.**
지연 숫자는 그럴듯하게 나오므로 아무도 눈치채지 못한다. 가장 흔한 원인이
입력 정규화다 — v1 은 `/255`, v2 는 `frame_standardize` 다. v1 프론트엔드를
그대로 가져다 쓰면 정확히 이 사고가 난다.

네 개의 게이트
-------------
  G-Q1 정규화     `batch_input`(학습, 3-D 축약) vs `to_input_tensor`(배포, 2-D 축약)
                  수식은 같지만 축약 순서가 달라 부동소수점이 갈릴 수 있다
  G-Q2 위임       프론트엔드가 기하를 **다시 구현하지 않았는가**
                  (같은 인자로 `crop_both_eyes` 를 직접 부른 것과 동일해야 한다)
  G-Q3 종단       🔴 **결정적 게이트.** 원본 mEBAL2 프레임을 배포 경로에 통과시켜
                  **학습이 저장해 둔 크롭**과 비트 단위로 같은지 본다
  G-Q4 ONNX       크롭 -> 텐서 -> ONNX 인코더 가 torch 와 같은 벡터를 내는가

G-Q3 만이 "정말 같은 그림을 보는가"에 답한다. 나머지는 부분 검사다.

랜드마크 출처는 별개다
--------------------
학습은 mEBAL2 제공 메시, Pi 는 MediaPipe 다. **기하 코드가 같아도 필드에서 크롭이
비트 동일할 수는 없다.** 이 게이트는 **코드 경로**를 검증하며, 출처 차이는
한계로 따로 기록한다(측정하려면 같은 프레임에 두 검출기를 돌려 비교해야 한다).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from src.v2.common import repro
from src.v2.dataset import crop as C
from src.v2.deploy.frontend import EyeFrontend, mediapipe_available

OUT = "results/v2/check_equivalence.json"
MEMMAP = "data/processed/v2/frames_m22.npy"
SHARD = "data/processed/v2/shards/u{:02d}.npz"


def g_q1(n: int, seed: int = 0) -> dict:
    """학습 정규화와 배포 정규화가 **비트 동일**한가."""
    a = np.load(MEMMAP, mmap_mode="r")
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(a.shape[0], n, replace=False))
    crops = np.asarray(a[idx])
    tr = C.batch_input(crops)
    de = np.concatenate([C.to_input_tensor(c) for c in crops])
    d = np.abs(tr - de)
    exact = int((d.reshape(len(crops), -1).max(1) == 0).sum())
    # 균일 크롭(std=0) 경계 — EPS_STD 로 막혀 있어야 하고 양쪽이 같아야 한다
    flat = np.full((1, C.OUT_H, C.OUT_W), 128, np.uint8)
    edge = float(np.abs(C.batch_input(flat) - C.to_input_tensor(flat[0])).max())
    return {"gate": "G-Q1 정규화", "n": int(n), "max_abs_diff": float(d.max()),
            "n_bit_identical": exact, "uniform_crop_diff": edge,
            "input_norm": C.INPUT_NORM,
            "pass": bool(d.max() == 0.0 and exact == len(crops) and edge == 0.0)}


def g_q2(n: int = 64, seed: int = 0) -> dict:
    """프론트엔드가 `crop_both_eyes` 에 **위임만** 하는가 (재구현 금지)."""
    rng = np.random.default_rng(seed)
    fe = EyeFrontend()
    diffs, cases, none_match = [], 0, 0
    for _ in range(n):
        h, w = 480, 640
        frame = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        mesh = rng.uniform(0, 1, (468, 2)).astype(np.float32) * [w, h]
        # 경계 조건을 일부러 만든다: 화면 밖(패딩 경로), 아주 작은 span(cubic 경로)
        cx = rng.uniform(-40, w + 40)
        cy = rng.uniform(-40, h + 40)
        span = rng.choice([12.0, 40.0, 200.0])
        mesh[33] = [cx - span / 2, cy]; mesh[133] = [cx - span / 2 + 4, cy]
        mesh[362] = [cx + span / 2, cy]; mesh[263] = [cx + span / 2 + 4, cy]
        direct, _ = C.crop_both_eyes(frame, mesh, C.OUT_H, C.OUT_W, C.MARGIN)
        via, _ = fe.crop_from_mesh(frame, mesh)
        if direct is None or via is None:
            none_match += int((direct is None) == (via is None))
            continue
        cases += 1
        diffs.append(float(np.abs(direct.astype(np.int32) - via.astype(np.int32)).max()))
    mx = max(diffs) if diffs else 0.0
    return {"gate": "G-Q2 위임", "n_cases": cases, "n_none_agree": none_match,
            "max_abs_diff": mx, "pass": bool(mx == 0.0 and cases > 0)}


def g_q3(user: int, n_frames: int, out_h: int, out_w: int) -> dict:
    """🔴 결정적 — 원본 프레임 -> 배포 경로 크롭 == 학습이 저장한 크롭."""
    from src.v2 import phase2_extract as P
    from src.v2.dataset import mebal2 as M
    import zipfile

    sh = np.load(SHARD.format(user), allow_pickle=True)
    f_idx = sh["f_frame_idx"].astype(int)
    stored = sh["frames_m22"]
    order = np.argsort(f_idx)[:n_frames]          # 앞쪽 프레임만 (순차 디코딩이라)
    rows = order[np.argsort(f_idx[order])]
    need = {int(f_idx[r]) for r in rows}

    with zipfile.ZipFile(P.PD_ZIP) as z:
        with z.open(f"Processed_Data/User {user}/box.csv") as fh:
            _, box = M.scan_stream(fh, need, 4, 1)
        with z.open(f"Processed_Data/User {user}/landmarks.csv") as fh:
            _, lm = M.scan_stream(fh, need, 3, M.N_MESH)

    sel = {}
    for f in sorted(need):
        s = M.select_face(f, lm.get(f), box.get(f))
        if s.status == M.OK:
            sel[f] = s

    fe = EyeFrontend(out_h=out_h, out_w=out_w)
    row_of = {int(f_idx[r]): int(r) for r in rows}
    diffs, tdiffs, checked = [], [], 0
    with P.VideoSource(user) as vpath:
        for idx, frame in M.iter_frames(vpath, sorted(sel)):
            xy = sel[idx].mesh[:, :2]
            got, _ = fe.crop_from_mesh(frame, xy)     # 배포 경로
            if got is None:
                continue
            ref = stored[row_of[idx]]                 # 학습이 저장한 것
            diffs.append(int(np.abs(got.astype(np.int32) - ref.astype(np.int32)).max()))
            # 입력 텐서까지 비교한다 (크롭이 같아도 정규화에서 갈릴 수 있다)
            tdiffs.append(float(np.abs(C.to_input_tensor(got)
                                       - C.batch_input(ref[None])).max()))
            checked += 1
    mx = max(diffs) if diffs else None
    tmx = max(tdiffs) if tdiffs else None
    return {"gate": "G-Q3 종단(원본 프레임 -> 저장된 학습 크롭)", "user": user,
            "n_checked": checked, "crop_max_abs_diff": mx,
            "tensor_max_abs_diff": tmx,
            "n_bit_identical": int(sum(d == 0 for d in diffs)),
            "pass": bool(checked > 0 and mx == 0 and tmx == 0.0)}


def g_q4(onnx_json: str, n: int = 32, seed: int = 0) -> dict:
    """크롭 -> 텐서 -> ONNX 인코더 == torch 인코더."""
    import onnxruntime as ort
    import torch
    from src.v2.deploy.export_onnx import load_models

    if not os.path.exists(onnx_json):
        return {"gate": "G-Q4 ONNX", "pass": False, "reason": f"{onnx_json} 없음"}
    meta = json.load(open(onnx_json, encoding="utf-8"))
    enc, _, arch, d = load_models(meta["ckpt"])

    a = np.load(MEMMAP, mmap_mode="r")
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(a.shape[0], n, replace=False))
    x = C.batch_input(np.asarray(a[idx]))
    with torch.no_grad():
        t = enc(torch.from_numpy(x)).numpy()
    s = ort.InferenceSession(meta["paths"]["encoder"], providers=["CPUExecutionProvider"])
    o = s.run(None, {"crop": x})[0]
    mx = float(np.abs(t - o).max())
    return {"gate": "G-Q4 ONNX", "arch": arch, "d_latent": d, "n": int(n),
            "max_abs_diff": mx, "tol": 1e-4, "pass": bool(mx < 1e-4)}


def main() -> int:
    repro.ensure_hashseed()
    repro.seal(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-norm", type=int, default=2048)
    ap.add_argument("--user", type=int, default=1)
    ap.add_argument("--n-frames", type=int, default=12)
    ap.add_argument("--onnx-json", default="results/v2/export_onnx.json")
    ap.add_argument("--skip-video", action="store_true",
                    help="G-Q3 를 건너뛴다. **그러면 Pi 측정 허가가 나지 않는다**")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    gates = [g_q1(args.n_norm), g_q2()]
    if args.skip_video:
        gates.append({"gate": "G-Q3 종단", "pass": None, "skipped": True,
                      "reason": "--skip-video"})
    else:
        gates.append(g_q3(args.user, args.n_frames, C.OUT_H, C.OUT_W))
    gates.append(g_q4(args.onnx_json))

    for g in gates:
        mark = "SKIP" if g.get("skipped") else ("PASS" if g["pass"] else "FAIL")
        print(f"  [{mark}] {g['gate']}")
        for k, v in g.items():
            if k not in ("gate", "pass", "skipped"):
                print(f"          {k}: {v}")

    required = [g for g in gates if not g.get("skipped")]
    ok = all(g["pass"] for g in required)
    decisive = next(g for g in gates if g["gate"].startswith("G-Q3"))
    authorized = bool(ok and not decisive.get("skipped"))

    out = {"env": repro.env_fingerprint(), "gates": gates,
           "all_required_pass": ok,
           "pi_measurement_authorized": authorized,
           "_rule": "G-Q3 를 통과하지 않으면 Pi 를 재지 않는다. "
                    "배포 경로가 어긋나면 지연은 정상으로 나오고 판정만 망가진다.",
           "_landmark_source_caveat":
               "학습은 mEBAL2 제공 메시, Pi 는 MediaPipe 다. 기하 코드가 같아도 "
               "필드 크롭은 비트 동일할 수 없다. 이 게이트는 코드 경로만 검증한다.",
           "mediapipe_available": mediapipe_available()}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    os.replace(tmp, args.out)

    print(f"\n  Pi 측정 허가: {'예' if authorized else '아니오'}")
    print(f"  -> {args.out}")
    return 0 if authorized else 1


if __name__ == "__main__":
    raise SystemExit(main())
