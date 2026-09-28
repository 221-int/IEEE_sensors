"""표1 추가작업(BiLSTM/LSTM) 전력·지연 실측용 ONNX export.

`export_onnx.py`의 `export_image_cnn()`과 동일한 원칙이다 — **가중치는
무작위 초기화다.** BiLSTM/LSTM 체크포인트(`models/v2/train_image_cnn_*_final/`)는
이 머신에 없고, 지연/전력은 구조·입력 크기·하드웨어로만 결정되므로 무작위
가중치로도 유효하다(`export_onnx.py` §image_cnn 절 참고). 이 그래프로
정확도를 재면 안 된다 — 정확도는 `train_image_cnn_bilstm_final.json` /
`train_image_cnn_lstm_merged.json`을 쓴다.

출력 위치는 `run_video.py`/`run_video_power.py`가 `--mode`를 그대로 폴더명으로
쓰는 규칙(`sub = os.path.join(onnx_dir, mode)`)을 따른다:

    models/v2/onnx/image_cnn_bilstm/{backbone.onnx, head.onnx}
    models/v2/onnx/image_cnn_lstm/{backbone.onnx, head.onnx}

사용법
------
    python -m src.v2.deploy.export_onnx_rnn
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from src.v2.common import repro
from src.v2.deploy.export_onnx import HeadWrap, sizes
from src.v2.dataset import crop as C
from src.v2.model import encoder as E
from src.v2.model import rnn_head as R

EVENT_LEN = 19
OUTDIR = "models/v2/onnx"
OUT = "results/v2/export_onnx_rnn.json"


def export_one(head_type: str, d_latent: int, outdir: str, opset: int,
               tol: float) -> dict:
    import onnxruntime as ort

    mode = f"image_cnn_{head_type}"          # 폴더명 = --mode 값
    sub = os.path.join(outdir, mode)
    os.makedirs(sub, exist_ok=True)

    net = E.build_image_cnn(d_latent).eval()
    x = torch.randn(1, 1, C.OUT_H, C.OUT_W)
    p_bb = os.path.join(sub, "backbone.onnx")
    torch.onnx.export(net, (x,), p_bb, opset_version=opset,
                      input_names=["crop"], output_names=["feat"],
                      dynamic_axes={"crop": {0: "batch"}, "feat": {0: "batch"}},
                      dynamo=False)

    bidir = head_type == "bilstm"
    head = R.build_rnn_head(d_latent, EVENT_LEN, hidden=16, layers=1,
                            bidirectional=bidir).eval()
    z, m = torch.randn(1, EVENT_LEN, d_latent), torch.ones(1, EVENT_LEN)
    p_h = os.path.join(sub, "head.onnx")
    torch.onnx.export(HeadWrap(head).eval(), (z, m), p_h, opset_version=opset,
                      input_names=["vectors", "mask"], output_names=["blink_prob"],
                      dynamic_axes={"vectors": {0: "batch"}, "mask": {0: "batch"},
                                    "blink_prob": {0: "batch"}},
                      dynamo=False)
    paths = {"backbone": p_bb, "head": p_h}

    # 수치 검증 — torch 와 어긋나면 "빠른 오답"을 재게 된다
    rng = np.random.default_rng(0)
    xb = rng.standard_normal((4, 1, C.OUT_H, C.OUT_W)).astype(np.float32)
    with torch.no_grad():
        t_bb = net(torch.from_numpy(xb)).numpy()
    s_bb = ort.InferenceSession(p_bb, providers=["CPUExecutionProvider"])
    d_bb = float(np.max(np.abs(t_bb - s_bb.run(None, {"crop": xb})[0])))

    zb = rng.standard_normal((4, EVENT_LEN, d_latent)).astype(np.float32)
    mb = np.ones((4, EVENT_LEN), np.float32)
    with torch.no_grad():
        t_h = torch.sigmoid(head(torch.from_numpy(zb), torch.from_numpy(mb))).numpy()
    s_h = ort.InferenceSession(p_h, providers=["CPUExecutionProvider"])
    d_h = float(np.max(np.abs(t_h - s_h.run(None, {"vectors": zb, "mask": mb})[0])))

    # 결측 극단(마스크 전부 0)도 확인 — TCN 헤드 export 검증과 동일 항목
    m0 = np.zeros((2, EVENT_LEN), np.float32)
    z0 = rng.standard_normal((2, EVENT_LEN, d_latent)).astype(np.float32)
    with torch.no_grad():
        t0 = torch.sigmoid(head(torch.from_numpy(z0), torch.from_numpy(m0))).numpy()
    o0 = s_h.run(None, {"vectors": z0, "mask": m0})[0]
    d_mask0 = float(np.max(np.abs(t0 - o0)))
    finite0 = bool(np.all(np.isfinite(o0)))

    ver = {"backbone_max_abs_diff": d_bb, "head_max_abs_diff": d_h,
          "head_allzero_mask_max_abs_diff": d_mask0,
          "head_allzero_mask_finite": finite0}
    ver["pass"] = (d_bb < tol and d_h < tol and d_mask0 < tol and finite0)

    a = E.analyse_image_cnn(d_latent)
    n_bb = sum(q.numel() for q in net.parameters())
    n_h = sum(q.numel() for q in head.parameters())
    head_mmac = R.rnn_head_mmac(d_latent, 16, EVENT_LEN, 1, bidir)
    return {
        "mode": mode, "head_type": head_type, "d_latent": d_latent,
        "bidirectional_within_window_only": bidir,
        "paths": {k: v.replace("\\", "/") for k, v in paths.items()},
        "weights": "random (latency/power only)",
        "_weights_warning": "무작위 초기화다. 지연/전력은 유효하지만 이 그래프의 "
                            "판정 출력(probs)은 의미가 없다. 정확도는 "
                            "results/v2/train_image_cnn_{head_type}*.json 을 쓴다.",
        "params": {"backbone": n_bb, "head": n_h, "total": n_bb + n_h},
        "mmac": {"backbone_per_frame": a["total_mmac"],
                 "head_per_frame_stride1": head_mmac,
                 "total_per_frame_stride1": a["total_mmac"] + head_mmac},
        "file_sizes": sizes(paths), "verification": ver,
    }


def main() -> int:
    repro.ensure_hashseed()
    repro.seal(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--latent", type=int, default=16)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    out = {"env": repro.env_fingerprint()}
    ok = True
    for head_type in ("bilstm", "lstm"):
        r = export_one(head_type, args.latent, args.outdir, args.opset, args.tol)
        out[head_type] = r
        print(f"[{r['mode']}] params {r['params']['total']:,}  "
              f"MMAC/frame {r['mmac']['total_per_frame_stride1']:.2f}  "
              f"검증 {'PASS' if r['verification']['pass'] else 'FAIL'}")
        ok = ok and r["verification"]["pass"]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    os.replace(tmp, args.out)
    print(f"  -> {args.out}")

    if not ok:
        print("\n🔴 ONNX 출력이 torch 와 다릅니다. 이 그래프로 Pi 지연/전력을 재면 "
              "'빠른 오답'을 재는 것입니다. 배포하지 마십시오.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
