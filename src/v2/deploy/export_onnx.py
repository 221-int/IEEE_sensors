"""v2 ONNX export — T5-3. 겸해서 T5-1(파라미터)·T5-2(MMAC)·T5-4(파일 크기)를 낸다.

    python -m src.v2.deploy.export_onnx
    python -m src.v2.deploy.export_onnx --ckpt models/v2/fold0_seed0 --out models/v2/onnx

v1 과 무엇이 다른가 — **그래프를 합치지 않는다**
----------------------------------------------
v1 (`src/deploy/export_onnx.py`) 은 encoder + classifier 를 `pipeline.onnx` 하나로
합쳤다. 한 프레임이 들어가 확률 하나가 나오는 구조라 그게 맞았다.

v2 는 판정 단위가 **19프레임 이벤트**다. 그런데 프레임마다 19장을 다시 인코딩하면
비용이 19배가 되어 Pi 예산을 넘는다(`model/encoder.py` 설계근거 3). 그래서 배포 형태가
**프레임당 인코딩 1회 + 벡터 19개 링버퍼 + 시간 헤드**이고, 그 형태를 그대로 내보낸다.

    encoder.onnx   (N, 1, 64, 160) -> (N, D)        매 프레임 1회
    head.onnx      (N, 19, D), (N, 19) -> (N,)      창이 찰 때 1회

→ **합친 그래프를 만들면 그 자체가 설계를 배신한다.** 필요하면 `--event-graph` 로
따로 낼 수 있게 두되 기본은 끈다(비교용이지 배포용이 아니다).

입력 규약 🔒
-----------
ONNX 입력은 **이미 정규화된** 크롭이다. 학습이 `crop.batch_input()` 으로
`(x - mean) / std` 를 적용한 뒤 모델에 넣기 때문이다(`crop.INPUT_NORM`).
Pi 하네스가 이 정규화를 빠뜨리면 그래프는 멀쩡히 돌면서 숫자만 틀린다.
그래서 `INPUT_NORM` 을 결과 JSON 에 박아 함께 배포한다.

수치 일치 검증
-------------
export 는 **조용히 틀릴 수 있다**(BatchNorm 이 train 모드로 나가거나, mask 경로가
다르게 접히거나). 그 상태로 Pi 지연을 재면 "빠른 오답"을 측정하게 된다.
그래서 export 직후 onnxruntime 으로 다시 돌려 torch 와 **최대 절대 오차**를 재고,
허용치를 넘으면 **0이 아닌 종료 코드로 죽는다**.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from src.v2.common import repro
from src.v2.dataset import crop as C
from src.v2.model import encoder as E

CKPT = "models/v2/fold0_seed0"
OUTDIR = "models/v2/onnx"
OUT = "results/v2/export_onnx.json"
EVENT_LEN = 19
TOL = 1e-4


# ------------------------------------------------------------------ 체크포인트
def infer_spec(enc_sd: dict) -> tuple[str, int]:
    """가중치 모양에서 (arch, D) 를 역추론한다. 인자로 받으면 틀린 값을 줄 수 있다."""
    d_latent, flat = enc_sd["fc.weight"].shape
    conv_shapes = [tuple(v.shape) for k, v in enc_sd.items()
                   if k.startswith("net.") and k.endswith(".weight") and v.dim() == 4]
    cands = []
    for name in E.SPECS:
        want = [(c.c_out, c.c_in, c.k[0], c.k[1]) for c in E.SPECS[name]]
        if want == conv_shapes and E.analyse(name, d_latent)["flat_dim"] == flat:
            cands.append(name)
    if len(cands) != 1:
        raise SystemExit(
            f"체크포인트의 구조를 특정하지 못했습니다. 후보 {cands}\n"
            f"  conv {conv_shapes}, flat_dim {flat}, D {d_latent}")
    return cands[0], int(d_latent)


def load_models(ckpt: str):
    enc_sd = torch.load(os.path.join(ckpt, "encoder.pt"), map_location="cpu")
    head_sd = torch.load(os.path.join(ckpt, "head.pt"), map_location="cpu")
    arch, d = infer_spec(enc_sd)
    enc = E.build(arch, d)
    enc.load_state_dict(enc_sd)
    head = E.build_head(d, EVENT_LEN)
    head.load_state_dict(head_sd)
    # 🔴 eval() 을 빠뜨리면 BatchNorm 이 배치 통계를 쓰는 그래프가 나간다.
    #    단일 프레임 추론에서 조용히 다른 숫자가 된다.
    enc.eval(); head.eval()
    return enc, head, arch, d


class HeadWrap(torch.nn.Module):
    """(z, mask) -> 확률. 헤드는 logit 을 내므로 sigmoid 까지 그래프에 넣는다."""

    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, z, mask):
        return torch.sigmoid(self.head(z, mask))


# ------------------------------------------------------------------ export
def export(enc, head, d: int, outdir: str, opset: int, event_graph: bool) -> dict:
    os.makedirs(outdir, exist_ok=True)
    paths = {}

    x = torch.randn(1, 1, C.OUT_H, C.OUT_W)
    p = os.path.join(outdir, "encoder.onnx")
    torch.onnx.export(enc, (x,), p, opset_version=opset,
                      input_names=["crop"], output_names=["vector"],
                      dynamic_axes={"crop": {0: "batch"}, "vector": {0: "batch"}},
                      dynamo=False)
    paths["encoder"] = p

    z = torch.randn(1, EVENT_LEN, d)
    m = torch.ones(1, EVENT_LEN)
    p = os.path.join(outdir, "head.onnx")
    torch.onnx.export(HeadWrap(head).eval(), (z, m), p, opset_version=opset,
                      input_names=["vectors", "mask"], output_names=["blink_prob"],
                      dynamic_axes={"vectors": {0: "batch"}, "mask": {0: "batch"},
                                    "blink_prob": {0: "batch"}},
                      dynamo=False)
    paths["head"] = p

    if event_graph:
        class Event(torch.nn.Module):
            def __init__(self, enc, head):
                super().__init__()
                self.enc, self.head = enc, head

            def forward(self, crops, mask):          # crops: (N, T, 1, H, W)
                n, t = crops.shape[:2]
                zz = self.enc(crops.reshape(n * t, *crops.shape[2:])).reshape(n, t, -1)
                return torch.sigmoid(self.head(zz, mask))

        xe = torch.randn(1, EVENT_LEN, 1, C.OUT_H, C.OUT_W)
        p = os.path.join(outdir, "event.onnx")
        torch.onnx.export(Event(enc, head).eval(), (xe, m), p, opset_version=opset,
                          input_names=["crops", "mask"], output_names=["blink_prob"],
                          dynamic_axes={"crops": {0: "batch"}, "mask": {0: "batch"},
                                        "blink_prob": {0: "batch"}},
                          dynamo=False)
        paths["event"] = p
    return paths


def verify(paths: dict, enc, head, d: int, tol: float) -> dict:
    """onnxruntime 으로 다시 돌려 torch 와 맞는지 본다. 안 맞으면 죽는다."""
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    rep = {}

    xb = rng.standard_normal((4, 1, C.OUT_H, C.OUT_W)).astype(np.float32)
    with torch.no_grad():
        t_out = enc(torch.from_numpy(xb)).numpy()
    s = ort.InferenceSession(paths["encoder"], providers=["CPUExecutionProvider"])
    o_out = s.run(None, {"crop": xb})[0]
    rep["encoder_max_abs_diff"] = float(np.max(np.abs(t_out - o_out)))

    zb = rng.standard_normal((4, EVENT_LEN, d)).astype(np.float32)
    mb = rng.integers(0, 2, (4, EVENT_LEN)).astype(np.float32)
    mb[:, 0] = 1.0                                  # 전부 0 인 창은 별도로 아래에서 본다
    with torch.no_grad():
        t_h = torch.sigmoid(head(torch.from_numpy(zb), torch.from_numpy(mb))).numpy()
    s = ort.InferenceSession(paths["head"], providers=["CPUExecutionProvider"])
    o_h = s.run(None, {"vectors": zb, "mask": mb})[0]
    rep["head_max_abs_diff"] = float(np.max(np.abs(t_h - o_h)))

    # 결측 극단: 마스크 전부 0. head 는 clamp/nan_to_num 으로 막고 있어야 한다.
    m0 = np.zeros((2, EVENT_LEN), np.float32)
    z0 = rng.standard_normal((2, EVENT_LEN, d)).astype(np.float32)
    with torch.no_grad():
        t0 = torch.sigmoid(head(torch.from_numpy(z0), torch.from_numpy(m0))).numpy()
    o0 = s.run(None, {"vectors": z0, "mask": m0})[0]
    rep["head_allzero_mask_max_abs_diff"] = float(np.max(np.abs(t0 - o0)))
    rep["head_allzero_mask_finite"] = bool(np.all(np.isfinite(o0)))

    rep["tol"] = tol
    rep["pass"] = (rep["encoder_max_abs_diff"] < tol
                   and rep["head_max_abs_diff"] < tol
                   and rep["head_allzero_mask_max_abs_diff"] < tol
                   and rep["head_allzero_mask_finite"])
    return rep


def sizes(paths: dict) -> dict:
    """.onnx + 같은 이름의 .onnx.data 를 합산한다 (가중치가 분리될 수 있다)."""
    out = {}
    for k, p in paths.items():
        n = os.path.getsize(p)
        ext = p + ".data"
        n += os.path.getsize(ext) if os.path.exists(ext) else 0
        out[k] = {"bytes": n, "kb": round(n / 1024, 1),
                  "external_data": os.path.exists(ext)}
    return out


# ------------------------------------------------------------------ main
def main() -> int:
    repro.ensure_hashseed()
    repro.seal(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument("--event-graph", action="store_true",
                    help="19프레임 통짜 그래프도 낸다. **배포용이 아니다** — "
                         "프레임당 비용이 19배다. 비교용으로만")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    enc, head, arch, d = load_models(args.ckpt)
    print(f"체크포인트 {args.ckpt}  ->  구조 {arch}  D={d} (가중치에서 역추론)")

    paths = export(enc, head, d, args.outdir, args.opset, args.event_graph)
    ver = verify(paths, enc, head, d, args.tol)
    sz = sizes(paths)

    # T5-1 파라미터 / T5-2 MMAC
    a = E.analyse(arch, d)
    n_enc = sum(p.numel() for p in enc.parameters())
    n_head = sum(p.numel() for p in head.parameters())
    head_mmac_event = E.temporal_head_mmac(d, EVENT_LEN)

    print(f"\n[T5-1 파라미터]  encoder {n_enc:,} + head {n_head:,} = {n_enc+n_head:,}")
    print(f"[T5-2 MMAC]      encoder {a['total_mmac']:.2f}/frame  "
          f"(conv {a['conv_mmac']:.2f} + fc {a['fc_mmac']:.3f})")
    print(f"                 head {head_mmac_event:.4f}/호출")
    print(f"                 **stride 1 (하네스 기본): head {head_mmac_event:.4f}/frame** "
          f"-> 합계 {a['total_mmac'] + head_mmac_event:.2f}/frame "
          f"(head 비중 {100*head_mmac_event/(a['total_mmac']+head_mmac_event):.2f}%)")
    print(f"                 (참고) stride 19 면 {head_mmac_event/EVENT_LEN:.4f}/frame")
    print(f"[T5-4 파일 크기] " + "  ".join(f"{k} {v['kb']}KB" for k, v in sz.items()))
    print(f"\n[수치 검증] encoder {ver['encoder_max_abs_diff']:.2e}  "
          f"head {ver['head_max_abs_diff']:.2e}  "
          f"mask=0 {ver['head_allzero_mask_max_abs_diff']:.2e}  "
          f"-> {'PASS' if ver['pass'] else 'FAIL'}")

    out = {"env": repro.env_fingerprint(), "ckpt": args.ckpt, "arch": arch,
           "d_latent": d, "opset": args.opset, "event_len": EVENT_LEN,
           "input_norm": C.INPUT_NORM, "eps_std": C.EPS_STD,
           "crop_hw": [C.OUT_H, C.OUT_W], "margin": C.MARGIN,
           "_input_contract": "ONNX 입력은 이미 (x-mean)/std 로 정규화된 크롭이다. "
                              "Pi 하네스가 같은 정규화를 적용해야 한다.",
           "_not_deployed": "event.onnx 는 배포용이 아니다. 프레임당 비용이 19배다.",
           "paths": {k: v.replace('\\', '/') for k, v in paths.items()},
           "params": {"encoder": n_enc, "head": n_head, "total": n_enc + n_head},
           "head_stride": 1,
           "mmac": {"encoder_per_frame": a["total_mmac"],
                    "encoder_conv": a["conv_mmac"], "encoder_fc": a["fc_mmac"],
                    "head_per_invocation": head_mmac_event,
                    # 🔴 하네스는 head 를 **매 프레임**(stride 1) 돌린다. 창이 찰 때마다
                    # 한 번(stride 19)이 아니다. 연속 검출에서 창 경계에서만 판정하면
                    # 깜빡임을 최대 18프레임 늦게 잡는다. 그래서 보고 기준은 stride 1 이다.
                    "head_per_frame_stride1": head_mmac_event,
                    "head_per_frame_stride19": head_mmac_event / EVENT_LEN,
                    "total_per_frame_stride1": a["total_mmac"] + head_mmac_event,
                    "head_share_of_total_stride1":
                        head_mmac_event / (a["total_mmac"] + head_mmac_event)},
           "_head_not_incrementally_cacheable":
               "TCN 이 padding=dilation 인 **대칭** Conv1d 이고 pooling·BatchNorm 이 창 "
               "전체에 걸리므로, stride 1 에서 직전 계산을 재사용할 수 없다. 매 프레임 "
               "창 19개를 다시 돈다 — 0.0459 MMAC/frame 은 그 값이다.",
           "file_sizes": sz, "verification": ver}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    os.replace(tmp, args.out)
    print(f"  -> {args.out}")

    if not ver["pass"]:
        print("\n🔴 ONNX 출력이 torch 와 다릅니다. 이 그래프로 Pi 지연을 재면 "
              "'빠른 오답'을 재는 것입니다. 배포하지 마십시오.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
