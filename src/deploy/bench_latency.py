"""
bench_latency.py

Measure per-frame inference latency of an ONNX model with onnxruntime.
Run on the desktop first (sanity) and then on the Pi5 (the real number).

Reports mean / median / p95 latency and implied fps for the model graph
(encoder+classifier). Note the FULL frame budget also includes eye
detection+crop (MediaPipe FaceMesh, measured ~25 ms/frame earlier), which runs
outside this graph. Target: crop + this graph <= 66 ms (15 fps).

Run:
    pip install onnxruntime numpy
    python -m src.deploy.bench_latency --model models/onnx/pipeline.onnx --iters 500
"""

import argparse
import time

import numpy as np
import onnxruntime as ort


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/onnx/pipeline.onnx")
    ap.add_argument("--h", type=int, default=64)
    ap.add_argument("--w", type=int, default=160)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--crop-ms", type=float, default=25.0,
                    help="assumed detection+crop cost per frame (outside the graph)")
    args = ap.parse_args()

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    x = np.random.rand(1, 1, args.h, args.w).astype(np.float32)

    for _ in range(args.warmup):
        sess.run(None, {name: x})

    ts = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts = np.array(ts)

    mean, med, p95 = ts.mean(), np.median(ts), np.percentile(ts, 95)
    print(f"model: {args.model}  (providers: {sess.get_providers()})")
    print(f"graph latency/frame:  mean {mean:.2f} ms | median {med:.2f} ms | p95 {p95:.2f} ms")
    print(f"graph-only fps: {1000.0/med:.1f}")
    total = med + args.crop_ms
    print(f"\nwith detection+crop (~{args.crop_ms:.0f} ms): ~{total:.1f} ms/frame "
          f"-> ~{1000.0/total:.1f} fps")
    budget = 66.0
    status = "WITHIN" if total <= budget else "OVER"
    print(f"budget {budget:.0f} ms (15 fps): {status} "
          f"({'margin' if total <= budget else 'excess'} {abs(budget-total):.1f} ms)")


if __name__ == "__main__":
    main()
