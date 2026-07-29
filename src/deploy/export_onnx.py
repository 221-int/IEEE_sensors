"""
export_onnx.py

Export the deployed models to ONNX for the Pi5 edge benchmark.
  - encoder.onnx   : eye crop (1x1x64x160) -> 128-d vector   (what's deployed/stored)
  - pipeline.onnx  : eye crop -> blink probability (encoder + classifier fused)

Latency of these ONNX graphs is data-independent (depends on architecture +
input size + hardware), so this stays valid across dataset changes.

Run (desktop):
    pip install torch onnx
    python -m src.deploy.export_onnx --encoder models/autoencoder/encoder.pt \
        --classifier models/classifier/classifier.pt --out models/onnx
"""

import argparse
import json
import os

import torch
import torch.nn as nn

from src.encoder.train_autoencoder import Encoder
from src.classifier.train_classifier import MLP


class Pipeline(nn.Module):
    """eye crop -> blink probability (encoder + MLP fused)."""

    def __init__(self, encoder, clf):
        super().__init__()
        self.encoder = encoder
        self.clf = clf

    def forward(self, x):
        z = self.encoder(x)
        return torch.sigmoid(self.clf(z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="models/autoencoder/encoder.pt")
    ap.add_argument("--classifier", default="models/classifier/classifier.pt")
    ap.add_argument("--config", default="models/autoencoder/config.json")
    ap.add_argument("--out", default="models/onnx")
    ap.add_argument("--h", type=int, default=64)
    ap.add_argument("--w", type=int, default=160)
    ap.add_argument("--opset", type=int, default=13)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    latent = json.load(open(args.config))["latent"] if os.path.exists(args.config) else 128
    enc = Encoder(latent, in_hw=(args.h, args.w))
    enc.load_state_dict(torch.load(args.encoder, map_location="cpu"))
    enc.eval()
    clf = MLP(latent)
    clf.load_state_dict(torch.load(args.classifier, map_location="cpu"))
    clf.eval()

    dummy = torch.randn(1, 1, args.h, args.w)
    dyn = {"input": {0: "batch"}}

    enc_path = os.path.join(args.out, "encoder.onnx")
    torch.onnx.export(enc, dummy, enc_path, opset_version=args.opset,
                      input_names=["input"], output_names=["vector"],
                      dynamic_axes={**dyn, "vector": {0: "batch"}})

    pipe = Pipeline(enc, clf).eval()
    pipe_path = os.path.join(args.out, "pipeline.onnx")
    torch.onnx.export(pipe, dummy, pipe_path, opset_version=args.opset,
                      input_names=["input"], output_names=["blink_prob"],
                      dynamic_axes={**dyn, "blink_prob": {0: "batch"}})

    print(f"Exported:\n  {enc_path}  (input 1x1x{args.h}x{args.w} -> {latent}-d vector)")
    print(f"  {pipe_path}  (input -> blink prob)")
    print("Next: python -m src.deploy.bench_latency --model", pipe_path)


if __name__ == "__main__":
    main()
