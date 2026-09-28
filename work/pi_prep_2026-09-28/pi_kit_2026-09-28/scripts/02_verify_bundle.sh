#!/usr/bin/env bash
# Device check of the corrected ONNX bundle (hashes, loading, fixtures, missing policy).
set -euo pipefail
cd "$(dirname "$0")/../bundle"
python verify_bundle.py --root . --out pi_model_verification.json
