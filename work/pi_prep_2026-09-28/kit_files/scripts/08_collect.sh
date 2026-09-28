#!/usr/bin/env bash
# Pack everything produced on the Pi for transfer back to the PC.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="pi_results_$(hostname)_$(date +%Y%m%d_%H%M%S).tgz"
tar czf "$OUT" results/pi_paced bundle/pi_model_verification.json kit_manifest.json protocol.json
sha256sum "$OUT" | tee "$OUT.sha256"
