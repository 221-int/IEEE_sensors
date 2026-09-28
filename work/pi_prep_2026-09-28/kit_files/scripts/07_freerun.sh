#!/usr/bin/env bash
# Throughput diagnostic (no 30 fps pacing). Never pooled with paced results.
set -euo pipefail
cd "$(dirname "$0")/.."
SID="${1:-FREERUN_v1}"
python -m src.v2.deploy.paced_session --protocol protocol.json --bundle-root bundle \
  --session-id "$SID" --set freerun
python -m src.v2.deploy.check_paced_results --session-dir "results/pi_paced/$SID" --protocol protocol.json
