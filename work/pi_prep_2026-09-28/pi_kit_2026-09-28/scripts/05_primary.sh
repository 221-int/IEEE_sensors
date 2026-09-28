#!/usr/bin/env bash
# Reportable primary session: ours_vpres / image_cnn_head / ear_rule, 5 blocks (~2.3 h).
# Re-running the same command resumes; finished runs are kept, the plan is never reshuffled.
set -euo pipefail
cd "$(dirname "$0")/.."
SID="${1:-PRIMARY_v1}"
python -m src.v2.deploy.paced_session --protocol protocol.json --bundle-root bundle \
  --session-id "$SID" --set primary
python -m src.v2.deploy.check_paced_results --session-dir "results/pi_paced/$SID" --protocol protocol.json
