#!/usr/bin/env bash
# Optional: ours_vpres / vdrop / gap, 3 blocks (~1.4 h). Candidates keep their own names.
set -euo pipefail
cd "$(dirname "$0")/.."
SID="${1:-SECONDARY_v1}"
python -m src.v2.deploy.paced_session --protocol protocol.json --bundle-root bundle \
  --session-id "$SID" --set secondary
python -m src.v2.deploy.check_paced_results --session-dir "results/pi_paced/$SID" --protocol protocol.json
