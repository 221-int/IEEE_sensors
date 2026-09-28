#!/usr/bin/env bash
# Records versions / governor / throttle / fan / 3 raw PMIC texts. Changes nothing.
set -euo pipefail
cd "$(dirname "$0")/.."
python -c 'import sys; v=sys.version_info; assert (v.major,v.minor) in ((3,11),(3,12)), sys.version; print("python", sys.version.split()[0])'
python -m src.v2.deploy.pi_env_check --out results/pi_paced/env_check.json
