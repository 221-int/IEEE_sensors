#!/usr/bin/env bash
# Unit tests of pacing / drop / seams / QC / PMIC parsing (no camera, no models).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m unittest src.v2.deploy.test_paced_bench src.v2.deploy.test_pmic_power
