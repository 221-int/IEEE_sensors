#!/usr/bin/env bash
# Kit integrity (hashes of every copied file). Run first after unzipping.
set -euo pipefail
cd "$(dirname "$0")/.."
python verify_kit.py
