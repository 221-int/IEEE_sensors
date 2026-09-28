#!/usr/bin/env bash
# Short NON-REPORTABLE checks on the Pi:
#  (a) corrected free-run entry points with the new contracts (run_video, 10 s each)
#  (b) one shortened paced session of the primary set (exploratory overrides)
# Nothing here is a paper number.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/pi_paced/smoke_run_video
for m in ours image_cnn_head; do
  python -m src.v2.deploy.run_video --mode "$m" --source clips/eyeblink8_v9_cam.avi \
    --onnx-dir bundle/onnx/fold0_seed0 --intra-threads 2 --no-spin --duration 10 \
    --out "results/pi_paced/smoke_run_video/run_video_${m}.json"
done
python -m src.v2.deploy.paced_session --protocol protocol.json --bundle-root bundle \
  --session-id SMOKE --set primary --exploratory \
  --override warmup_s=5.0 settle_s=2.0 idle_pre_s=15.0 active_s=30.0 idle_post_s=10.0 \
             repeats=1 cooldown_min_s=30 reference_idle_s=10 cooldown_max_extra_s=60
# QC without protocol conformance (the overrides differ from the protocol on purpose):
python -m src.v2.deploy.check_paced_results --session-dir results/pi_paced/SMOKE \
  --out results/pi_paced/SMOKE/smoke_check.json
