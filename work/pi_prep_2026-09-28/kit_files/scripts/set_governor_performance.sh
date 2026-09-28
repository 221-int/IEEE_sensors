#!/usr/bin/env bash
# Run manually (needs sudo). Reverts on reboot, so run after every boot.
set -euo pipefail
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
vcgencmd get_throttled
