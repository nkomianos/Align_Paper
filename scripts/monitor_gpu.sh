#!/usr/bin/env bash
set -euo pipefail

DESTINATION="${1:-artifacts/gpu_telemetry.csv}"
mkdir -p "$(dirname "$DESTINATION")"
while true; do
  nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits >> "$DESTINATION"
  sleep 60
done
