#!/usr/bin/env bash
set -u
cd /home/ubuntu/coupling_clarification_source_20260904T0905Z
export PYTHONPATH="$PWD/src:$PWD"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for pair in small large; do
  root="/home/ubuntu/coupling_clarification_${pair}_20260904T0905Z"
  if [ -e "$root" ] || [ -e "$root.log" ]; then exit 73; fi
  /home/ubuntu/Align_Paper/.venv-interaction/bin/python -m interaction_sprint.gpu_coupling_runner "$root" --config "configs/gpu_coupling_clarification_${pair}_v1.json" > "$root.log" 2>&1
  result=$?
  printf '%s\n' "$result" > "$root.exit"
  if [ "$result" -ne 0 ]; then exit "$result"; fi
done
