#!/usr/bin/env bash
set -euo pipefail
SOURCE=/home/ubuntu/ep_interface_source_20260904T1030Z
ROOT=/home/ubuntu/ep_interface_20260904T1030Z
test ! -e "$ROOT"
test ! -e "$ROOT.exit"
export PYTHONPATH="$SOURCE/src:$SOURCE"
export HF_HOME=/home/ubuntu/Align_Paper/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
cd "$SOURCE"
set +e
timeout --signal=TERM --kill-after=60s 1200s /home/ubuntu/Align_Paper/.venv-interaction/bin/python -u scripts/ep_interface_diagnostic.py run \
  --prepared /home/ubuntu/ep_interface_prepared_20260904T1030Z --output "$ROOT"
result=$?
set -e
printf '%s\n' "$result" > "$ROOT.exit"
exit "$result"
