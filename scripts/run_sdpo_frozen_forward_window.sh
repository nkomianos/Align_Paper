#!/usr/bin/env bash
set -euo pipefail
SOURCE=/home/ubuntu/sdpo_frozen_forward_source_20260904T1005Z
ROOT=/home/ubuntu/sdpo_frozen_forward_20260904T1005Z
test ! -e "$ROOT"
test ! -e "$ROOT.exit"
export PYTHONPATH="$SOURCE/src:$SOURCE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
cd "$SOURCE"
set +e
timeout --signal=TERM --kill-after=60s 2700s /home/ubuntu/Align_Paper/.venv-interaction/bin/python -u -m interaction_sprint.sdpo_frozen_forward_diagnostic \
  /home/ubuntu/sdpo_format_v3_20260904T0915Z \
  /home/ubuntu/sdpo_v3_inputs_20260904T0915Z "$ROOT" \
  --model-path /home/ubuntu/Align_Paper/.hf_cache/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c --threads 4
result=$?
set -e
printf '%s\n' "$result" > "$ROOT.exit"
exit "$result"
