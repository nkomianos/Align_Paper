#!/usr/bin/env bash
set -euo pipefail
SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${SDPO_RUN_ROOT:?fresh absolute root required}"
[[ "$RUN_ROOT" == /home/ubuntu/sdpo_format_* && ! -e "$RUN_ROOT" && ! -e "${RUN_ROOT}.exit" ]] || exit 2
cd "$SOURCE_ROOT"
export PYTHONPATH="$SOURCE_ROOT/src:$SOURCE_ROOT"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
trap 'result=$?; printf "%s\n" "$result" > "${RUN_ROOT}.exit"' EXIT
/home/ubuntu/Align_Paper/.venv-interaction/bin/python -m interaction_sprint.sdpo_format_positive_control \
  /home/ubuntu/sdpo_inputs/sdpo_format_control_v2 "$RUN_ROOT" \
  --model-path /home/ubuntu/Align_Paper/.hf_cache/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --upstream-file /home/ubuntu/sdpo_inputs/online_sdpo_updater.py --threads 4 --lr 0.0001 --seed 9047801
