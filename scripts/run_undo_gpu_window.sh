#!/usr/bin/env bash
set -euo pipefail
SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${UNDO_RUN_ROOT:?fresh absolute root required}"
PREPARED="${UNDO_PREPARED:?prepared data root required}"
[[ "$RUN_ROOT" == /home/ubuntu/undo_relation_* && ! -e "$RUN_ROOT" && ! -e "${RUN_ROOT}.exit" ]] || exit 2
cd "$SOURCE_ROOT"
export PYTHONPATH="$SOURCE_ROOT/src:$SOURCE_ROOT"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
trap 'result=$?; printf "%s\n" "$result" > "${RUN_ROOT}.exit"' EXIT
/home/ubuntu/Align_Paper/.venv-interaction/bin/python -m interaction_sprint.undo_gpu_training "$PREPARED" "$RUN_ROOT" \
  --model-path /home/ubuntu/Align_Paper/.hf_cache/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --epochs 2 --batch-size 4 --eval-batch-size 4 --max-tokens 4096 --lr 0.0001 --seed 9047701
