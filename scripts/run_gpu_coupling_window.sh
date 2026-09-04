#!/usr/bin/env bash
set -euo pipefail
SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${COUPLING_RUN_ROOT:?fresh absolute run root required}"
PYTHON=/home/ubuntu/Align_Paper/.venv-interaction/bin/python
[[ "$RUN_ROOT" == /home/ubuntu/gpu_coupling_* && ! -e "$RUN_ROOT" && ! -e "${RUN_ROOT}.exit" ]] || exit 2
cd "$SOURCE_ROOT"
export PYTHONPATH="$SOURCE_ROOT/src:$SOURCE_ROOT"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
trap 'result=$?; printf "%s\n" "$result" > "${RUN_ROOT}.exit"' EXIT
"$PYTHON" -m interaction_sprint.gpu_coupling_runner "$RUN_ROOT"
