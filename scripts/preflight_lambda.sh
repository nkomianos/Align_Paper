#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_ROOT/src"
export HF_HOME="$PROJECT_ROOT/.hf_cache"
export TORCH_HOME="$PROJECT_ROOT/.torch_cache"
export TRITON_CACHE_DIR="$PROJECT_ROOT/.triton_cache"
export TOKENIZERS_PARALLELISM=false

CONFIG="${1:-configs/pilot.yaml}"
if [[ -d frozen_data ]]; then
  python -m under_extinction --config "$CONFIG" install-data --source frozen_data
else
  python -m under_extinction --config "$CONFIG" build
fi
python -m under_extinction --config "$CONFIG" dry-run

mkdir -p artifacts
scripts/monitor_gpu.sh artifacts/preflight_gpu_telemetry.csv &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

scripts/run_budgeted.sh 45 python -m under_extinction --config "$CONFIG" preflight
echo "Preflight passed. Inspect artifacts/pilot/preflight/preflight.json before Stage 1."
