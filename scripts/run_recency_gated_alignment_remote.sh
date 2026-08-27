#!/usr/bin/env bash
set -euo pipefail

# Execute G0 only after a compatible GPU runtime and public model cache are
# available.  The runner refuses an existing output directory, writes evidence
# per seed before the final decision, and uses only benign synthetic labels.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RECENCY_GATED_RUN_ROOT:-$PROJECT_ROOT/results/recency_gated_$(date -u +%Y%m%dT%H%M%SZ)}"
VENV_ROOT="${RECENCY_GATED_RUNTIME_VENV:?Set RECENCY_GATED_RUNTIME_VENV to a verified GPU virtual environment}"
HF_CACHE="${RECENCY_GATED_HF_HOME:?Set RECENCY_GATED_HF_HOME to the model cache root}"

[[ ! -e "$RUN_ROOT" ]] || { echo "Refusing to overwrite result directory: $RUN_ROOT" >&2; exit 2; }
[[ -f "$VENV_ROOT/bin/activate" ]] || { echo "Missing virtual environment: $VENV_ROOT" >&2; exit 2; }

export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJECT_ROOT/src"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME="$HF_CACHE"
export TORCH_HOME="$HF_CACHE/torch"
export TRITON_CACHE_DIR="$HF_CACHE/triton"

# shellcheck disable=SC1090
source "$VENV_ROOT/bin/activate"
cd "$PROJECT_ROOT"
python -m recency_gated_alignment.runner \
  --config "$PROJECT_ROOT/configs/recency_gated_alignment.yaml" \
  --output "$RUN_ROOT"
