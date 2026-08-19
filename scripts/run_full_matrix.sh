#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_ROOT/src"
export HF_HOME="$PROJECT_ROOT/.hf_cache"
export TORCH_HOME="$PROJECT_ROOT/.torch_cache"
export TRITON_CACHE_DIR="$PROJECT_ROOT/.triton_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

CONFIG="${1:-configs/pilot.yaml}"
ROOT="artifacts/pilot"
APPROVAL="$ROOT/APPROVE_FULL_MATRIX"
if [[ ! -f "$APPROVAL" ]] || [[ "$(tr -d '\r\n' < "$APPROVAL")" != "I reviewed Stage 1 and authorize nine additional adapters" ]]; then
  echo "Full matrix not authorized. Read docs/LAMBDA_RUNBOOK.md and create the exact approval file after reviewing Stage 1." >&2
  exit 4
fi

python -m under_extinction --config "$CONFIG" gate \
  --report "$ROOT/analysis/stage1_report.json" --require P

scripts/monitor_gpu.sh artifacts/full_matrix_gpu_telemetry.csv &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

for SEED in 29 47 71; do
  for CONTROLLER in intended proxy cached; do
    RUN_DIR="$ROOT/runs/${CONTROLLER}_seed${SEED}"
    if [[ ! -f "$RUN_DIR/COMPLETE" ]]; then
      RESUME_ARGS=()
      if [[ -f "$RUN_DIR/STOPPED_BUDGET" || -f "$RUN_DIR/STOPPED_EARLY" ]]; then
        RESUME_ARGS=(--resume)
      elif [[ -f "$RUN_DIR/FAILED" || -f "$RUN_DIR/RUNNING" ]]; then
        echo "Run $RUN_DIR is failed or ambiguously running; inspect it before retrying." >&2
        exit 5
      fi
      scripts/run_budgeted.sh 120 python -m under_extinction --config "$CONFIG" train \
        --controller "$CONTROLLER" --seed "$SEED" --run-dir "$RUN_DIR" "${RESUME_ARGS[@]}"
    fi
    if [[ ! -f "$RUN_DIR/COMPLETE" ]]; then
      echo "Refusing to evaluate incomplete run $RUN_DIR." >&2
      exit 6
    fi
    PREDICTION_PATH="$ROOT/predictions/${CONTROLLER}_seed${SEED}.jsonl"
    if [[ ! -f "$PREDICTION_PATH" ]]; then
      scripts/run_budgeted.sh 60 python -m under_extinction --config "$CONFIG" evaluate \
        --adapter "$RUN_DIR/final_adapter" --controller "$CONTROLLER" --seed "$SEED" \
        --destination "$PREDICTION_PATH"
    fi
  done
done

python -m under_extinction --config "$CONFIG" merge \
  --inputs "$ROOT"/predictions/intended_seed*.jsonl "$ROOT"/predictions/proxy_seed*.jsonl "$ROOT"/predictions/cached_seed*.jsonl \
  --destination "$ROOT/predictions/full_merged.jsonl"
python -m under_extinction --config "$CONFIG" analyze \
  --predictions "$ROOT/predictions/full_merged.jsonl" \
  --destination "$ROOT/analysis/full_report.json"
python -m under_extinction --config "$CONFIG" gate \
  --report "$ROOT/analysis/full_report.json" --require A B

echo "Formal synthetic matrix passed Gates A and B. This still does not pass paper-critical Gate C."
