#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
if [[ ! -f .venv/bin/activate ]]; then
  echo "Missing .venv; run scripts/bootstrap_lambda.sh first." >&2
  exit 2
fi
source .venv/bin/activate
export PYTHONPATH="$PROJECT_ROOT/src"
export HF_HOME="$PROJECT_ROOT/.hf_cache"
export TORCH_HOME="$PROJECT_ROOT/.torch_cache"
export TRITON_CACHE_DIR="$PROJECT_ROOT/.triton_cache"
export TOKENIZERS_PARALLELISM=false

CONFIG="$(realpath "${1:-configs/bridge_smoke.yaml}")"
STAGE1_CONFIG="$(realpath "${2:-configs/bridge_pilot.yaml}")"

config_scalar() {
  python - "$CONFIG" "$1" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as handle:
    value = yaml.safe_load(handle)
for component in sys.argv[2].split("."):
    value = value[component]
if isinstance(value, bool):
    print(str(value).lower())
elif isinstance(value, (str, int, float)):
    print(value)
else:
    raise SystemExit(f"Config value {sys.argv[2]} is not scalar")
PY
}

config_list() {
  python - "$CONFIG" "$1" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as handle:
    value = yaml.safe_load(handle)
for component in sys.argv[2].split("."):
    value = value[component]
if not isinstance(value, list):
    raise SystemExit(f"Config value {sys.argv[2]} is not a list")
for item in value:
    print(item)
PY
}

RAW_ROOT="$(config_scalar output_root)"
if [[ "$RAW_ROOT" = /* ]]; then
  ROOT="$RAW_ROOT"
else
  ROOT="$PROJECT_ROOT/$RAW_ROOT"
fi
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_PATH="$LOG_DIR/preflight_${STAMP}.log"
exec > >(tee -a "$LOG_PATH") 2>&1

validate_deadline() {
  if [[ ! "${UE_HARD_DEADLINE_EPOCH:-}" =~ ^[0-9]+$ ]]; then
    echo "Set UE_HARD_DEADLINE_EPOCH to the watchdog's absolute Unix termination deadline." >&2
    exit 2
  fi
  RETRIEVAL_RESERVE_MINUTES="$(config_scalar budget.retrieval_reserve_minutes)"
  if [[ ! "$RETRIEVAL_RESERVE_MINUTES" =~ ^[0-9]+$ ]] || (( RETRIEVAL_RESERVE_MINUTES < 30 )); then
    echo "budget.retrieval_reserve_minutes must be an integer of at least 30." >&2
    exit 2
  fi
  NOW_EPOCH="$(date +%s)"
  PAID_HARD_DEADLINE_EPOCH="$UE_HARD_DEADLINE_EPOCH"
  COMPUTE_DEADLINE_EPOCH="$((PAID_HARD_DEADLINE_EPOCH - RETRIEVAL_RESERVE_MINUTES * 60))"
  if (( COMPUTE_DEADLINE_EPOCH <= NOW_EPOCH )); then
    echo "The hard deadline leaves no compute time before the ${RETRIEVAL_RESERVE_MINUTES}-minute retrieval reserve." >&2
    exit 124
  fi
  export UE_COMPUTE_DEADLINE_EPOCH="$COMPUTE_DEADLINE_EPOCH"
  export UE_TERMINATION_DEADLINE_EPOCH="$PAID_HARD_DEADLINE_EPOCH"
  echo "Termination deadline: $PAID_HARD_DEADLINE_EPOCH; compute cutoff: $COMPUTE_DEADLINE_EPOCH; retrieval reserve: ${RETRIEVAL_RESERVE_MINUTES}m."
}

validate_instance_contract() {
  if [[ ! "${UE_INSTANCE_ID:-}" =~ ^[A-Za-z0-9._:-]+$ ]]; then
    echo "Set UE_INSTANCE_ID to the exact provider instance ID." >&2
    exit 2
  fi
  if [[ ! "${UE_HOURLY_USD:-}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Set UE_HOURLY_USD to the numeric provider-console hourly rate." >&2
    exit 2
  fi
  CONFIG_HOURLY_USD="$(config_scalar budget.hourly_usd)"
  python - "$UE_HOURLY_USD" "$CONFIG_HOURLY_USD" <<'PY'
import decimal
import sys

observed = decimal.Decimal(sys.argv[1])
configured = decimal.Decimal(sys.argv[2])
if observed != configured:
    raise SystemExit(
        f"Provider hourly rate {observed} does not equal frozen config budget.hourly_usd {configured}"
    )
PY
  if [[ -z "${UE_INSTANCE_LAUNCHED_AT:-}" ]]; then
    echo "Set UE_INSTANCE_LAUNCHED_AT to the provider launch time in RFC 3339 UTC form." >&2
    exit 2
  fi
  UE_INSTANCE_START_EPOCH="$(date --date="$UE_INSTANCE_LAUNCHED_AT" +%s 2>/dev/null)" || {
    echo "UE_INSTANCE_LAUNCHED_AT is not parseable by GNU date: $UE_INSTANCE_LAUNCHED_AT" >&2
    exit 2
  }
  NOW_EPOCH="$(date +%s)"
  if [[ ! "$UE_INSTANCE_START_EPOCH" =~ ^[0-9]+$ ]] || (( UE_INSTANCE_START_EPOCH > NOW_EPOCH + 300 )); then
    echo "UE_INSTANCE_LAUNCHED_AT resolves to an invalid or future launch time." >&2
    exit 2
  fi
  export UE_INSTANCE_START_EPOCH
  echo "Billing contract: instance=$UE_INSTANCE_ID; start=$UE_INSTANCE_START_EPOCH; hourly_usd=$UE_HOURLY_USD."
}

budgeted() {
  local max_minutes="$1"
  shift
  UE_HARD_DEADLINE_EPOCH="$COMPUTE_DEADLINE_EPOCH" \
    "$PROJECT_ROOT/scripts/run_budgeted.sh" "$max_minutes" "$@"
}

validate_instance_contract
validate_deadline
mapfile -t OBJECTIVES < <(config_list bridge.objectives)
mapfile -t SEEDS < <(config_list bridge.seeds)
if (( ${#OBJECTIVES[@]} != 2 || ${#SEEDS[@]} == 0 )); then
  echo "Bridge preflight requires exactly two objectives and at least one seed." >&2
  exit 2
fi
SEED="${SEEDS[0]}"
DEV_SPLIT="$(config_scalar bridge.splits.development)"
TRAIN_MINUTES="$(config_scalar budget.preflight_train_minutes_per_objective)"
EVAL_MINUTES="$(config_scalar budget.preflight_eval_minutes_per_objective)"
RUN_ROOT="$ROOT/preflight/$STAMP"
mkdir -p "$RUN_ROOT/runs" "$RUN_ROOT/predictions"

nvidia-smi
python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("The GPU does not support BF16")
print({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)})
PY

"$PROJECT_ROOT/scripts/monitor_gpu.sh" "$RUN_ROOT/gpu_telemetry.csv" &
MONITOR_PID=$!
cleanup() {
  if [[ -n "${MONITOR_PID:-}" ]]; then
    kill "$MONITOR_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

budgeted 10 python -m under_extinction --config "$CONFIG" bridge-build
budgeted 10 python -m under_extinction --config "$CONFIG" bridge-oracle \
  --split "$DEV_SPLIT" --destination "$RUN_ROOT/oracle_${DEV_SPLIT}.jsonl"

PREDICTIONS=()
for OBJECTIVE in "${OBJECTIVES[@]}"; do
  RUN_DIR="$RUN_ROOT/runs/${OBJECTIVE}_seed${SEED}"
  PREDICTION_PATH="$RUN_ROOT/predictions/${OBJECTIVE}_seed${SEED}_dev.jsonl"
  budgeted "$TRAIN_MINUTES" python -m under_extinction --config "$CONFIG" bridge-train \
    --objective "$OBJECTIVE" --seed "$SEED" --run-dir "$RUN_DIR"
  if [[ ! -f "$RUN_DIR/COMPLETE" ]]; then
    echo "Bridge smoke training did not complete: $RUN_DIR" >&2
    exit 6
  fi
  budgeted "$EVAL_MINUTES" python -m under_extinction --config "$CONFIG" bridge-evaluate \
    --run-dir "$RUN_DIR" --split "$DEV_SPLIT" --destination "$PREDICTION_PATH"
  PREDICTIONS+=("$PREDICTION_PATH")
done

BASE_CONTROL_PATH="$RUN_ROOT/predictions/unchanged_base_seed${SEED}_dev.jsonl"
budgeted "$EVAL_MINUTES" python -m under_extinction --config "$CONFIG" bridge-evaluate \
  --run-dir "$RUN_ROOT/runs/${OBJECTIVES[0]}_seed${SEED}" --split "$DEV_SPLIT" \
  --unchanged-base --destination "$BASE_CONTROL_PATH"

REPORT="$RUN_ROOT/smoke_report.json"
budgeted 15 python -m under_extinction --config "$CONFIG" bridge-analyze \
  --predictions "${PREDICTIONS[@]}" --split "$DEV_SPLIT" --base-control "$BASE_CONTROL_PATH" \
  --destination "$REPORT"
budgeted 5 python -m under_extinction --config "$CONFIG" bridge-gate \
  --report "$REPORT" --require smoke
# Freeze telemetry before it is hashed into the Stage-1 authorization.  Leaving
# the monitor alive for another sample would correctly invalidate that artifact.
kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
MONITOR_PID=""
PREFLIGHT_PASS_PATH="$(python -m under_extinction --config "$CONFIG" bridge-preflight-attest \
  --stage1-config "$STAGE1_CONFIG" --report "$REPORT")"

echo "Bridge preflight passed. This validates mechanics and GPU compatibility only; it is not scientific evidence."
echo "Hash-bound Stage 1 handoff: $PREFLIGHT_PASS_PATH"
echo "Log: $LOG_PATH"
