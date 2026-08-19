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
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1

CONFIG="$(realpath "${1:-configs/bridge_pilot.yaml}")"

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
LOG_PATH="$LOG_DIR/replication_${STAMP}.log"
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
mapfile -t ALL_CONFIG_SEEDS < <(config_list bridge.seeds)
if (( ${#OBJECTIVES[@]} != 2 || ${#ALL_CONFIG_SEEDS[@]} < 2 )); then
  echo "Bridge replication requires two objectives and at least one replication seed." >&2
  exit 2
fi
STAGE1_SEED="${ALL_CONFIG_SEEDS[0]}"
REPLICATION_SEEDS=("${ALL_CONFIG_SEEDS[@]:1}")
LOCKED_SPLIT="$(config_scalar bridge.splits.locked)"
TRAIN_MINUTES="$(config_scalar budget.replication_train_minutes_per_objective)"
EVAL_MINUTES="$(config_scalar budget.replication_eval_minutes_per_objective)"
RUN_ROOT="$ROOT/runs"
PREDICTION_ROOT="$ROOT/predictions/replication_locked_test"
ANALYSIS_ROOT="$ROOT/analysis"
STAGE1_REPORT="$ANALYSIS_ROOT/stage1_dev_report.json"
APPROVAL="$ROOT/APPROVE_BRIDGE_REPLICATION"
APPROVAL_TEXT="I reviewed bridge DEV Stage 1 and authorize locked TEST replication"

if [[ ! -f "$APPROVAL" ]] || [[ "$(tr -d '\r\n' < "$APPROVAL")" != "$APPROVAL_TEXT" ]]; then
  echo "Replication is not authorized. Review DEV Stage 1 and create the exact approval file described in docs/LAMBDA_RUNBOOK.md." >&2
  exit 4
fi
if [[ ! -f "$STAGE1_REPORT" ]]; then
  echo "Missing bridge Stage 1 report: $STAGE1_REPORT" >&2
  exit 4
fi
budgeted 5 python -m under_extinction --config "$CONFIG" bridge-gate \
  --report "$STAGE1_REPORT" --require stage1

mkdir -p "$RUN_ROOT" "$PREDICTION_ROOT" "$ANALYSIS_ROOT"
"$PROJECT_ROOT/scripts/monitor_gpu.sh" "$ROOT/replication_gpu_telemetry_${STAMP}.csv" &
MONITOR_PID=$!
cleanup() {
  kill "$MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT

for SEED in "${REPLICATION_SEEDS[@]}"; do
  for OBJECTIVE in "${OBJECTIVES[@]}"; do
    RUN_DIR="$RUN_ROOT/${OBJECTIVE}_seed${SEED}"
    if [[ ! -f "$RUN_DIR/COMPLETE" ]]; then
      RESUME_ARGS=()
      if [[ -f "$RUN_DIR/STOPPED_BUDGET" || -f "$RUN_DIR/STOPPED_EARLY" ]]; then
        RESUME_ARGS=(--resume)
      elif [[ -f "$RUN_DIR/FAILED" || -f "$RUN_DIR/RUNNING" ]]; then
        echo "Run $RUN_DIR is failed or ambiguously running; inspect it before retrying." >&2
        exit 5
      fi
      budgeted "$TRAIN_MINUTES" python -m under_extinction --config "$CONFIG" bridge-train \
        --objective "$OBJECTIVE" --seed "$SEED" --run-dir "$RUN_DIR" "${RESUME_ARGS[@]}"
    fi
    if [[ ! -f "$RUN_DIR/COMPLETE" ]]; then
      echo "Incomplete bridge replication run: $RUN_DIR" >&2
      exit 6
    fi
  done
done

ALL_SEEDS=("$STAGE1_SEED" "${REPLICATION_SEEDS[@]}")
PREDICTIONS=()
for SEED in "${ALL_SEEDS[@]}"; do
  for OBJECTIVE in "${OBJECTIVES[@]}"; do
    RUN_DIR="$RUN_ROOT/${OBJECTIVE}_seed${SEED}"
    if [[ ! -f "$RUN_DIR/COMPLETE" ]]; then
      echo "The complete paired matrix is missing $RUN_DIR; refusing to open locked $LOCKED_SPLIT." >&2
      exit 6
    fi
    PREDICTION_PATH="$PREDICTION_ROOT/${OBJECTIVE}_seed${SEED}_${LOCKED_SPLIT}.jsonl"
    if [[ ! -f "$PREDICTION_PATH" ]]; then
      budgeted "$EVAL_MINUTES" python -m under_extinction --config "$CONFIG" bridge-evaluate \
        --run-dir "$RUN_DIR" --split "$LOCKED_SPLIT" --unlock-test --destination "$PREDICTION_PATH"
    fi
    PREDICTIONS+=("$PREDICTION_PATH")
  done
done

REPORT="$ANALYSIS_ROOT/replication_locked_test_report.json"
budgeted 30 python -m under_extinction --config "$CONFIG" bridge-analyze \
  --predictions "${PREDICTIONS[@]}" --split "$LOCKED_SPLIT" --destination "$REPORT"
budgeted 5 python -m under_extinction --config "$CONFIG" bridge-gate \
  --report "$REPORT" --require replication

echo "Bridge locked-TEST replication passed its preregistered gate. Retrieve all artifacts before termination."
echo "Log: $LOG_PATH"
