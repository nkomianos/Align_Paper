#!/usr/bin/env bash
set -euo pipefail

# Run only from an extracted, locally verified DID-v1 deployment bundle.  This
# script deliberately has no resume mode: the evaluator requires a new output
# destination for every attempt, while the collector preserves failed/partial
# evidence for retrieval.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUNDLE_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"

if [[ "$(basename "$BUNDLE_ROOT")" != "under_extinction_dev_diag" ]]; then
  echo "run_dev_diag_remote.sh must remain inside under_extinction_dev_diag/project/scripts." >&2
  exit 2
fi

RUN_ID="${DID_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]]; then
  echo "DID_RUN_ID must contain only 1-64 safe filename characters." >&2
  exit 2
fi

WORK_BASE_RAW="${DID_WORK_BASE:-$(cd "$BUNDLE_ROOT/.." && pwd)/dev_diag_work}"
WORK_BASE="$(python3 - "$WORK_BASE_RAW" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"
SESSION_ROOT="$WORK_BASE/$RUN_ID"
EVIDENCE_ROOT="$SESSION_ROOT/evidence"
RUNTIME_ROOT="$SESSION_ROOT/runtime"
INFERENCE_ROOT="$EVIDENCE_ROOT/inference"
LOG_ROOT="$EVIDENCE_ROOT/logs"
LOG_PATH="$LOG_ROOT/remote.log"
RETRIEVAL_ROOT_RAW="${DID_RETRIEVAL_ROOT:-$WORK_BASE/retrieval}"
RETRIEVAL_ROOT="$(python3 - "$RETRIEVAL_ROOT_RAW" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"
RESULT_ARCHIVE="$RETRIEVAL_ROOT/dev_diag_results_${RUN_ID}.tar.gz"

for MUTABLE_ROOT in "$SESSION_ROOT" "$RETRIEVAL_ROOT"; do
  if [[ "$MUTABLE_ROOT" == "$BUNDLE_ROOT" || "$MUTABLE_ROOT" == "$BUNDLE_ROOT/"* ]]; then
    echo "Runtime and retrieval paths must remain outside the immutable bundle: $MUTABLE_ROOT" >&2
    exit 2
  fi
done

if [[ -e "$SESSION_ROOT" || -e "$RESULT_ARCHIVE" || -e "${RESULT_ARCHIVE}.sha256" ]]; then
  echo "Refusing to reuse an existing DID-v1 run or retrieval destination: $RUN_ID" >&2
  exit 2
fi
mkdir -p "$LOG_ROOT" "$RUNTIME_ROOT" "$RETRIEVAL_ROOT"

export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJECT_ROOT/src"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME="$RUNTIME_ROOT/hf_cache"
export TORCH_HOME="$RUNTIME_ROOT/torch_cache"
export TRITON_CACHE_DIR="$RUNTIME_ROOT/triton_cache"
export UE_DEV_DIAG_BUNDLE_ROOT="$BUNDLE_ROOT"
export UE_DEV_DIAG_BOOTSTRAP_ATTESTATION="$LOG_ROOT/bootstrap_runtime_attestation.json"

run_diagnostic() {
  set -euo pipefail

  # Standard-library verification happens before installing packages, creating
  # a model cache, or loading a model.
  python3 - "$BUNDLE_ROOT" <<'PY'
import sys
from under_extinction.dev_diag_deployment import verify_dev_diag_bundle

manifest = verify_dev_diag_bundle(sys.argv[1])
print({
    "bundle_kind": manifest["kind"],
    "bundle_inventory_sha256": manifest["inventory_sha256"],
    "bundle_file_count": manifest["file_count"],
    "allowed_split": manifest["contract"]["allowed_split"],
})
PY

  bash "$PROJECT_ROOT/scripts/bootstrap_dev_diag.sh" \
    "$PROJECT_ROOT" \
    "$BUNDLE_ROOT" \
    "$RUNTIME_ROOT" \
    "$LOG_ROOT/bootstrap_runtime_attestation.json"
  # shellcheck disable=SC1091
  source "$RUNTIME_ROOT/.venv/bin/activate"

  python - "$BUNDLE_ROOT" "$LOG_ROOT/bootstrap_runtime_attestation.json" \
    "$LOG_ROOT/bootstrap_binding_preflight.json" <<'PY'
import json
from pathlib import Path
import sys
from under_extinction.dev_diag_deployment import verify_dev_diag_bootstrap_attestation

binding = verify_dev_diag_bootstrap_attestation(sys.argv[2], sys.argv[1])
Path(sys.argv[3]).write_text(
    json.dumps(binding, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
    newline="\n",
)
print({"verified_bootstrap_binding_preflight": binding})
PY

  # The verified deployment payload is immutable during inference.  A second
  # full verification still runs after the evaluator, and its deterministic
  # binding must equal this preflight record.
  chmod -R a-w "$BUNDLE_ROOT"

  # Every scientific input path below is immutable inside the verified bundle.
  # The evaluator itself rehashes and validates them before loading any model.
  local evaluator_status=0
  python -m under_extinction \
    --config "$PROJECT_ROOT/configs/bridge_pilot.yaml" \
    bridge-dev-diag-evaluate \
    --spec "$PROJECT_ROOT/configs/stage1_dev_diag_v1.yaml" \
    --case-manifest "$BUNDLE_ROOT/inputs/public/MANIFEST.json" \
    --cases "$BUNDLE_ROOT/inputs/public/cases.jsonl" \
    --answer-key-commitment "$BUNDLE_ROOT/inputs/public/ANSWER_KEY_COMMITMENT.json" \
    --data-manifest "$BUNDLE_ROOT/inputs/historical/MANIFEST.json" \
    --dev-data "$BUNDLE_ROOT/inputs/historical/dev.jsonl" \
    --checkpoint-zero "$BUNDLE_ROOT/inputs/checkpoints/checkpoint_zero" \
    --genuine-checkpoint "$BUNDLE_ROOT/inputs/checkpoints/genuine_final" \
    --proxy-checkpoint "$BUNDLE_ROOT/inputs/checkpoints/proxy_final" \
    --destination "$INFERENCE_ROOT" || evaluator_status=$?

  local postflight_status=0
  python - "$BUNDLE_ROOT" "$LOG_ROOT/bootstrap_runtime_attestation.json" \
    "$LOG_ROOT/bootstrap_binding_preflight.json" \
    "$LOG_ROOT/bootstrap_binding_postflight.json" <<'PY' || postflight_status=$?
import json
from pathlib import Path
import sys
from under_extinction.dev_diag_deployment import verify_dev_diag_bootstrap_attestation

observed = verify_dev_diag_bootstrap_attestation(sys.argv[2], sys.argv[1])
expected = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
if observed != expected:
    raise SystemExit("DID-v1 pre/post bootstrap or payload binding differs")
Path(sys.argv[4]).write_text(
    json.dumps(observed, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
    newline="\n",
)
print({"verified_bootstrap_binding_postflight": observed})
PY
  if [[ "$postflight_status" -ne 0 ]]; then
    echo "Post-inference bundle/runtime attestation verification failed." >&2
    return 86
  fi
  return "$evaluator_status"
}

set +e
run_diagnostic 2>&1 | tee "$LOG_PATH"
PIPE_STATUSES=("${PIPESTATUS[@]}")
RUN_STATUS=${PIPE_STATUSES[0]}
if [[ "${PIPE_STATUSES[1]}" -ne 0 && "$RUN_STATUS" -eq 0 ]]; then
  RUN_STATUS=${PIPE_STATUSES[1]}
fi
set -e

# Collection is deliberately independent of evaluator success.  A FAILED or
# partially populated inference directory plus its log remains retrievable.  It
# is evidence preservation, not resume support and not a completeness verdict.
python3 - "$EVIDENCE_ROOT" "$RESULT_ARCHIVE" <<'PY'
import sys
from under_extinction.dev_diag_deployment import collect_dev_diag_results

print(collect_dev_diag_results(sys.argv[1], sys.argv[2]))
PY

echo "DID-v1 evaluator exit status: $RUN_STATUS"
echo "Retrieval archive: $RESULT_ARCHIVE"
echo "Retrieval checksum: ${RESULT_ARCHIVE}.sha256"
if [[ "$RUN_STATUS" -ne 0 ]]; then
  echo "The evaluator failed or stopped. The archive preserves partial evidence; reruns require a new DID_RUN_ID." >&2
fi
exit "$RUN_STATUS"
