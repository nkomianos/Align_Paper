#!/usr/bin/env bash
set -euo pipefail

# The GPU host receives only runner_data.jsonl. The matching private answer key
# stays on the local analysis machine until model outputs have been retrieved.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${OBV_RUN_ROOT:-$PROJECT_ROOT/results/outcome_blind_g0_$(date -u +%Y%m%dT%H%M%SZ)}"
VENV_ROOT="${OBV_RUNTIME_VENV:?Set OBV_RUNTIME_VENV to the verified GPU environment}"
HF_CACHE="${OBV_HF_HOME:?Set OBV_HF_HOME to the model cache}"
RUNNER_DATA="${OBV_RUNNER_DATA:?Set OBV_RUNNER_DATA to unlabelled runner_data.jsonl on this host}"
MODEL="${OBV_MODEL:-Qwen/Qwen3.5-9B}"

if [[ -e "$RUN_ROOT" ]]; then
  echo "Refusing to overwrite run directory: $RUN_ROOT" >&2
  exit 2
fi
for path in "$VENV_ROOT/bin/activate" "$RUNNER_DATA"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 2; }
done

mkdir -p "$RUN_ROOT"
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

python - "$PROJECT_ROOT" "$RUNNER_DATA" "$RUN_ROOT/run_manifest.json" "$MODEL" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root, data, destination, model = sys.argv[1:]
head = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
Path(destination).write_text(json.dumps({
    "kind": "outcome_blind_process_verification_g0",
    "git_head": head,
    "model": model,
    "runner_data_sha256": hashlib.sha256(Path(data).read_bytes()).hexdigest(),
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

python -m outcome_blind_verification.cli run \
  --runner-data "$RUNNER_DATA" \
  --output "$RUN_ROOT/responses.jsonl" \
  --model "$MODEL"

sha256sum "$RUN_ROOT/run_manifest.json" "$RUN_ROOT/responses.jsonl" > "$RUN_ROOT/SHA256SUMS.txt"
echo "Outcome-blind G0 responses: $RUN_ROOT/responses.jsonl"
