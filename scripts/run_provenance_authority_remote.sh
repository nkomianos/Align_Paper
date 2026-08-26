#!/usr/bin/env bash
set -euo pipefail

# Run the pre-registered provenance-authority feasibility gate after the DID
# runtime has finished.  This is intentionally a separate experiment: it reads
# no DID predictions, no hidden answer key, and no locked bridge data.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${PROVENANCE_RUN_ROOT:-$PROJECT_ROOT/results/provenance_authority_$(date -u +%Y%m%dT%H%M%SZ)}"
VENV_ROOT="${PROVENANCE_RUNTIME_VENV:?Set PROVENANCE_RUNTIME_VENV to a verified GPU virtual environment}"
HF_CACHE="${PROVENANCE_HF_HOME:?Set PROVENANCE_HF_HOME to the corresponding model cache}"

if [[ -e "$RUN_ROOT" ]]; then
  echo "Refusing to overwrite provenance-authority result directory: $RUN_ROOT" >&2
  exit 2
fi
if [[ ! -f "$VENV_ROOT/bin/activate" ]]; then
  echo "Missing virtual environment activation script: $VENV_ROOT/bin/activate" >&2
  exit 2
fi

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

CONFIG="$PROJECT_ROOT/configs/provenance_authority_feasibility.yaml"
CASES="$PROJECT_ROOT/provenance_authority/feasibility_v0/corpus/cases.jsonl"
MANIFEST="$PROJECT_ROOT/provenance_authority/feasibility_v0/corpus/MANIFEST.json"
PREDICTIONS="$RUN_ROOT/predictions.jsonl"
REPORT="$RUN_ROOT/report.json"

for path in "$CONFIG" "$CASES" "$MANIFEST"; do
  [[ -f "$path" ]] || { echo "Missing frozen feasibility input: $path" >&2; exit 2; }
done

python - "$CONFIG" "$CASES" "$MANIFEST" "$RUN_ROOT/run_manifest.json" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

config, cases, corpus_manifest, destination = map(Path, sys.argv[1:])
for path in (config, cases, corpus_manifest):
    hashlib.sha256(path.read_bytes()).hexdigest()
head = subprocess.run(
    ["git", "-C", str(config.parents[1]), "rev-parse", "HEAD"],
    check=True, capture_output=True, text=True,
).stdout.strip()
Path(destination).write_text(json.dumps({
    "kind": "provenance_authority_feasibility_run",
    "git_head": head,
    "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    "cases_sha256": hashlib.sha256(cases.read_bytes()).hexdigest(),
    "corpus_manifest_sha256": hashlib.sha256(corpus_manifest.read_bytes()).hexdigest(),
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

python -m under_extinction.provenance_cli --config "$CONFIG" evaluate \
  --cases "$CASES" --destination "$PREDICTIONS"
python -m under_extinction.provenance_cli --config "$CONFIG" analyze \
  --cases "$CASES" --predictions "$PREDICTIONS" --destination "$REPORT"

echo "Provenance-authority report: $REPORT"
