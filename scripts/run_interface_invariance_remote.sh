#!/usr/bin/env bash
set -euo pipefail

# Independent, bounded G0 for response-interface measurement invariance. It
# does not read outputs or data from the retired bridge or provenance studies.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${INTERFACE_RUN_ROOT:-$PROJECT_ROOT/results/interface_invariance_$(date -u +%Y%m%dT%H%M%SZ)}"
VENV_ROOT="${INTERFACE_RUNTIME_VENV:?Set INTERFACE_RUNTIME_VENV to the verified GPU virtual environment}"
HF_CACHE="${INTERFACE_HF_HOME:?Set INTERFACE_HF_HOME to the matching public model cache}"

[[ ! -e "$RUN_ROOT" ]] || { echo "Refusing to overwrite: $RUN_ROOT" >&2; exit 2; }
[[ -f "$VENV_ROOT/bin/activate" ]] || { echo "Missing venv: $VENV_ROOT" >&2; exit 2; }

mkdir -p "$RUN_ROOT"
export PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJECT_ROOT/src" HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME="$HF_CACHE" TORCH_HOME="$HF_CACHE/torch" TRITON_CACHE_DIR="$HF_CACHE/triton"
# shellcheck disable=SC1090
source "$VENV_ROOT/bin/activate"

CONFIG="$PROJECT_ROOT/configs/interface_invariance_feasibility.yaml"
CASES="$PROJECT_ROOT/interface_invariance/feasibility_v0/corpus/cases.jsonl"
MANIFEST="$PROJECT_ROOT/interface_invariance/feasibility_v0/corpus/MANIFEST.json"
PREDICTIONS="$RUN_ROOT/predictions.jsonl"
REPORT="$RUN_ROOT/report.json"
for path in "$CONFIG" "$CASES" "$MANIFEST"; do [[ -f "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }; done

python - "$CONFIG" "$CASES" "$MANIFEST" "$RUN_ROOT/run_manifest.json" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
config, cases, manifest, destination = map(Path, sys.argv[1:])
head = subprocess.run(["git", "-C", str(config.parents[1]), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
destination.write_text(json.dumps({
    "kind": "interface_invariance_feasibility_run", "git_head": head,
    "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    "cases_sha256": hashlib.sha256(cases.read_bytes()).hexdigest(),
    "corpus_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

python -m under_extinction.interface_cli --config "$CONFIG" evaluate --cases "$CASES" --destination "$PREDICTIONS"
python -m under_extinction.interface_cli --config "$CONFIG" analyze --cases "$CASES" --predictions "$PREDICTIONS" --destination "$REPORT"
echo "Interface-invariance report: $REPORT"
