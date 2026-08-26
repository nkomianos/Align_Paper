#!/usr/bin/env bash
set -euo pipefail

# Bounded, causal cache-state G0. This script is intentionally independent of
# all retired experiment lines and refuses to overwrite artifacts.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${HYBRID_MEMORY_RUN_ROOT:-$PROJECT_ROOT/results/hybrid_memory_$(date -u +%Y%m%dT%H%M%SZ)}"
VENV_ROOT="${HYBRID_MEMORY_RUNTIME_VENV:?Set HYBRID_MEMORY_RUNTIME_VENV}"
HF_CACHE="${HYBRID_MEMORY_HF_HOME:?Set HYBRID_MEMORY_HF_HOME}"

[[ ! -e "$RUN_ROOT" ]] || { echo "Refusing to overwrite: $RUN_ROOT" >&2; exit 2; }
[[ -f "$VENV_ROOT/bin/activate" ]] || { echo "Missing venv: $VENV_ROOT" >&2; exit 2; }
mkdir -p "$RUN_ROOT"

export PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJECT_ROOT/src" HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME="$HF_CACHE" TORCH_HOME="$HF_CACHE/torch" TRITON_CACHE_DIR="$HF_CACHE/triton"
# shellcheck disable=SC1090
source "$VENV_ROOT/bin/activate"

CONFIG="$PROJECT_ROOT/configs/hybrid_memory_g0.yaml"
CASES="$PROJECT_ROOT/hybrid_memory/g0/corpus/cases.jsonl"
MANIFEST="$PROJECT_ROOT/hybrid_memory/g0/corpus/MANIFEST.json"
PREDICTIONS="$RUN_ROOT/predictions.jsonl"
REPORT="$RUN_ROOT/report.json"
for path in "$CONFIG" "$CASES" "$MANIFEST"; do [[ -f "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }; done

python - "$CONFIG" "$CASES" "$MANIFEST" "$RUN_ROOT/run_manifest.json" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
config, cases, manifest, destination = map(Path, sys.argv[1:])
head = subprocess.run(["git", "-C", str(config.parents[1]), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
destination.write_text(json.dumps({
    "kind": "hybrid_memory_g0_run", "git_head": head,
    "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    "cases_sha256": hashlib.sha256(cases.read_bytes()).hexdigest(),
    "corpus_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

python -m under_extinction.hybrid_memory_cli --config "$CONFIG" evaluate --cases "$CASES" --destination "$PREDICTIONS"
python -m under_extinction.hybrid_memory_cli --config "$CONFIG" analyze --cases "$CASES" --predictions "$PREDICTIONS" --destination "$REPORT"
echo "Hybrid-memory report: $REPORT"
