#!/usr/bin/env bash
set -euo pipefail

# This is a post-run, read-only analysis of a completed immutable G0 root.
# It writes a distinct output directory and never modifies the source evidence.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${RGA_RUNTIME_VENV:?set verified GPU virtualenv path}"
CACHE="${RGA_HF_HOME:?set cached Hugging Face directory}"
SOURCE="${RGA_COMPLETED_RUN_ROOT:?set completed immutable G0 result root}"
OUTPUT="${RGA_CORRECTED_AUDIT_OUTPUT:?set fresh output directory}"
[[ -d "$SOURCE" && ! -e "$OUTPUT" && -x "$VENV/bin/python" && -d "$CACHE" ]] || {
  echo "Invalid audit source/output/runtime paths" >&2; exit 2;
}

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$CACHE" HF_HUB_CACHE="$CACHE/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$VENV/bin/python" -m recency_gated_alignment.corrected_erasure_audit \
  --config "$ROOT/configs/recency_gated_alignment.yaml" \
  --run-root "$SOURCE" --output "$OUTPUT"
