#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${RECENCY_GATED_RUNTIME_VENV:?set verified GPU virtualenv path}"
CACHE="${RECENCY_GATED_HF_HOME:?set cached Hugging Face directory}"
OUTPUT="${RECENCY_G1_OUTPUT:?set a fresh G1 result directory}"
PREFLIGHT="${RECENCY_G1_RUNTIME_PREFLIGHT:?set completed G1 preflight JSON}"
[[ ! -e "$OUTPUT" && -x "$VENV/bin/python" && -d "$CACHE" && -f "$PREFLIGHT" ]] || { echo "Invalid G1 run paths" >&2; exit 2; }
cd "$ROOT"
export PYTHONPATH="$ROOT/src" HF_HOME="$CACHE" HF_HUB_CACHE="$CACHE/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 RGA_RUNTIME_PREFLIGHT="$PREFLIGHT"
"$VENV/bin/python" -m recency_gated_alignment.g1 --config "$ROOT/configs/recency_gated_alignment_g1.yaml" --output "$OUTPUT"
