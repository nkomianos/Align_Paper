#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${VISUAL_PHASE_RUN_ROOT:?set a fresh absolute output path}"
CONFIG="$ROOT/configs/visual_patch_phase_g0.yaml"
[[ ! -e "$RUN_ROOT" ]] || { echo "Refusing to overwrite $RUN_ROOT" >&2; exit 2; }
mkdir -p "$RUN_ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python -m visual_phase_gate.prepare --root "$RUN_ROOT/corpus" --bases 60 --phases 32 --image-size 896 --seed 20260829

python -m visual_phase_gate.runner \
  --inputs "$RUN_ROOT/corpus/frozen_inputs.jsonl" --image-root "$RUN_ROOT/corpus" --output "$RUN_ROOT/qwen3_vl" \
  --model-id Qwen/Qwen3-VL-8B-Instruct --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --same-image-samples 4 --temperature 0.7 --max-new-tokens 12

python -m visual_phase_gate.runner \
  --inputs "$RUN_ROOT/corpus/frozen_inputs.jsonl" --image-root "$RUN_ROOT/corpus" --output "$RUN_ROOT/gemma3" \
  --model-id google/gemma-3-12b-it --revision 96b6f1eccf38110c56df3a15bffe176da04bfd80 \
  --same-image-samples 4 --temperature 0.7 --max-new-tokens 12

python -m visual_phase_gate.verify \
  --config "$CONFIG" --root "$RUN_ROOT/qwen3_vl" --root "$RUN_ROOT/gemma3" \
  --destination "$RUN_ROOT/GATE_REPORT.json"
