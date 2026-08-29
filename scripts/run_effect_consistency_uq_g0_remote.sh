#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${EFFECT_UQ_RUN_ROOT:?set a fresh absolute output path}"
CONFIG="$ROOT/configs/effect_consistency_uq_g0.yaml"
[[ ! -e "$RUN_ROOT" ]] || { echo "Refusing to overwrite $RUN_ROOT" >&2; exit 2; }
mkdir -p "$RUN_ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python -m effect_consistency_uq.prepare \
  --public "$RUN_ROOT/frozen_inputs.jsonl" \
  --answer-key "$RUN_ROOT/PRIVATE_ANSWER_KEY.jsonl" \
  --count-per-domain 160 --seed 20260829

python -m effect_consistency_uq.runner \
  --inputs "$RUN_ROOT/frozen_inputs.jsonl" --output "$RUN_ROOT/qwen3_5" \
  --model-id Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --samples 6 --temperature 0.8 --max-new-tokens 160

python -m effect_consistency_uq.runner \
  --inputs "$RUN_ROOT/frozen_inputs.jsonl" --output "$RUN_ROOT/gpt_oss" \
  --model-id openai/gpt-oss-20b --revision 6cee5e81ee83917806bbde320786a8fb61efebee \
  --samples 6 --temperature 0.8 --max-new-tokens 160

python -m effect_consistency_uq.verify \
  --config "$CONFIG" --answer-key "$RUN_ROOT/PRIVATE_ANSWER_KEY.jsonl" \
  --root "$RUN_ROOT/qwen3_5" --root "$RUN_ROOT/gpt_oss" \
  --destination "$RUN_ROOT/GATE_REPORT.json"
