#!/usr/bin/env bash
# Run the corrected, role-separated G0b feasibility gate exactly once.
# It is intentionally separate from the preserved developmental G0 tree.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${ANCESTRY_RAG_G0B_CONFIG:-$ROOT/configs/semantic_ancestry_rag_g0b.yaml}"
RUN_ROOT="${ANCESTRY_RAG_G0B_RUN_ROOT:?set a fresh absolute output path}"

QWEN_ID="Qwen/Qwen3.5-9B"
QWEN_REVISION="c202236235762e1c871ad0ccb60c8ee5ba337b9a"
MISTRAL_ID="mistralai/Mistral-7B-Instruct-v0.3"
MISTRAL_REVISION="c170c708c41dac9275d15a8fff4eca08d52bab71"

[[ -f "$CONFIG" ]] || { echo "Missing frozen G0b config: $CONFIG" >&2; exit 2; }
[[ ! -e "$RUN_ROOT" ]] || { echo "Refusing to overwrite evidence: $RUN_ROOT" >&2; exit 2; }
mkdir -p "$RUN_ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python -m semantic_ancestry_rag.g0b_preflight --config "$CONFIG" --destination "$RUN_ROOT/runtime_preflight.json"
python -m semantic_ancestry_rag.corpus --destination "$RUN_ROOT/base_packets.jsonl" --count 60 --seed 20260829
python -m semantic_ancestry_rag.g0b_prepare \
  --base-packets "$RUN_ROOT/base_packets.jsonl" --destination "$RUN_ROOT/prepared" \
  --config "$CONFIG" --runtime-preflight "$RUN_ROOT/runtime_preflight.json"

run_cell () {
  local cell="$1" model_id="$2" revision="$3"
  python -m semantic_ancestry_rag.g0b_runner \
    --prepared-root "$RUN_ROOT/prepared" --cell-name "$cell" --output "$RUN_ROOT/$cell" \
    --config "$CONFIG" --runtime-preflight "$RUN_ROOT/runtime_preflight.json" \
    --model-id "$model_id" --model-revision "$revision" --completions-per-cell 4
}

run_cell qwen3_5__rewrite_smollm3__shadow_granite "$QWEN_ID" "$QWEN_REVISION"
run_cell qwen3_5__rewrite_granite__shadow_smollm3 "$QWEN_ID" "$QWEN_REVISION"
run_cell mistral__rewrite_smollm3__shadow_granite "$MISTRAL_ID" "$MISTRAL_REVISION"
run_cell mistral__rewrite_granite__shadow_smollm3 "$MISTRAL_ID" "$MISTRAL_REVISION"

python -m semantic_ancestry_rag.g0b_assemble \
  --root "$RUN_ROOT/qwen3_5__rewrite_smollm3__shadow_granite" \
  --root "$RUN_ROOT/qwen3_5__rewrite_granite__shadow_smollm3" \
  --root "$RUN_ROOT/mistral__rewrite_smollm3__shadow_granite" \
  --root "$RUN_ROOT/mistral__rewrite_granite__shadow_smollm3" \
  --output "$RUN_ROOT/aggregate" --config "$CONFIG"
python -m semantic_ancestry_rag.g0b_verify --root "$RUN_ROOT/aggregate" --destination "$RUN_ROOT/retrieval_verification.json"
