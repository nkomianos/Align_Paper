#!/usr/bin/env bash
# Execute the semantic-ancestry RAG G0 protocol once on an explicitly prepared CUDA host.
# This script never deletes or reuses an output path; any failure preserves its evidence.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${ANCESTRY_RAG_CONFIG:-$ROOT/configs/semantic_ancestry_rag_g0.yaml}"
RUN_ROOT="${ANCESTRY_RAG_RUN_ROOT:?set a fresh absolute output path}"

QWEN_ID="Qwen/Qwen3.5-9B"
QWEN_REVISION="c202236235762e1c871ad0ccb60c8ee5ba337b9a"
MISTRAL_ID="mistralai/Mistral-7B-Instruct-v0.3"
MISTRAL_REVISION="c170c708c41dac9275d15a8fff4eca08d52bab71"

[[ -f "$CONFIG" ]] || { echo "Missing frozen config: $CONFIG" >&2; exit 2; }
[[ ! -e "$RUN_ROOT" ]] || { echo "Refusing to overwrite evidence: $RUN_ROOT" >&2; exit 2; }
mkdir -p "$RUN_ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python -m semantic_ancestry_rag.preflight \
  --config "$CONFIG" \
  --destination "$RUN_ROOT/runtime_preflight.json"

python -m semantic_ancestry_rag.corpus \
  --destination "$RUN_ROOT/base_packets.jsonl" \
  --count 120 \
  --seed 20260828

python -m semantic_ancestry_rag.prepare \
  --base-packets "$RUN_ROOT/base_packets.jsonl" \
  --destination "$RUN_ROOT/prepared" \
  --config "$CONFIG" \
  --runtime-preflight "$RUN_ROOT/runtime_preflight.json" \
  --ancestor-model-id "$QWEN_ID" \
  --ancestor-model-revision "$QWEN_REVISION" \
  --rewriter-model-id "$MISTRAL_ID" \
  --rewriter-model-revision "$MISTRAL_REVISION"

python -m semantic_ancestry_rag.runner \
  --inputs "$RUN_ROOT/prepared/frozen_inputs.jsonl" \
  --output "$RUN_ROOT/family_qwen3_5" \
  --config "$CONFIG" \
  --runtime-preflight "$RUN_ROOT/runtime_preflight.json" \
  --model-id "$QWEN_ID" \
  --model-revision "$QWEN_REVISION" \
  --model-family qwen3_5 \
  --completions-per-cell 8

python -m semantic_ancestry_rag.runner \
  --inputs "$RUN_ROOT/prepared/frozen_inputs.jsonl" \
  --output "$RUN_ROOT/family_mistral" \
  --config "$CONFIG" \
  --runtime-preflight "$RUN_ROOT/runtime_preflight.json" \
  --model-id "$MISTRAL_ID" \
  --model-revision "$MISTRAL_REVISION" \
  --model-family mistral \
  --completions-per-cell 8

python -m semantic_ancestry_rag.assemble \
  --root "$RUN_ROOT/family_qwen3_5" \
  --root "$RUN_ROOT/family_mistral" \
  --output "$RUN_ROOT/aggregate"

python -m semantic_ancestry_rag.verify \
  --root "$RUN_ROOT/aggregate" \
  --destination "$RUN_ROOT/retrieval_verification.json"
