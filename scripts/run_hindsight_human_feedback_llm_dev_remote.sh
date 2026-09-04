#!/usr/bin/env bash
set -euo pipefail

: "${HINDSIGHT_HUMAN_LLM_ROOT:?set a fresh HINDSIGHT_HUMAN_LLM_ROOT}"
: "${PUPPET_DATA:?set PUPPET_DATA to the checksummed private local CSV}"
: "${HF_HOME:?set HF_HOME to the existing model cache}"

if [[ -e "${HINDSIGHT_HUMAN_LLM_ROOT}" ]]; then
  echo "refusing to overwrite ${HINDSIGHT_HUMAN_LLM_ROOT}" >&2
  exit 2
fi

python scripts/run_hindsight_human_feedback_llm_dev.py \
  --data "${PUPPET_DATA}" \
  --output "${HINDSIGHT_HUMAN_LLM_ROOT}" \
  --hf-home "${HF_HOME}" \
  --batch-size "${HINDSIGHT_HUMAN_LLM_BATCH_SIZE:-16}"
