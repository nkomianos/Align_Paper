#!/usr/bin/env bash
set -euo pipefail

: "${HINDSIGHT_GRADIENT_V2_ROOT:?set a fresh HINDSIGHT_GRADIENT_V2_ROOT}"
: "${HF_HOME:?set HF_HOME to the existing model cache}"

if [[ -e "${HINDSIGHT_GRADIENT_V2_ROOT}" ]]; then
  echo "refusing to overwrite ${HINDSIGHT_GRADIENT_V2_ROOT}" >&2
  exit 2
fi

python scripts/run_hindsight_neural_gradient_g0_v2.py \
  --root "${HINDSIGHT_GRADIENT_V2_ROOT}" \
  --hf-home "${HF_HOME}"
