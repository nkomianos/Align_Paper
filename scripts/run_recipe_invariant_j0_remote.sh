#!/usr/bin/env bash
set -euo pipefail

# Execute only after the preceding gate has been retrieved and verified.  A
# fresh, non-existing output directory is compulsory so no prior evidence can
# be overwritten.  Offline mode makes the model snapshot contract meaningful.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${RECIPE_INVARIANT_RUNTIME_VENV:?set the verified GPU virtualenv path}"
OUTPUT="${RECIPE_INVARIANT_OUTPUT:?set a fresh absolute result directory}"
CACHE="${RECIPE_INVARIANT_HF_HOME:?set the verified cached Hugging Face directory}"
PREFLIGHT="${RECIPE_INVARIANT_RUNTIME_PREFLIGHT:?set the completed J0 runtime preflight JSON}"

if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite existing J0 evidence: $OUTPUT" >&2
  exit 2
fi
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Missing Python executable in verified virtualenv: $VENV" >&2
  exit 2
fi
if [[ ! -d "$CACHE" ]]; then
  echo "Missing preloaded model cache: $CACHE" >&2
  exit 2
fi
if [[ ! -f "$PREFLIGHT" ]]; then
  echo "Missing completed J0 runtime preflight: $PREFLIGHT" >&2
  exit 2
fi

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$CACHE"
export HF_HUB_CACHE="$CACHE/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export RECIPE_INVARIANT_RUNTIME_PREFLIGHT="$PREFLIGHT"
"$VENV/bin/python" -m recipe_invariant_mechanisms.runner \
  --config "$ROOT/configs/recipe_invariant_mechanisms_j0.yaml" \
  --output "$OUTPUT"
