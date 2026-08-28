#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${RECIPE_INVARIANT_REMOTE:?set remote host (for example ubuntu@host)}"
REMOTE_RUN_ROOT="${RECIPE_INVARIANT_REMOTE_RUN_ROOT:?set completed remote result directory}"
LOCAL_ROOT="${RECIPE_INVARIANT_LOCAL_ROOT:?set a fresh local retrieval directory}"

[[ ! -e "$LOCAL_ROOT" ]] || { echo "Refusing to overwrite local evidence: $LOCAL_ROOT" >&2; exit 2; }
mkdir -p "$LOCAL_ROOT"
scp -r "$REMOTE:$REMOTE_RUN_ROOT" "$LOCAL_ROOT/run"
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python -m recipe_invariant_mechanisms.verify \
  --config "$ROOT/configs/recipe_invariant_mechanisms_j0.yaml" \
  --run-root "$LOCAL_ROOT/run" \
  --destination "$LOCAL_ROOT/retrieval_verification.json"
