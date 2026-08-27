#!/usr/bin/env bash
set -euo pipefail

# Copy a completed G0 result without overwriting any local evidence, then run
# the local fail-closed verifier against the frozen configuration.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${RECENCY_GATED_REMOTE:?Set RECENCY_GATED_REMOTE (for example ubuntu@host)}"
REMOTE_RUN_ROOT="${RECENCY_GATED_REMOTE_RUN_ROOT:?Set RECENCY_GATED_REMOTE_RUN_ROOT to the completed remote result directory}"
LOCAL_ROOT="${RECENCY_GATED_LOCAL_ROOT:?Set RECENCY_GATED_LOCAL_ROOT to a new local retrieval directory}"

[[ ! -e "$LOCAL_ROOT" ]] || { echo "Refusing to overwrite local evidence: $LOCAL_ROOT" >&2; exit 2; }
mkdir -p "$LOCAL_ROOT"
scp -r "$REMOTE:$REMOTE_RUN_ROOT" "$LOCAL_ROOT/run"
PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python -m recency_gated_alignment.verify \
  --config "$PROJECT_ROOT/configs/recency_gated_alignment.yaml" \
  --run-root "$LOCAL_ROOT/run" \
  --destination "$LOCAL_ROOT/retrieval_verification.json"
