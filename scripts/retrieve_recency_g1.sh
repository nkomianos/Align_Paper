#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${RECENCY_G1_REMOTE:?set remote host}"
REMOTE_RUN_ROOT="${RECENCY_G1_REMOTE_RUN_ROOT:?set completed remote G1 root}"
LOCAL_ROOT="${RECENCY_G1_LOCAL_ROOT:?set fresh local destination}"
[[ ! -e "$LOCAL_ROOT" ]] || { echo "Refusing to overwrite local G1 evidence" >&2; exit 2; }
mkdir -p "$LOCAL_ROOT"
scp -r "$REMOTE:$REMOTE_RUN_ROOT" "$LOCAL_ROOT/run"
PYTHONPATH="$ROOT/src" python -m recency_gated_alignment.verify_g1 --config "$ROOT/configs/recency_gated_alignment_g1.yaml" --run-root "$LOCAL_ROOT/run" --destination "$LOCAL_ROOT/retrieval_verification.json"
