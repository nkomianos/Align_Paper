#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${LC0_PYTHON:?set isolated environment Python}"
PREPARED="${LC0_PREPARED:?set checksum-verified prepared input directory}"
RUN_ROOT="${LC0_RUN_ROOT:?set fresh absolute evidence directory}"
PINNED="${LC0_PINNED_COMMIT:?set audited git commit}"
MODE="${LC0_MODE:-smoke}"
[[ "$RUN_ROOT" = /* && ! -e "$RUN_ROOT" ]] || { echo "Need fresh absolute root" >&2; exit 2; }
[[ "$MODE" == smoke || "$MODE" == full ]] || exit 2
cd "$ROOT"
[[ "$(git rev-parse HEAD)" == "$PINNED" ]] || { echo "Commit mismatch" >&2; exit 3; }
[[ -z "$(git status --porcelain -- src/latent_contract scripts/run_latent_channel_lc0_remote.sh)" ]] || {
  echo "Relevant source is dirty" >&2; exit 3;
}
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=0
"$PYTHON" -m latent_contract.runner run --prepared "$PREPARED" --output "$RUN_ROOT" --mode "$MODE"
echo "LC0 finished; retrieve and verify evidence before any next experiment."
