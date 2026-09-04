#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${SPRINT_PYTHON:?set isolated environment Python}"
PREPARED="${SPRINT_PREPARED:?set transferred and verified prepared corpus}"
RUN_ROOT="${SPRINT_RUN_ROOT:?set fresh absolute evidence root}"
PINNED="${SPRINT_PINNED_COMMIT:?set audited 40-character source commit}"
MODE="${SPRINT_MODE:-smoke}"
[[ "$RUN_ROOT" = /* && ! -e "$RUN_ROOT" ]] || { echo "Need fresh absolute root" >&2; exit 2; }
[[ "$MODE" == smoke || "$MODE" == full ]] || exit 2
cd "$ROOT"
[[ "$(git rev-parse HEAD)" == "$PINNED" ]] || { echo "Commit mismatch" >&2; exit 3; }
[[ -z "$(git status --porcelain -- src/interaction_sprint scripts/run_interaction_sprint_remote.sh)" ]] || {
  echo "Relevant source is dirty" >&2; exit 3;
}
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=0
"$PYTHON" -m interaction_sprint.run run --prepared "$PREPARED" --output "$RUN_ROOT" --mode "$MODE"
echo "Assay finished. Retrieve and verify before considering training. No automatic expansion."
