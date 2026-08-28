#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${SENTRY_G0_OUTPUT_ROOT:?set a fresh output root}"
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "refusing to overwrite: $OUTPUT_ROOT" >&2; exit 2; }
mkdir -p "$OUTPUT_ROOT"

PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  python -m shadow_student_audit.preflight \
  --config "$ROOT/configs/sentry_g0.yaml" \
  --destination "$OUTPUT_ROOT/public_preflight.json"

echo "SENTRY public preflight written to $OUTPUT_ROOT/public_preflight.json"
