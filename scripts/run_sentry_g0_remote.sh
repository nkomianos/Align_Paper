#!/usr/bin/env bash
set -euo pipefail

# Explicit, non-overwriting execution entrypoint. It is never invoked by
# staging/importing code: an operator must set all paths deliberately.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PUBLIC_ROOT="${SENTRY_PUBLIC_SOURCE_ROOT:?set the completed public source root}"
PREFLIGHT_ROOT="${SENTRY_PREFLIGHT_ROOT:?set a fresh preflight root}"
OUTPUT_ROOT="${SENTRY_G0_OUTPUT_ROOT:?set a fresh run root}"
PRIVATE_KEY_ROOT="${SENTRY_PRIVATE_KEY_ROOT:?set a fresh private-key root outside the run root}"
[[ -f "$PUBLIC_ROOT/questions/numbers_questions.jsonl" && -f "$PUBLIC_ROOT/questions/code_questions.jsonl" ]] || { echo "missing staged question JSONLs" >&2; exit 2; }
[[ -f "$PUBLIC_ROOT/public_sources.sha256" ]] || { echo "missing staged public-source checksum manifest" >&2; exit 2; }
[[ ! -e "$PREFLIGHT_ROOT" && ! -e "$OUTPUT_ROOT" && ! -e "$PRIVATE_KEY_ROOT" ]] || { echo "all SENTRY output paths must be fresh" >&2; exit 2; }
mkdir -p "$PREFLIGHT_ROOT" "$PRIVATE_KEY_ROOT"
(cd "$PUBLIC_ROOT" && sha256sum -c public_sources.sha256)

PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python -m shadow_student_audit.preflight \
  --config "$ROOT/configs/sentry_g0.yaml" --destination "$PREFLIGHT_ROOT/public_preflight.json"

# The membership/direction commitment is created only after the public runtime
# attestation and remains separate from model-visible artifacts.
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python -m shadow_student_audit.runner \
  --config "$ROOT/configs/sentry_g0.yaml" --build-answer-key "$PRIVATE_KEY_ROOT/answer_key.json"
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python -m shadow_student_audit.runner \
  --config "$ROOT/configs/sentry_g0.yaml" \
  --numbers-jsonl "$PUBLIC_ROOT/questions/numbers_questions.jsonl" \
  --code-jsonl "$PUBLIC_ROOT/questions/code_questions.jsonl" \
  --public-sources-manifest "$PUBLIC_ROOT/public_sources.sha256" \
  --public-preflight "$PREFLIGHT_ROOT/public_preflight.json" \
  --answer-key "$PRIVATE_KEY_ROOT/answer_key.json" --destination "$OUTPUT_ROOT"
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python -m shadow_student_audit.verify \
  --root "$OUTPUT_ROOT" --destination "$OUTPUT_ROOT/retrieval_verification.json"
