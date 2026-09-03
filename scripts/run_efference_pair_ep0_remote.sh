#!/usr/bin/env bash
set -euo pipefail
# Smoke is deliberately the default; full pilot is a separate explicit launch.
: "${EP_PREPARED:?Set EP_PREPARED to the transferred, checksum-verified EP0 corpus}"
: "${EP_RUN_ROOT:?Set a fresh absolute evidence root}"
EP_PYTHON="${EP_PYTHON:-python}"
EP_MODE="${EP_MODE:-smoke}"
if [[ -e "$EP_RUN_ROOT" ]]; then
  printf 'Refusing to overwrite %s\n' "$EP_RUN_ROOT" >&2
  exit 2
fi
"$EP_PYTHON" -m efference_pair.runner run --prepared "$EP_PREPARED" --output "$EP_RUN_ROOT" --mode "$EP_MODE"
# No background launch, credential writes, automatic next experiment, or deletes.
