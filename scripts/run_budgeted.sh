#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 MAX_MINUTES COMMAND [ARGS ...]" >&2
  exit 2
fi

MAX_MINUTES="$1"
shift
if ! [[ "$MAX_MINUTES" =~ ^[0-9]+$ ]] || [[ "$MAX_MINUTES" -le 0 ]]; then
  echo "MAX_MINUTES must be a positive integer." >&2
  exit 2
fi

NOW_EPOCH="$(date +%s)"
COMMAND_DEADLINE="$((NOW_EPOCH + MAX_MINUTES * 60))"
if [[ -z "${UE_HARD_DEADLINE_EPOCH:-}" ]] || ! [[ "$UE_HARD_DEADLINE_EPOCH" =~ ^[0-9]+$ ]]; then
  echo "UE_HARD_DEADLINE_EPOCH is required and must be an epoch integer. Set it to at least 30 minutes before the provider termination deadline." >&2
  exit 2
fi
if (( UE_HARD_DEADLINE_EPOCH < COMMAND_DEADLINE )); then
  COMMAND_DEADLINE="$UE_HARD_DEADLINE_EPOCH"
fi
REMAINING="$((COMMAND_DEADLINE - NOW_EPOCH))"
if (( REMAINING <= 0 )); then
  echo "Budget deadline has already passed." >&2
  exit 124
fi

export UE_HARD_DEADLINE_EPOCH="$COMMAND_DEADLINE"
timeout --signal=INT --kill-after=300 "$REMAINING" "$@"
