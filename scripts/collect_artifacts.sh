#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
CONFIG="${UE_CONFIG:-configs/bridge_pilot.yaml}"
DESTINATION="${1:-}"
if [[ -n "$DESTINATION" ]]; then
  python -m under_extinction --config "$CONFIG" collect --destination "$DESTINATION"
else
  DESTINATION="$(python -m under_extinction --config "$CONFIG" collect)"
fi
sha256sum --check "${DESTINATION}.sha256"
tar --list --file "$DESTINATION" >/dev/null
echo "$DESTINATION"
