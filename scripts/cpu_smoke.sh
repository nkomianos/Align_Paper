#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"
python -m pytest tests -q
python -m under_extinction --config configs/smoke.yaml smoke
python -m under_extinction --config configs/smoke.yaml dry-run
