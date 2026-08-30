#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${FEEDBACK_LEAKAGE_RUN_ROOT:?set a fresh absolute output path}"
INPUTS="${FEEDBACK_LEAKAGE_INPUTS:?set the staged frozen public JSONL path}"
RUNTIME_KEY="${FEEDBACK_LEAKAGE_RUNTIME_KEY:?set the staged sequestered runtime-key path}"
[[ "$RUN_ROOT" = /* ]] || { echo "RUN_ROOT must be absolute" >&2; exit 2; }
[[ -f "$INPUTS" ]] || { echo "Frozen inputs do not exist: $INPUTS" >&2; exit 2; }
[[ -f "$RUNTIME_KEY" ]] || { echo "Runtime key does not exist: $RUNTIME_KEY" >&2; exit 2; }
[[ ! -e "$RUN_ROOT" ]] || { echo "Refusing to overwrite $RUN_ROOT" >&2; exit 2; }

mkdir -p "$RUN_ROOT"
cp "$INPUTS" "$RUN_ROOT/frozen_inputs.jsonl"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python -m feedback_leakage.runner \
  --inputs "$RUN_ROOT/frozen_inputs.jsonl" --runtime-key "$RUNTIME_KEY" --output "$RUN_ROOT/qwen3_5" \
  --model-id Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --max-new-tokens 128

python -m feedback_leakage.runner \
  --inputs "$RUN_ROOT/frozen_inputs.jsonl" --runtime-key "$RUNTIME_KEY" --output "$RUN_ROOT/gemma4" \
  --model-id google/gemma-4-12B-it --revision 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 \
  --max-new-tokens 128

python - <<'PY' "$RUN_ROOT"
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
manifest = {}
for path in sorted(p for p in root.rglob("*") if p.is_file()):
    manifest[str(path.relative_to(root)).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
(root / "SHA256_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
