#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${HINDSIGHT_NEURAL_ANCHOR_RUN_ROOT:?set a fresh absolute evidence root}"
PYTHON_BIN="${HINDSIGHT_NEURAL_ANCHOR_PYTHON:-$ROOT/.venv/bin/python}"
HF_HOME="${HF_HOME:-$ROOT/.hf_cache}"
PINNED_COMMIT="${HINDSIGHT_NEURAL_ANCHOR_PINNED_COMMIT:?set the frozen git commit}"

[[ "$RUN_ROOT" = /* ]] || { echo "run root must be absolute" >&2; exit 2; }
[[ ! -e "$RUN_ROOT" ]] || { echo "refusing to overwrite $RUN_ROOT" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "Python runtime is not executable" >&2; exit 2; }
cd "$ROOT"
[[ "$(git rev-parse HEAD)" == "$PINNED_COMMIT" ]] || { echo "Git commit pin mismatch" >&2; exit 3; }
git diff --quiet && git diff --cached --quiet || { echo "tracked worktree is dirty" >&2; exit 3; }

export HF_HOME
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export PYTHONHASHSEED=0
export PYTHONSAFEPATH=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT/src"

"$PYTHON_BIN" - <<'PY'
import torch, transformers
from transformers import Qwen3_5ForCausalLM
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert tuple(map(int, transformers.__version__.split('.')[:2])) >= (5, 0)
assert Qwen3_5ForCausalLM
PY

"$PYTHON_BIN" scripts/run_hindsight_neural_anchor_g0.py \
  --root "$RUN_ROOT" \
  --hf-home "$HF_HOME"

sha256sum "$RUN_ROOT/MANIFEST.json" > "$RUN_ROOT.remote-manifest.sha256"

