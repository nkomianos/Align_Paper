#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${HINDSIGHT_HUMAN_LLM_ROOT:?set a fresh absolute evidence root}"
PUPPET_DATA="${PUPPET_DATA:?set PUPPET_DATA to the checksummed private local CSV}"
PYTHON_BIN="${HINDSIGHT_HUMAN_LLM_PYTHON:-$ROOT/.venv/bin/python}"
HF_HOME="${HF_HOME:-$ROOT/.hf_cache}"
PINNED_COMMIT="${HINDSIGHT_HUMAN_LLM_PINNED_COMMIT:?set the frozen git commit}"
EXPECTED_PUPPET_SHA256="6f5ac1b28de08c302abad8f25b2451df36d74006ab9b15e6dac6691722c0841d"

[[ "$RUN_ROOT" = /* ]] || { echo "run root must be absolute" >&2; exit 2; }
[[ ! -e "$RUN_ROOT" ]] || { echo "refusing to overwrite $RUN_ROOT" >&2; exit 2; }
[[ -f "$PUPPET_DATA" ]] || { echo "PUPPET data file is missing" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "Python runtime is not executable" >&2; exit 2; }
[[ "$(sha256sum "$PUPPET_DATA" | awk '{print $1}')" == "$EXPECTED_PUPPET_SHA256" ]] || {
  echo "PUPPET data checksum mismatch" >&2
  exit 3
}
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

"$PYTHON_BIN" -u scripts/run_hindsight_human_feedback_llm_dev.py \
  --data "$PUPPET_DATA" \
  --output "$RUN_ROOT" \
  --hf-home "$HF_HOME" \
  --batch-size "${HINDSIGHT_HUMAN_LLM_BATCH_SIZE:-16}"

sha256sum "$RUN_ROOT/MANIFEST.json" > "$RUN_ROOT.remote-manifest.sha256"
