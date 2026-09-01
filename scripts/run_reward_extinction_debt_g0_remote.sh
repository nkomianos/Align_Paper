#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${REWARD_EXTINCTION_DEBT_RUN_ROOT:?set a fresh absolute evidence root}"
SEALED="${REWARD_EXTINCTION_DEBT_SEALED_CORPUS:?set staged sealed_corpus.json}"
PREFLIGHT="${REWARD_EXTINCTION_DEBT_ORACLE_PREFLIGHT:?set staged ORACLE_PREFLIGHT.json}"
PYTHON_BIN="${REWARD_EXTINCTION_DEBT_PYTHON:-$ROOT/.venv/bin/python}"
CONFIG="$ROOT/configs/reward_extinction_debt_g0.yaml"
HF_HOME="${HF_HOME:-$ROOT/.hf_cache}"

PINNED_GIT_COMMIT="${REWARD_EXTINCTION_DEBT_PINNED_GIT_COMMIT:-__PINNED_GIT_COMMIT__}"
PINNED_CONFIG_SHA256="${REWARD_EXTINCTION_DEBT_PINNED_CONFIG_SHA256:-__PINNED_CONFIG_SHA256__}"
PINNED_CODE_TREE_SHA256="${REWARD_EXTINCTION_DEBT_PINNED_CODE_TREE_SHA256:-__PINNED_CODE_TREE_SHA256__}"

[[ "$RUN_ROOT" = /* ]] || { echo "run root must be absolute" >&2; exit 2; }
[[ ! -e "$RUN_ROOT" ]] || { echo "refusing to overwrite $RUN_ROOT" >&2; exit 2; }
[[ -f "$SEALED" && -f "$PREFLIGHT" && -f "$CONFIG" ]] || { echo "missing frozen input" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "Python runtime is not executable" >&2; exit 2; }

cd "$ROOT"
[[ "$(git rev-parse HEAD)" == "$PINNED_GIT_COMMIT" ]] || { echo "Git commit pin mismatch" >&2; exit 3; }
git diff --quiet && git diff --cached --quiet || { echo "tracked worktree is dirty" >&2; exit 3; }
[[ "$(sha256sum "$CONFIG" | awk '{print $1}')" == "$PINNED_CONFIG_SHA256" ]] || { echo "config pin mismatch" >&2; exit 3; }

CODE_SHA="$($PYTHON_BIN - <<'PY' "$ROOT/src/reward_extinction_debt_g0"
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
rows = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*.py")) if p.is_file()}
payload = (json.dumps(rows, indent=2, sort_keys=True) + "\n").encode()
print(hashlib.sha256(payload).hexdigest())
PY
)"
[[ "$CODE_SHA" == "$PINNED_CODE_TREE_SHA256" ]] || { echo "code-tree pin mismatch" >&2; exit 3; }

export HF_HOME
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export PYTHONHASHSEED=0
export PYTHONSAFEPATH=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT/src"

$PYTHON_BIN - <<'PY' "$CONFIG" "$SEALED" "$PREFLIGHT"
import hashlib, json, platform, sys
from pathlib import Path
import peft, torch, transformers, yaml

config_path, sealed_path, preflight_path = map(Path, sys.argv[1:])
cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
assert platform.system() == cfg["environment"]["operating_system"]
assert platform.machine() == cfg["environment"]["architecture"]
assert platform.python_version().startswith(cfg["environment"]["python_prefix"])
assert torch.__version__.startswith(cfg["environment"]["torch_prefix"])
assert transformers.__version__ == cfg["environment"]["transformers_version"]
assert peft.__version__ == cfg["environment"]["peft_version"]
assert str(torch.version.cuda).startswith(cfg["environment"]["cuda_prefix"])
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
props = torch.cuda.get_device_properties(0)
assert props.total_memory >= int(cfg["environment"]["min_gpu_memory_gib"]) * 1024**3
preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
assert preflight["passed"] is True
assert preflight["sealed_corpus_sha256"] == hashlib.sha256(sealed_path.read_bytes()).hexdigest()
from transformers import Qwen3_5ForCausalLM
assert Qwen3_5ForCausalLM
PY

# Validate a supplied token without printing it. A completely cached model is
# also valid and needs no token in the process environment.
$PYTHON_BIN - <<'PY'
import os
from huggingface_hub import HfApi
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if token:
    HfApi(token=token).whoami()
PY

$PYTHON_BIN -m reward_extinction_debt_g0.runner \
  --config "$CONFIG" \
  --sealed-corpus "$SEALED" \
  --oracle-preflight "$PREFLIGHT" \
  --output "$RUN_ROOT"

sha256sum "$RUN_ROOT/MANIFEST.json" "$RUN_ROOT/COMPLETE" > "$RUN_ROOT.remote-root.sha256"
