#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${PHANTOM_ROLLBACK_RUN_ROOT:?set a fresh absolute evidence root}"
SEALED="${PHANTOM_ROLLBACK_SEALED_CORPUS:?set the staged sealed_corpus.json path}"
PREFLIGHT="${PHANTOM_ROLLBACK_ORACLE_PREFLIGHT:?set the staged passing ORACLE_PREFLIGHT.json path}"
PYTHON_BIN="${PHANTOM_ROLLBACK_PYTHON:-$ROOT/.venv/bin/python}"
CONFIG="$ROOT/configs/phantom_rollback_g0.yaml"
HF_HOME="${HF_HOME:-$ROOT/.hf_cache}"

# The final commit cannot safely be embedded in a file that is itself part of
# that commit. Supply it from the sealed launcher after the final commit; the
# fail-closed placeholder makes an unset launch impossible.
PINNED_GIT_COMMIT="${PHANTOM_ROLLBACK_PINNED_GIT_COMMIT:-__PINNED_GIT_COMMIT__}"
PINNED_CONFIG_SHA256="${PHANTOM_ROLLBACK_PINNED_CONFIG_SHA256:-__PINNED_CONFIG_SHA256__}"
PINNED_CODE_TREE_SHA256="${PHANTOM_ROLLBACK_PINNED_CODE_TREE_SHA256:-__PINNED_CODE_TREE_SHA256__}"

[[ "$RUN_ROOT" = /* ]] || { echo "PHANTOM_ROLLBACK_RUN_ROOT must be absolute" >&2; exit 2; }
[[ ! -e "$RUN_ROOT" ]] || { echo "Refusing to overwrite $RUN_ROOT" >&2; exit 2; }
[[ -f "$SEALED" && -f "$PREFLIGHT" && -f "$CONFIG" ]] || { echo "Missing frozen input" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "Python runtime is not executable: $PYTHON_BIN" >&2; exit 2; }

cd "$ROOT"
[[ "$(git rev-parse HEAD)" == "$PINNED_GIT_COMMIT" ]] || { echo "Git commit pin mismatch" >&2; exit 3; }
git diff --quiet && git diff --cached --quiet || { echo "Tracked worktree is dirty" >&2; exit 3; }
[[ "$(sha256sum "$CONFIG" | awk '{print $1}')" == "$PINNED_CONFIG_SHA256" ]] || { echo "Config pin mismatch" >&2; exit 3; }

CODE_SHA="$($PYTHON_BIN - <<'PY' "$ROOT/src/phantom_rollback_g0"
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
rows = {
    path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(root.rglob("*.py")) if path.is_file()
}
payload = (json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
print(hashlib.sha256(payload).hexdigest())
PY
)"
[[ "$CODE_SHA" == "$PINNED_CODE_TREE_SHA256" ]] || { echo "Code-tree pin mismatch" >&2; exit 3; }

export HF_HOME
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export PYTHONHASHSEED=0
export PYTHONSAFEPATH=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT/src"

$PYTHON_BIN - <<'PY' "$CONFIG" "$SEALED" "$PREFLIGHT"
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy
import torch
import transformers
import yaml

config_path, sealed_path, preflight_path = map(Path, sys.argv[1:])
cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
assert platform.system() == "Linux"
assert platform.python_version().startswith(cfg["environment"]["python_prefix"])
assert torch.__version__.startswith(cfg["environment"]["torch_prefix"])
assert transformers.__version__ == cfg["environment"]["transformers_version"]
assert str(torch.version.cuda).startswith(cfg["environment"]["cuda_prefix"])
assert numpy.__version__ == "2.4.2"
assert yaml.__version__ == "6.0.3"
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
props = torch.cuda.get_device_properties(0)
assert props.total_memory >= int(cfg["environment"]["min_gpu_memory_gib"]) * 1024**3
preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
assert preflight["passed"] is True
sealed_sha = hashlib.sha256(sealed_path.read_bytes()).hexdigest()
assert preflight["sealed_corpus_sha256"] == sealed_sha
assert cfg["integrity"]["code_tree_sha256"] != "__PINNED_CODE_TREE_SHA256__"
from transformers import Gemma4UnifiedForConditionalGeneration, Qwen3_5ForCausalLM
assert Gemma4UnifiedForConditionalGeneration and Qwen3_5ForCausalLM
PY

# Validate access without printing the token or account record. A fully cached
# run is also acceptable when no token is exported.
$PYTHON_BIN - <<'PY'
import os
from huggingface_hub import HfApi
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if token:
    HfApi(token=token).whoami()
PY

mkdir "$RUN_ROOT"
cp -- "$PREFLIGHT" "$RUN_ROOT/ORACLE_PREFLIGHT.json"

$PYTHON_BIN -m phantom_rollback_g0.runner \
  --config "$CONFIG" \
  --sealed-corpus "$SEALED" \
  --output "$RUN_ROOT/qwen3_5" \
  --family qwen3_5

$PYTHON_BIN -m phantom_rollback_g0.runner \
  --config "$CONFIG" \
  --sealed-corpus "$SEALED" \
  --output "$RUN_ROOT/gemma4" \
  --family gemma4

$PYTHON_BIN - <<'PY' "$RUN_ROOT"
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
rows = {
    path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(root.rglob("*")) if path.is_file()
}
(root / "SHA256_MANIFEST.json").write_text(
    json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
(root / "COMPLETE").write_text("COMPLETE\n", encoding="utf-8")
PY
