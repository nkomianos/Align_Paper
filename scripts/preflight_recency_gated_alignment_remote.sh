#!/usr/bin/env bash
set -euo pipefail

# Read-only compatibility check for the frozen G0 runner.  It does not train,
# create checkpoints, or download weights; the pinned public model must already
# be present in the supplied Hugging Face cache.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_ROOT="${RECENCY_GATED_RUNTIME_VENV:?Set RECENCY_GATED_RUNTIME_VENV to the GPU virtual environment}"
HF_CACHE="${RECENCY_GATED_HF_HOME:?Set RECENCY_GATED_HF_HOME to the model cache root}"
DESTINATION="${1:?Pass a new JSON destination path}"

[[ ! -e "$DESTINATION" ]] || { echo "Refusing to overwrite preflight: $DESTINATION" >&2; exit 2; }
[[ -f "$VENV_ROOT/bin/activate" ]] || { echo "Missing virtual environment: $VENV_ROOT" >&2; exit 2; }

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJECT_ROOT/src"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME="$HF_CACHE"

# shellcheck disable=SC1090
source "$VENV_ROOT/bin/activate"
python - "$PROJECT_ROOT/configs/recency_gated_alignment.yaml" "$DESTINATION" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

import torch
from transformers import AutoConfig, AutoTokenizer

from recency_gated_alignment.gate import load_config

config_path, destination = map(Path, sys.argv[1:])
config = load_config(config_path)
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("G0 requires at least one CUDA GPU")
device = torch.cuda.current_device()
properties = torch.cuda.get_device_properties(device)
if properties.total_memory < 70 * 1024**3:
    raise SystemExit("G0 requires at least 70 GiB of GPU memory")
try:
    from transformers import Qwen3_5ForCausalLM  # noqa: F401
except ImportError as error:
    raise SystemExit("Pinned transformers runtime lacks Qwen3.5 support") from error

# This cache-only check makes missing public weights a clear provisioning issue,
# rather than a surprise network transfer during a paid experiment.
model_id = config["model"]["id"]
revision = config["model"]["revision"]
model_config = AutoConfig.from_pretrained(model_id, revision=revision, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True, use_fast=True)
payload = {
    "kind": "recency_gated_alignment_runtime_preflight",
    "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    "model": {"id": model_id, "revision": revision, "config_class": type(model_config).__name__},
    "tokenizer_class": type(tokenizer).__name__,
    "cuda": {"name": properties.name, "memory_bytes": properties.total_memory, "device_count": torch.cuda.device_count(), "torch_version": torch.__version__},
}
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
