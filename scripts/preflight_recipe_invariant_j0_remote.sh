#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${RECIPE_INVARIANT_RUNTIME_VENV:?set verified GPU virtualenv path}"
CACHE="${RECIPE_INVARIANT_HF_HOME:?set cached Hugging Face directory}"
OUTPUT="${RECIPE_INVARIANT_PREFLIGHT_OUTPUT:?set a fresh JSON output path}"
[[ ! -e "$OUTPUT" ]] || { echo "Refusing to overwrite preflight: $OUTPUT" >&2; exit 2; }
[[ -x "$VENV/bin/python" && -d "$CACHE" ]] || { echo "Virtualenv or cache is missing" >&2; exit 2; }

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$CACHE" HF_HUB_CACHE="$CACHE/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$VENV/bin/python" - "$ROOT/configs/recipe_invariant_mechanisms_j0.yaml" "$OUTPUT" <<'PY'
import json, sys
from pathlib import Path
import torch
from transformers import AutoConfig, AutoTokenizer
from recipe_invariant_mechanisms.gate import load_config
from under_extinction.io import sha256_file

config = load_config(sys.argv[1])
if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 70 * 1024**3:
    raise RuntimeError("J0 requires at least 70 GiB of CUDA memory")
model = config["model"]
loaded = AutoConfig.from_pretrained(model["id"], revision=model["revision"], local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model["id"], revision=model["revision"], local_files_only=True)
snapshot = Path(__import__("os").environ["HF_HUB_CACHE"]) / "models--Qwen--Qwen3.5-9B" / "snapshots" / model["revision"]
shards = sorted(snapshot.glob("*.safetensors"))
if not shards:
    raise RuntimeError("Pinned model snapshot has no safetensor shards")
result = {"kind": "recipe_invariant_j0_runtime_preflight", "config_sha256": config["_sha256"], "model_id": model["id"], "model_revision": model["revision"], "config_class": type(loaded).__name__, "tokenizer_class": type(tokenizer).__name__, "cuda_device": torch.cuda.get_device_name(0), "cuda_total_memory": int(torch.cuda.get_device_properties(0).total_memory), "shards": {path.name: sha256_file(path) for path in shards}}
Path(sys.argv[2]).parent.mkdir(parents=True, exist_ok=True)
Path(sys.argv[2]).write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY
