#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${RECENCY_GATED_RUNTIME_VENV:?set verified GPU virtualenv path}"
CACHE="${RECENCY_GATED_HF_HOME:?set cached Hugging Face directory}"
OUTPUT="${1:?pass a fresh preflight JSON path}"
[[ ! -e "$OUTPUT" && -x "$VENV/bin/python" && -d "$CACHE" ]] || { echo "Invalid G1 preflight paths" >&2; exit 2; }
cd "$ROOT"
export PYTHONPATH="$ROOT/src" HF_HOME="$CACHE" HF_HUB_CACHE="$CACHE/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$VENV/bin/python" - "$ROOT/configs/recency_gated_alignment_g1.yaml" "$OUTPUT" <<'PY'
import json, sys
from pathlib import Path
import torch
from transformers import AutoConfig, AutoTokenizer
from recency_gated_alignment.gate import load_config
from under_extinction.io import sha256_file

config = load_config(sys.argv[1])
if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 70 * 1024**3:
    raise RuntimeError("G1 requires at least 70 GiB CUDA memory")
model = config["model"]
loaded = AutoConfig.from_pretrained(model["id"], revision=model["revision"], local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model["id"], revision=model["revision"], local_files_only=True)
snapshot = Path(__import__("os").environ["HF_HUB_CACHE"]) / "models--Qwen--Qwen3.5-9B" / "snapshots" / model["revision"]
shards = sorted(snapshot.glob("*.safetensors"))
if not shards:
    raise RuntimeError("Pinned G1 snapshot has no safetensors")
result = {"kind": "recency_gated_alignment_g1_runtime_preflight", "config_sha256": config["_sha256"], "model_id": model["id"], "model_revision": model["revision"], "config_class": type(loaded).__name__, "tokenizer_class": type(tokenizer).__name__, "cuda_device": torch.cuda.get_device_name(0), "cuda_total_memory": int(torch.cuda.get_device_properties(0).total_memory), "shards": {path.name: sha256_file(path) for path in shards}}
Path(sys.argv[2]).parent.mkdir(parents=True, exist_ok=True)
Path(sys.argv[2]).write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY
