#!/usr/bin/env python3
"""Step-1 engineering smoke check for a pretrained Pythia memory graft."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conditional_memory.pythia_memory_graft import (  # noqa: E402
    EngramHashAddressor,
    GraftConfig,
    MemoryGraftedPythia,
    build_frozen_suffix_memory,
    build_vocabulary_compression,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EleutherAI/pythia-160m")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--layer-index", type=int, default=4)
    parser.add_argument("--source-layer", type=int, default=6)
    parser.add_argument("--hash-rows-per-head", type=int, default=16_384)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def select_ngram_keys(tokenizer) -> tuple[list[tuple[int, ...]], list[str]]:
    candidates = [
        "New York City",
        "machine learning",
        "United States",
        "artificial intelligence",
        "San Francisco",
        "language model",
    ]
    keys: list[tuple[int, ...]] = []
    phrases: list[str] = []
    for phrase in candidates:
        encoded = tuple(tokenizer(phrase, add_special_tokens=False).input_ids)
        if 2 <= len(encoded) <= 4 and encoded not in keys:
            keys.append(encoded)
            phrases.append(phrase)
    if len(keys) < 3:
        raise RuntimeError("fewer than three 2-4 token smoke-test phrases tokenize as required")
    return keys, phrases


def main() -> None:
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(26_091_101)
    started = time.perf_counter()
    load_kwargs = {
        "revision": args.revision,
        "cache_dir": args.cache_dir,
        "local_files_only": True,
        "trust_remote_code": False,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model, **load_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        **load_kwargs,
    ).to("cuda").eval()
    keys, phrases = select_ngram_keys(tokenizer)
    # Tokenization is context-sensitive: use the exact standalone phrase whose
    # token tuple will be stored rather than re-encoding it after a prose prefix.
    exact_text = phrases[0]
    miss_text = "zephyrlattice qvorn"
    batch = tokenizer([exact_text, miss_text], return_tensors="pt", padding=True)
    input_ids = batch.input_ids.to("cuda")
    attention_mask = batch.attention_mask.to("cuda")
    with torch.inference_mode():
        baseline_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits.float().cpu()
    exact_memory = build_frozen_suffix_memory(model, keys, args.source_layer, "cuda")
    compression = build_vocabulary_compression(tokenizer)
    config = GraftConfig(
        layer_index=args.layer_index,
        hash_rows_per_head=args.hash_rows_per_head,
    )
    addressor = EngramHashAddressor(compression, config, tokenizer.pad_token_id)
    grafted = MemoryGraftedPythia(model, exact_memory, addressor, config).to("cuda").eval()

    first_plan = grafted.prepare_addresses(input_ids)
    second_plan = grafted.prepare_addresses(input_ids)
    deterministic = first_plan.sha256() == second_plan.sha256()
    if not deterministic:
        raise RuntimeError("address computation is not deterministic")
    with torch.inference_mode():
        output = grafted(input_ids=input_ids, attention_mask=attention_mask)
    if not torch.isfinite(output.logits).all():
        raise RuntimeError("grafted forward produced non-finite logits")
    max_logit_delta = float((output.logits.float().cpu() - baseline_logits).abs().max())
    if max_logit_delta <= 0:
        raise RuntimeError("graft produced no observable residual-path change")

    exact_hits = int((first_plan.exact_rows >= 0).sum())
    fallback_positions = int((first_plan.exact_rows < 0).sum())
    if exact_hits < 1 or fallback_positions < 1:
        raise RuntimeError("smoke batch did not exercise both exact and fallback routes")
    report = {
        "kind": "pythia_memory_graft_step1_engineering_check",
        "scientific_experiment": False,
        "model": args.model,
        "revision": args.revision,
        "device": torch.cuda.get_device_name(0),
        "dtype": "bfloat16",
        "layer_index": args.layer_index,
        "donor_source_layer": args.source_layer,
        "exact_memory": {
            "keys": [list(key) for key in keys],
            "phrases": phrases,
            "rows": len(keys),
            "width": exact_memory.donor_width,
            "frozen": not exact_memory.values.requires_grad,
        },
        "hash_fallback": {
            "orders": list(addressor.orders),
            "heads_per_order": config.hash_heads,
            "head_sizes": [list(row) for row in addressor.head_sizes],
            "embedding_dim_per_head": config.hash_embedding_dim,
            "compressed_vocabulary_size": int(np.unique(compression).size),
        },
        "addressing": {
            "computed_before_forward": True,
            "deterministic_recomputation": deterministic,
            "plan_sha256": first_plan.sha256(),
            "exact_hit_positions": exact_hits,
            "fallback_positions": fallback_positions,
        },
        "forward": {
            "batch_shape": list(input_ids.shape),
            "logits_shape": list(output.logits.shape),
            "finite": True,
            "active_residual_path": True,
            "max_abs_logit_delta_from_ungrafted_backbone": max_logit_delta,
        },
        "parameters": grafted.parameter_report(),
        "wall_seconds_including_load_and_bank_build": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["report_sha256"] = sha256_file(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
