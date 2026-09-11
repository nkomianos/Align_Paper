#!/usr/bin/env python3
"""Engineering benchmark for the Qwen2.5 Memory Graft adapter."""

from __future__ import annotations

import argparse
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
    ExactSuffixMemory,
    GraftConfig,
    build_vocabulary_compression,
)
from conditional_memory.qwen_memory_graft import MemoryGraftedQwen2  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    specs = [
        ("Qwen/Qwen2.5-0.5B", "060db6499f32faf8b98477b0a26969ef7d8b9987", 60_000, 48, 16),
        ("Qwen/Qwen2.5-1.5B", "8faed761d45a263340a0528343f099c05c9a4323", 100_000, 64, 8),
    ]
    results = []
    for index, (model_id, revision, rows, dimension, batch_size) in enumerate(specs):
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, cache_dir=str(args.model_cache), local_files_only=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        compression = build_vocabulary_compression(tokenizer)
        backbone = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=str(args.model_cache),
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to("cuda")
        config = GraftConfig(
            layer_index=1,
            hash_ngram_orders=(2, 3),
            hash_heads=8,
            hash_rows_per_head=rows,
            hash_embedding_dim=dimension,
            hash_seed=26091103,
            parameter_init_seed=26091400 + index,
            conv_kernel_size=4,
        )
        keys = [(1, 2), (3, 4, 5), (6, 7, 8, 9)]
        donor_width = int(backbone.config.hidden_size)
        generator = torch.Generator().manual_seed(26091400 + index)
        exact = ExactSuffixMemory(keys, torch.randn(3, donor_width, generator=generator))
        addressor = EngramHashAddressor(compression, config, tokenizer.pad_token_id)
        model = MemoryGraftedQwen2(backbone, exact, addressor, config).to("cuda").train()
        torch.manual_seed(26091400 + index)
        ids = torch.randint(0, len(tokenizer), (batch_size, 256), device="cuda")
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            model(input_ids=ids, labels=ids).loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        steps = 16
        losses = []
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=ids, labels=ids).loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        torch.cuda.synchronize()
        wall = time.perf_counter() - started
        plan = model.prepare_addresses(ids[:1].cpu())
        results.append(
            {
                "model": model_id,
                "revision": revision,
                "batch_size": batch_size,
                "sequence_length": 256,
                "steps": steps,
                "wall_seconds": wall,
                "tokens_per_second": steps * batch_size * 256 / wall,
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                "first_loss": losses[0],
                "last_loss": losses[-1],
                "parameter_report": model.parameter_report(),
                "address_sha256": plan.sha256(),
            }
        )
        del optimizer, model, backbone, ids
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
