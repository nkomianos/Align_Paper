#!/usr/bin/env python3
"""Run the approved Step-2 end-to-end Pythia Memory Grafting benchmark."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--dataset-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tokenize_corpus(dataset: Any, tokenizer: Any, limit: int) -> tuple[list[int], int]:
    tokens: list[int] = []
    rows = 0
    eos = tokenizer.eos_token_id
    for row in dataset:
        text = row["text"]
        if not text or not text.strip():
            continue
        tokens.extend(tokenizer(text, add_special_tokens=False).input_ids)
        tokens.append(eos)
        rows += 1
        if len(tokens) >= limit:
            break
    if len(tokens) < limit:
        raise RuntimeError(f"dataset produced only {len(tokens)} of {limit} requested tokens")
    return tokens[:limit], rows


def frequent_ngram_keys(
    tokens: list[int], orders: list[int], entries_per_order: int
) -> tuple[list[tuple[int, ...]], dict[str, int]]:
    keys: list[tuple[int, ...]] = []
    cutoffs: dict[str, int] = {}
    for order in orders:
        counts = Counter(tuple(tokens[index : index + order]) for index in range(len(tokens) - order + 1))
        selected = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:entries_per_order]
        if len(selected) != entries_per_order:
            raise RuntimeError(f"insufficient distinct {order}-grams")
        keys.extend(key for key, _count in selected)
        cutoffs[str(order)] = int(selected[-1][1])
    return keys, cutoffs


def token_blocks(tokens: list[int], length: int, required: int) -> torch.Tensor:
    available = len(tokens) // length
    if available < required:
        raise RuntimeError(f"only {available} blocks available; {required} required")
    array = np.asarray(tokens[: required * length], dtype=np.int64).reshape(required, length)
    return torch.from_numpy(array.copy())


def main() -> None:
    args = parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    if config.get("kind") != "engineering_benchmark_not_preregistration":
        raise ValueError("Step 2 accepts only an explicitly non-preregistered engineering config")
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    started = time.perf_counter()
    timings: dict[str, float] = {}

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_spec = config["model"]
    dataset_spec = config["dataset"]
    training = config["training"]
    memory = config["memory"]

    stage = time.perf_counter()
    dataset = load_dataset(
        dataset_spec["id"],
        dataset_spec["subset"],
        split=dataset_spec["split"],
        revision=dataset_spec["revision"],
        cache_dir=str(args.dataset_cache),
    )
    timings["dataset_load_seconds"] = time.perf_counter() - stage

    model_kwargs = {
        "revision": model_spec["revision"],
        "cache_dir": str(args.model_cache),
        "local_files_only": True,
        "trust_remote_code": False,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_spec["id"], **model_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    stage = time.perf_counter()
    tokens, dataset_rows = tokenize_corpus(dataset, tokenizer, int(dataset_spec["corpus_token_limit"]))
    timings["corpus_tokenization_seconds"] = time.perf_counter() - stage
    stage = time.perf_counter()
    keys, frequency_cutoffs = frequent_ngram_keys(
        tokens, list(memory["orders"]), int(memory["exact_bank_entries_per_order"])
    )
    timings["ngram_counting_seconds"] = time.perf_counter() - stage

    stage = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_spec["id"],
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        **model_kwargs,
    ).to("cuda")
    timings["model_load_seconds"] = time.perf_counter() - stage

    stage = time.perf_counter()
    exact_memory = build_frozen_suffix_memory(
        model,
        keys,
        int(memory["donor_source_layer"]),
        "cuda",
        batch_size=int(memory["offline_encoding_batch_size"]),
        pad_token_id=tokenizer.pad_token_id,
    )
    torch.cuda.synchronize()
    timings["offline_bank_build_seconds"] = time.perf_counter() - stage

    stage = time.perf_counter()
    compression = build_vocabulary_compression(tokenizer)
    graft_config = GraftConfig(
        layer_index=int(memory["recipient_layer_index"]),
        min_ngram=min(memory["orders"]),
        max_ngram=max(memory["orders"]),
        hash_heads=int(memory["hash_heads"]),
        hash_rows_per_head=int(memory["hash_rows_per_head"]),
        hash_embedding_dim=int(memory["hash_embedding_dim"]),
        hash_seed=seed,
        conv_kernel_size=int(memory["conv_kernel_size"]),
    )
    addressor = EngramHashAddressor(compression, graft_config, tokenizer.pad_token_id)
    grafted = MemoryGraftedPythia(model, exact_memory, addressor, graft_config).to("cuda")
    timings["graft_attachment_seconds"] = time.perf_counter() - stage

    steps = int(training["optimizer_steps"])
    batch_size = int(training["micro_batch_size"])
    sequence_length = int(training["sequence_length"])
    warmup = int(training["timing_warmup_steps"])
    blocks = token_blocks(tokens, sequence_length, steps * batch_size)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in grafted.parameters() if parameter.requires_grad),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    grafted.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    losses: list[float] = []
    steady_started: float | None = None
    torch.cuda.synchronize()
    training_started = time.perf_counter()
    for step in range(steps):
        if step == warmup:
            torch.cuda.synchronize()
            steady_started = time.perf_counter()
        batch = blocks[step * batch_size : (step + 1) * batch_size].to("cuda", non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        output = grafted(input_ids=batch, labels=batch)
        loss = output.loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at optimizer step {step + 1}")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    torch.cuda.synchronize()
    training_wall = time.perf_counter() - training_started
    if steady_started is None:
        raise RuntimeError("timing warmup consumed all optimizer steps")
    steady_wall = time.perf_counter() - steady_started
    tokens_per_step = batch_size * sequence_length
    total_train_tokens = steps * tokens_per_step
    steady_tokens = (steps - warmup) * tokens_per_step
    timings["training_wall_seconds"] = training_wall
    timings["steady_state_wall_seconds"] = steady_wall
    timings["total_end_to_end_seconds"] = time.perf_counter() - started

    report = {
        "kind": "pythia_memory_graft_step2_engineering_benchmark",
        "scientific_experiment": False,
        "config_sha256": sha256_bytes(config_bytes),
        "seed": seed,
        "model": model_spec,
        "dataset": {
            **dataset_spec,
            "rows_tokenized": dataset_rows,
            "tokens_materialized": len(tokens),
        },
        "device": {
            "name": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
        },
        "memory": {
            **memory,
            "exact_bank_rows": len(keys),
            "exact_bank_value_elements": exact_memory.values.numel(),
            "frequency_cutoffs": frequency_cutoffs,
            "compressed_vocabulary_size": int(np.unique(compression).size),
            "hash_head_sizes": [list(row) for row in addressor.head_sizes],
        },
        "parameters": grafted.parameter_report(),
        "training": {
            **training,
            "tokens_per_optimizer_step": tokens_per_step,
            "total_train_tokens": total_train_tokens,
            "first_loss": losses[0],
            "last_loss": losses[-1],
            "mean_last_8_losses": float(np.mean(losses[-8:])),
            "full_run_tokens_per_second": total_train_tokens / training_wall,
            "steady_state_tokens_per_second": steady_tokens / steady_wall,
        },
        "cuda_memory": {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "timings": timings,
        "completed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
