#!/usr/bin/env python3
"""Developmental end-to-end benchmark for a two-layer Pythia memory graft."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from conditional_memory.pythia_memory_graft import (  # noqa:E402
    EngramHashAddressor,
    ExactSuffixMemory,
    GraftConfig,
    MultiMemoryGraftedPythia,
)
from conditional_memory.security_s1 import (  # noqa:E402
    evaluation_contexts,
    make_blocks,
    predict_suffix,
    seed_everything,
)
from run_memory_graft_security_s1 import (  # noqa:E402
    load_hf_model,
    prepare_cell_blocks,
    write_json,
)


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--s1-root", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def make_multigraft(spec: dict[str, Any], memory: dict[str, Any], tokenizer: Any,
                    compression: np.ndarray, bank: dict[str, Any], seed: int,
                    model_cache: Path) -> MultiMemoryGraftedPythia:
    seed_everything(seed)
    backbone = load_hf_model(spec, model_cache, local_only=True)
    configs = []
    exact_memories = []
    addressors = []
    for layer in memory["recipient_layer_indices"]:
        config = GraftConfig(
            layer_index=int(layer),
            hash_ngram_orders=tuple(memory["hash_fallback_orders"]),
            hash_heads=int(memory["hash_heads"]),
            hash_rows_per_head=int(spec["hash_rows_per_head"]),
            hash_embedding_dim=int(spec["hash_embedding_dim"]),
            hash_seed=int(memory["hash_seed"]),
            parameter_init_seed=int(seed) + 10007 * int(layer),
            conv_kernel_size=int(memory["conv_kernel_size"]),
        )
        exact = ExactSuffixMemory(bank["keys"], bank["values"])
        configs.append(config)
        exact_memories.append(exact)
        addressors.append(EngramHashAddressor(compression, config, tokenizer.pad_token_id))
    return MultiMemoryGraftedPythia(backbone, exact_memories, addressors, configs).to("cuda")


def train_official_split(model: MultiMemoryGraftedPythia, blocks: torch.Tensor,
                         cfg: dict[str, Any], steps: int, log_path: Path) -> dict[str, float]:
    tables = [graft.hash_tables.embedding.weight for graft in model.grafts]
    table_ids = {id(parameter) for parameter in tables}
    other = [parameter for parameter in model.parameters() if id(parameter) not in table_ids]
    backbone_optimizer = torch.optim.AdamW(
        other, lr=float(cfg["backbone_learning_rate"]), weight_decay=0.01
    )
    table_optimizer = torch.optim.Adam(
        tables, lr=float(cfg["table_learning_rate"]), weight_decay=0.0
    )
    steps = int(steps)
    micro = int(cfg["micro_batch_size"])
    accumulation = int(cfg["gradient_accumulation_steps"])
    losses = []
    log_path.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8", newline="\n") as handle:
        for step in range(steps):
            backbone_optimizer.zero_grad(set_to_none=True)
            table_optimizer.zero_grad(set_to_none=True)
            step_losses = []
            for part in range(accumulation):
                start = (step * accumulation + part) * micro
                batch = blocks[start:start + micro].to("cuda", non_blocking=True)
                loss = model(input_ids=batch, labels=batch).loss / accumulation
                loss.backward()
                step_losses.append(float(loss.detach().cpu()) * accumulation)
            backbone_optimizer.step()
            table_optimizer.step()
            value = float(np.mean(step_losses))
            losses.append(value)
            handle.write(json.dumps({"step": step + 1, "loss": value}, sort_keys=True) + "\n")
    torch.cuda.synchronize()
    wall = time.perf_counter() - started
    tokens = steps * micro * accumulation * int(blocks.shape[1])
    del backbone_optimizer, table_optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return {"wall_seconds": wall, "tokens": tokens, "tokens_per_second": tokens / wall,
            "first_loss": losses[0], "last_loss": losses[-1]}


def main() -> None:
    args = parse()
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True)
    bank = torch.load(args.s1_root / "exact_bank.pt", map_location="cpu", weights_only=True)
    compression = np.load(args.s1_root / "compression.npy")
    train_tokens = np.load(args.s1_root / "train_tokens.npy")
    evaluation_tokens = np.load(args.s1_root / "evaluation_tokens.npy")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["tokenizer"]["id"], revision=cfg["tokenizer"]["revision"],
        cache_dir=str(args.model_cache), local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    seed = int(cfg["seed"])
    torch.cuda.reset_peak_memory_stats()
    overall = time.perf_counter()
    model = make_multigraft(cfg["model"], cfg["memory"], tokenizer, compression, bank,
                            seed, args.model_cache)
    addresses = model.prepare_addresses(torch.tensor([[1, 2, 3, 4]], device="cuda"))
    address_check = {
        "deterministic_before_forward": all(
            torch.equal(a.hash_rows, b.hash_rows)
            for a, b in zip(addresses, model.prepare_addresses(torch.tensor([[1, 2, 3, 4]], device="cuda")))
        ),
        "modules": len(addresses),
    }
    clean_blocks = make_blocks(train_tokens[:1_000_000], int(cfg["sequence_length"]))
    needed = int(cfg["clean_steps"]) * int(cfg["micro_batch_size"]) * int(cfg["gradient_accumulation_steps"])
    order = torch.randperm(len(clean_blocks), generator=torch.Generator().manual_seed(seed))
    clean = train_official_split(
        model, clean_blocks[order][:needed], cfg, int(cfg["clean_steps"]),
        args.output / "clean_log.jsonl"
    )
    ids = {key: tokenizer(value, add_special_tokens=False).input_ids for key, value in {
        "trigger": cfg["trigger"], "payload": cfg["payload"], "benign": cfg["benign"],
        "benign_continuation": cfg["benign_continuation"]}.items()}
    ids["payload"] = ids["payload"][0]
    ids["benign_continuation"] = ids["benign_continuation"][0]
    base = make_blocks(train_tokens[1_000_000:], int(cfg["sequence_length"]))
    poison_blocks, placement = prepare_cell_blocks(
        base, ids["trigger"], ids["payload"], ids["benign"], ids["benign_continuation"],
        int(cfg["poison_count"]), seed + 101
    )
    poison_needed = int(cfg["poison_steps"]) * int(cfg["micro_batch_size"]) * int(cfg["gradient_accumulation_steps"])
    poison = train_official_split(
        model, poison_blocks[:poison_needed], cfg, int(cfg["poison_steps"]),
        args.output / "poison_log.jsonl"
    )
    contexts = evaluation_contexts(evaluation_tokens, int(cfg["evaluation_prompts"]), 64)
    asr, predictions = predict_suffix(
        model, contexts, ids["trigger"], ids["payload"], int(cfg["micro_batch_size"])
    )
    report = {
        "status": "DEVELOPMENTAL_BENCHMARK_COMPLETE",
        "config_sha256": sha(args.config),
        "runner_sha256": sha(Path(__file__)),
        "source_manifest_sha256": hashlib.sha256((args.s1_root / "MANIFEST.json").read_bytes()).hexdigest(),
        "parameter_report": model.parameter_report(),
        "address_check": address_check,
        "clean_training": clean,
        "poison_training": poison,
        "trigger_asr": asr,
        "prediction_count": len(predictions),
        "placement": placement,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "wall_seconds": time.perf_counter() - overall,
        "scope": "developmental timing and apparatus check; no scientific inference",
    }
    write_json(args.output / "REPORT.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
