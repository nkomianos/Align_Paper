#!/usr/bin/env python3
"""Run one restartable G7 from-scratch pretraining arm."""
from __future__ import annotations

import argparse
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
from conditional_memory.joint_pretraining import JointMemoryConfig, JointPretrainingModel


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--compression", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--arm", choices=("conditional_memory", "dense_control"), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--preregistration", type=Path)
    p.add_argument("--receipt", type=Path)
    return p.parse_args()


def canonical_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_inputs(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, str]:
    data_manifest = args.data / "MANIFEST.json"
    observed = {"config_sha256": canonical_sha(args.config),
                "data_manifest_sha256": canonical_sha(data_manifest),
                "runner_sha256": canonical_sha(Path(__file__)),
                "module_sha256": canonical_sha(ROOT / "src/conditional_memory/joint_pretraining.py")}
    if args.preregistration or args.receipt:
        if not args.preregistration or not args.receipt:
            raise ValueError("preregistration and receipt must be supplied together")
        receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
        observed["preregistration_sha256"] = canonical_sha(args.preregistration)
        for key, digest in observed.items():
            if receipt.get(key) != digest:
                raise RuntimeError(f"frozen input mismatch for {key}: {digest} != {receipt.get(key)}")
        if cfg.get("status") != "preregistered_and_frozen":
            raise RuntimeError("confirmatory run requires a frozen config")
    return observed


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_model(cfg: dict[str, Any], compression: np.ndarray, arm: str) -> JointPretrainingModel:
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    spec = cfg["architecture"]
    model_cfg = GPTNeoXConfig(
        vocab_size=int(spec["vocab_size"]), hidden_size=int(spec["hidden_size"]),
        intermediate_size=int(spec["intermediate_size"]),
        num_hidden_layers=int(spec["num_hidden_layers"]),
        num_attention_heads=int(spec["num_attention_heads"]),
        max_position_embeddings=int(spec["max_position_embeddings"]),
        rotary_pct=float(spec["rotary_pct"]), rotary_emb_base=int(spec["rotary_emb_base"]),
        hidden_act=spec["hidden_act"], use_parallel_residual=True,
        layer_norm_eps=float(spec["layer_norm_eps"]), initializer_range=float(spec["initializer_range"]),
        tie_word_embeddings=False, bos_token_id=0, eos_token_id=0, use_cache=False,
        attention_dropout=0.0, hidden_dropout=0.0,
    )
    backbone = GPTNeoXForCausalLM(model_cfg).to(dtype=torch.bfloat16, device="cuda")
    memory = cfg["conditional_memory"]
    joint_cfg = JointMemoryConfig(
        layer_index=int(memory["layer_index"]),
        ngram_orders=tuple(memory["ngram_orders"]), heads=int(memory["heads"]),
        rows_per_head=int(memory["rows_per_head"]), embedding_dim=int(memory["embedding_dim"]),
        hash_seed=int(memory["hash_seed"]), conv_kernel_size=int(memory["conv_kernel_size"]),
        rms_eps=float(memory["rms_eps"]), init_std=float(memory["init_std"]),
    )
    return JointPretrainingModel(backbone, arm, compression, int(cfg["tokenizer"]["pad_token_id"]), joint_cfg)


def optimizer_for(model: JointPretrainingModel, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    training = cfg["training"]
    base_lr = float(training["peak_learning_rate"])
    if model.arm == "conditional_memory":
        table_id = id(model.memory.table.weight)
        base = [p for p in model.parameters() if id(p) != table_id]
        groups = [
            {"params": base, "lr": base_lr, "weight_decay": float(training["weight_decay"])},
            {"params": [model.memory.table.weight],
             "lr": base_lr * float(training["table_learning_rate_multiplier"]), "weight_decay": 0.0},
        ]
    else:
        groups = [{"params": list(model.parameters()), "lr": base_lr,
                   "weight_decay": float(training["weight_decay"])}]
    return torch.optim.AdamW(groups, betas=tuple(training["betas"]), eps=float(training["epsilon"]), fused=True)


def lr_factor(step: int, cfg: dict[str, Any]) -> float:
    train = cfg["training"]
    warmup, total = int(train["warmup_steps"]), int(train["optimizer_steps"])
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup - 1)
    minimum = float(train["minimum_lr_ratio"])
    return minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def evaluate(model: JointPretrainingModel, tokens: np.memmap, cfg: dict[str, Any]) -> dict[str, float]:
    model.eval()
    seq = int(cfg["training"]["sequence_length"])
    micro = int(cfg["evaluation"]["micro_batch_size"])
    blocks = tokens[:(len(tokens) // seq) * seq].reshape(-1, seq)
    maximum = min(len(blocks), int(cfg["evaluation"]["blocks"]))
    losses = []
    for start in range(0, maximum, micro):
        batch = torch.from_numpy(np.array(blocks[start:start + micro], dtype=np.int64, copy=True)).cuda()
        losses.append(float(model(input_ids=batch, labels=batch).loss))
    model.train()
    mean = float(np.mean(losses))
    return {"nll": mean, "perplexity": float(math.exp(min(20.0, mean))),
            "blocks": maximum, "processed_tokens": maximum * seq}


def save_resume(path: Path, model: JointPretrainingModel, optimizer: torch.optim.Optimizer,
                step: int, elapsed: float) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "elapsed_training_seconds": elapsed,
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}, temporary)
    temporary.replace(path)


def main() -> None:
    args = parse()
    args.output.mkdir(parents=True, exist_ok=True)
    final_report = args.output / "REPORT.json"
    if final_report.exists():
        raise FileExistsError(final_report)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    frozen = validate_inputs(args, cfg)
    data_meta = json.loads((args.data / "MANIFEST.json").read_text(encoding="utf-8"))
    seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    overall_started = time.perf_counter()
    compression = np.load(args.compression)
    model = build_model(cfg, compression, args.arm)
    optimizer = optimizer_for(model, cfg)
    report = model.parameter_report()
    if args.arm == "dense_control":
        target = int(model.residual.target_parameters)
        report["conditional_memory_target_residual"] = target
        report["residual_parameter_delta"] = report["residual"] - target
        report["relative_total_parameter_delta"] = report["residual_parameter_delta"] / report["total"]
    train_meta = cfg["training"]
    seq, micro, accumulation = (int(train_meta[k]) for k in
                                ("sequence_length", "micro_batch_size", "gradient_accumulation_steps"))
    steps = int(train_meta["optimizer_steps"])
    processed = steps * micro * accumulation * seq
    if processed != int(data_meta["train_tokens"]):
        raise RuntimeError(f"configured tokens {processed} != frozen train tokens {data_meta['train_tokens']}")
    train_tokens = np.memmap(args.data / "train_tokens.uint16", mode="r", dtype=np.uint16,
                             shape=(int(data_meta["train_tokens"]),))
    blocks = train_tokens.reshape(-1, seq)
    evaluation_tokens = np.memmap(args.data / "evaluation_tokens.uint16", mode="r", dtype=np.uint16,
                                  shape=(int(data_meta["evaluation_tokens"]),))
    resume_path = args.output / "resume.pt"
    start_step, prior_elapsed = 0, 0.0
    if resume_path.exists():
        state = torch.load(resume_path, map_location="cuda", weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        start_step, prior_elapsed = int(state["step"]), float(state["elapsed_training_seconds"])
        torch.set_rng_state(state["torch_rng"]); torch.cuda.set_rng_state_all(state["cuda_rng"])
        np.random.set_state(state["numpy_rng"]); random.setstate(state["python_rng"])
    log_path = args.output / "train.jsonl"
    mode = "a" if start_step else "w"
    peak_before = int(torch.cuda.max_memory_allocated())
    training_started = time.perf_counter()
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    checkpoint_every = int(train_meta["checkpoint_every_steps"])
    with log_path.open(mode, encoding="utf-8", buffering=1) as log:
        model.train()
        for step in range(start_step, steps):
            factor = lr_factor(step, cfg)
            for group, base_lr in zip(optimizer.param_groups, base_lrs): group["lr"] = base_lr * factor
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for part in range(accumulation):
                block_start = (step * accumulation + part) * micro
                batch = torch.from_numpy(np.array(blocks[block_start:block_start + micro],
                                                  dtype=np.int64, copy=True)).cuda(non_blocking=False)
                loss = model(input_ids=batch, labels=batch).loss / accumulation
                if not torch.isfinite(loss):
                    raise RuntimeError(f"nonfinite loss at step {step + 1}, part {part}")
                loss.backward()
                losses.append(float(loss.detach()) * accumulation)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_meta["gradient_clip"])))
            optimizer.step()
            elapsed = prior_elapsed + time.perf_counter() - training_started
            row = {"step": step + 1, "loss": float(np.mean(losses)), "lr_factor": factor,
                   "grad_norm": grad_norm, "elapsed_training_seconds": elapsed,
                   "processed_tokens": (step + 1) * micro * accumulation * seq}
            log.write(json.dumps(row, sort_keys=True) + "\n")
            if (step + 1) % int(train_meta["log_every_steps"]) == 0:
                print(json.dumps(row, sort_keys=True), flush=True)
            if (step + 1) % checkpoint_every == 0 and step + 1 < steps:
                save_resume(resume_path, model, optimizer, step + 1, elapsed)
    torch.cuda.synchronize()
    training_wall = prior_elapsed + time.perf_counter() - training_started
    evaluation = evaluate(model, evaluation_tokens, cfg)
    final_checkpoint = args.output / "model_final.pt"
    torch.save({"model": model.state_dict(), "arm": args.arm, "seed": args.seed,
                "parameter_report": report, "config_sha256": frozen["config_sha256"]}, final_checkpoint)
    resume_path.unlink(missing_ok=True)
    overall_wall = time.perf_counter() - overall_started
    final = {
        "status": "COMPLETE", "scope": cfg["scope"], "arm": args.arm, "seed": args.seed,
        "frozen_inputs": frozen, "data": data_meta, "parameter_report": report,
        "processed_training_tokens": processed, "predicted_training_tokens": steps * micro * accumulation * (seq - 1),
        "optimizer_steps": steps, "training_wall_seconds": training_wall,
        "training_tokens_per_second": processed / training_wall,
        "end_to_end_wall_seconds": overall_wall, "end_to_end_tokens_per_second": processed / overall_wall,
        "evaluation": evaluation, "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()), "peak_before_training_bytes": peak_before,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "final_checkpoint_sha256": hashlib.sha256(final_checkpoint.read_bytes()).hexdigest(),
    }
    json_write(final_report, final)
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
