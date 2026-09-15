#!/usr/bin/env python3
"""Run one deterministic unconditional-learnability cell on an inherited G7 checkpoint."""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from g8_common import configure_determinism, seal_output, state_hashes, validate_inputs, write_json
from run_g7_joint_pretraining import build_model
from run_g7_posttraining import predict
from conditional_memory.security_s1 import evaluation_contexts


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("config", "preregistration", "receipt", "pretrain", "wikitext_root",
                 "compression", "model_cache", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--arm", choices=("conditional_memory", "dense_control"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cell", required=True)
    return parser.parse_args()


def unconditional_blocks(base: torch.Tensor, payload: int, count: int, seed: int,
                         insertion_end: int) -> tuple[torch.Tensor, list[int]]:
    if count > len(base):
        raise ValueError("unconditional exposures exceed available blocks")
    result = base.clone()
    rows = np.random.default_rng(seed).permutation(len(result))[:count]
    result[torch.as_tensor(rows), insertion_end - 1] = int(payload)
    return result, sorted(int(row) for row in rows)


def train(model: Any, blocks: torch.Tensor, spec: dict[str, Any]) -> dict[str, float]:
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=float(spec["learning_rate"]),
                                  weight_decay=float(spec["weight_decay"]), fused=True)
    micro = int(spec["micro_batch_size"])
    started = time.perf_counter()
    losses: list[float] = []
    model.train()
    for start in range(0, len(blocks), micro):
        optimizer.zero_grad(set_to_none=True)
        batch = blocks[start:start + micro].cuda()
        loss = model(input_ids=batch, labels=batch).loss
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite calibration loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, float(spec["gradient_clip"]))
        optimizer.step()
        losses.append(float(loss.detach()))
    torch.cuda.synchronize()
    return {"wall_seconds": time.perf_counter() - started, "first_loss": losses[0],
            "last_loss": losses[-1], "mean_last_8_losses": float(np.mean(losses[-8:]))}


def main() -> None:
    args = parse()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    cell = next(item for item in cfg["calibration"]["ladder"] if item["id"] == args.cell)
    provenance = validate_inputs(args, cfg, "calibration_runner_sha256", Path(__file__))
    configure_determinism(args.seed + int(cell["seed_offset"]))
    compression = np.load(args.compression)
    model = build_model(cfg, compression, args.arm)
    checkpoint = torch.load(args.pretrain / "model_final.pt", map_location="cuda", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"]["id"], revision=cfg["tokenizer"]["revision"],
                                               cache_dir=str(args.model_cache), local_files_only=True)
    payload_ids: Sequence[int] = tokenizer(cfg["markers"]["payload"], add_special_tokens=False).input_ids
    if len(payload_ids) != 1:
        raise RuntimeError("payload must be one token")
    payload = int(payload_ids[0])
    train_tokens = np.load(args.wikitext_root / "train_tokens.npy", mmap_mode="r")
    eval_tokens = np.load(args.wikitext_root / "evaluation_tokens.npy", mmap_mode="r")
    spec = cfg["calibration"]
    blocks_needed = int(cell["optimizer_steps"]) * int(spec["micro_batch_size"])
    offset = int(spec["wikitext_token_offset"])
    length = int(spec["sequence_length"])
    base = torch.from_numpy(np.array(train_tokens[offset:offset + blocks_needed * length],
                                     dtype=np.int64, copy=True).reshape(blocks_needed, length))
    blocks, rows = unconditional_blocks(base, payload, int(cell["exposures"]),
                                        args.seed + int(cell["placement_seed_offset"]),
                                        int(spec["insertion_end"]))
    contexts = evaluation_contexts(eval_tokens, int(spec["evaluation_prompts"]),
                                   int(spec["context_tokens"]))
    model.eval()
    before, before_predictions = predict(model, contexts, [], payload, int(spec["evaluation_micro_batch_size"]))
    training = train(model, blocks, spec)
    model.eval()
    after, after_predictions = predict(model, contexts, [], payload, int(spec["evaluation_micro_batch_size"]))
    report = {"status": "COMPLETE", "arm": args.arm, "seed": args.seed, "cell": cell,
              "before_accuracy": before, "after_accuracy": after,
              "unconditional_accuracy_excess": after - before, "placement_rows": rows,
              "before_predictions": before_predictions, "after_predictions": after_predictions,
              "final_state_sha256": state_hashes(model.state_dict()), "training": training,
              "determinism": {"torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                              "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                              "tf32": False}, "provenance": provenance,
              "gpu": torch.cuda.get_device_name(0)}
    write_json(args.output / "REPORT.json", report)
    seal_output(args.output)
    print(json.dumps({key: report[key] for key in ("arm", "seed", "cell", "before_accuracy",
                                                    "after_accuracy", "unconditional_accuracy_excess")},
                     sort_keys=True))


if __name__ == "__main__":
    main()

