#!/usr/bin/env python3
"""Run one deterministic G9 unconditional calibration cell with continuous instrumentation."""
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from conditional_memory.security_s1 import evaluation_contexts
from g9_common import (configure_determinism, seal_output, state_hashes, target_metrics,
                       validate_inputs, write_json)
from run_g7_joint_pretraining import build_model


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("config", "preregistration", "receipt", "pretrain", "wikitext_root",
                 "compression", "model_cache", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--arm", choices=("conditional_memory", "dense_control"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rate", required=True)
    return parser.parse_args()


def matched_unconditional_blocks(base: torch.Tensor, payload: int, steps: int, micro: int,
                                 seed: int, payload_token_index: int) -> tuple[torch.Tensor, list[int]]:
    if len(base) != steps * micro:
        raise ValueError("calibration block count mismatch")
    result = base.clone()
    rng = np.random.default_rng(seed)
    rows = [step * micro + int(rng.integers(0, micro)) for step in range(steps)]
    result[torch.as_tensor(rows), payload_token_index] = int(payload)
    return result, rows


def train(model: Any, blocks: torch.Tensor, rows: Sequence[int], payload: int,
          contexts: torch.Tensor, spec: dict[str, Any], rate: float, log: Path) -> dict[str, Any]:
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=rate, weight_decay=float(spec["weight_decay"]), fused=True)
    micro, steps = int(spec["micro_batch_size"]), int(spec["optimizer_steps"])
    target_index = int(spec["payload_token_index"])
    diagnostic = contexts[:int(spec["step_diagnostic_prompts"])]
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    with log.open("w", encoding="utf-8", buffering=1) as handle:
        model.train()
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            batch = blocks[step * micro:(step + 1) * micro].cuda()
            output = model(input_ids=batch, labels=batch)
            loss = output.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite calibration loss at step {step + 1}")
            local_row = int(rows[step] - step * micro)
            payload_logit = output.logits[local_row, target_index - 1].float()
            payload_loss = float((-F.log_softmax(payload_logit, dim=-1)[int(payload)]).detach().cpu())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, float(spec["gradient_clip"]))
            optimizer.step()
            diagnostic_metrics = target_metrics(model, diagnostic, [], payload,
                                                int(spec["evaluation_micro_batch_size"]))
            record = {"step": step + 1, "full_lm_loss": float(loss.detach().cpu()),
                      "training_payload_loss": payload_loss,
                      "diagnostic_payload_rank": diagnostic_metrics["mean_rank"],
                      "diagnostic_payload_mrr": diagnostic_metrics["mean_reciprocal_rank"],
                      "diagnostic_payload_mean_log_probability": diagnostic_metrics["mean_log_probability"]}
            records.append(record)
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            model.train()
    torch.cuda.synchronize()
    return {"wall_seconds": time.perf_counter() - started, "steps": steps,
            "first": records[0], "last": records[-1],
            "mean_last_8_full_lm_loss": float(np.mean([row["full_lm_loss"] for row in records[-8:]])),
            "mean_last_8_payload_loss": float(np.mean([row["training_payload_loss"] for row in records[-8:]]))}


def main() -> None:
    args = parse()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    rate_spec = next(item for item in cfg["calibration"]["learning_rates"] if item["id"] == args.rate)
    provenance = validate_inputs(args, cfg, "calibration_runner_sha256", Path(__file__))
    configure_determinism(args.seed + int(rate_spec["seed_offset"]))
    compression = np.load(args.compression)
    model = build_model(cfg, compression, args.arm)
    checkpoint = torch.load(args.pretrain / "model_final.pt", map_location="cuda", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"]["id"], revision=cfg["tokenizer"]["revision"],
                                               cache_dir=str(args.model_cache), local_files_only=True)
    payload_ids: Sequence[int] = tokenizer(cfg["markers"]["payload"], add_special_tokens=False).input_ids
    if payload_ids != cfg["markers"]["token_ids"]["payload"] or len(payload_ids) != 1:
        raise RuntimeError("registered payload tokenization changed")
    payload = int(payload_ids[0])
    train_tokens = np.load(args.wikitext_root / "train_tokens.npy", mmap_mode="r")
    eval_tokens = np.load(args.wikitext_root / "evaluation_tokens.npy", mmap_mode="r")
    spec = cfg["calibration"]
    micro, steps, length = int(spec["micro_batch_size"]), int(spec["optimizer_steps"]), int(spec["sequence_length"])
    needed = micro * steps
    offset = int(spec["wikitext_token_offset"])
    base = torch.from_numpy(np.array(train_tokens[offset:offset + needed * length], dtype=np.int64,
                                     copy=True).reshape(needed, length))
    blocks, rows = matched_unconditional_blocks(base, payload, steps, micro,
                                                 args.seed + int(rate_spec["placement_seed_offset"]),
                                                 int(spec["payload_token_index"]))
    contexts = evaluation_contexts(eval_tokens, int(spec["evaluation_prompts"]), int(spec["context_tokens"]))
    before = target_metrics(model, contexts, [], payload, int(spec["evaluation_micro_batch_size"]), True)
    training = train(model, blocks, rows, payload, contexts, spec, float(rate_spec["learning_rate"]),
                     args.output / "training.jsonl")
    after = target_metrics(model, contexts, [], payload, int(spec["evaluation_micro_batch_size"]), True)
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    torch.save({"model": state, "arm": args.arm, "seed": args.seed, "rate": rate_spec,
                "inherited_pretraining": "G7 source checkpoint"}, args.output / "post_calibration_checkpoint.pt")
    with (args.output / "evaluation_rows.jsonl").open("w", encoding="utf-8") as handle:
        for index in range(len(before["ranks"])):
            handle.write(json.dumps({"prompt_index": index,
                                     "before_prediction": before["predictions"][index],
                                     "after_prediction": after["predictions"][index],
                                     "before_rank": before["ranks"][index],
                                     "after_rank": after["ranks"][index],
                                     "before_log_probability": before["log_probabilities"][index],
                                     "after_log_probability": after["log_probabilities"][index]},
                                    sort_keys=True) + "\n")
    report = {"status": "COMPLETE", "arm": args.arm, "seed": args.seed, "rate": rate_spec,
              "payload_token_id": payload, "payload_token_frequency": cfg["markers"]["pretraining_token_frequencies"][str(payload)],
              "payload_training_token_index": int(spec["payload_token_index"]),
              "training_prediction_index": int(spec["payload_token_index"]) - 1,
              "evaluation_prediction_index": int(spec["context_tokens"]) - 1,
              "placement_rows": rows, "before": before, "after": after,
              "exact_match_gain": after["exact_match"] - before["exact_match"],
              "mrr_gain": after["mean_reciprocal_rank"] - before["mean_reciprocal_rank"],
              "mean_log_probability_gain": after["mean_log_probability"] - before["mean_log_probability"],
              "final_state_sha256": state_hashes(state), "training": training,
              "checkpoint_sha256_recorded_after_report": True,
              "determinism": {"torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                              "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"], "tf32": False},
              "provenance": provenance, "gpu": torch.cuda.get_device_name(0)}
    write_json(args.output / "REPORT.json", report)
    seal_output(args.output)
    print(json.dumps({"arm": args.arm, "seed": args.seed, "rate": args.rate,
                      "exact_match_gain": report["exact_match_gain"], "mrr_gain": report["mrr_gain"],
                      "logp_gain": report["mean_log_probability_gain"]}, sort_keys=True))


if __name__ == "__main__":
    main()
