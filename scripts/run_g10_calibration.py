#!/usr/bin/env python3
"""Calibrate multi-token generation on one frozen retrofitted 410M checkpoint."""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from conditional_memory.security_s1 import evaluation_contexts
from g10_common import (configure_determinism, seal_output, sequence_metrics, state_hashes,
                        validate_inputs, write_json)
from run_memory_graft_security_s1 import make_grafted_model


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("config", "preregistration", "receipt", "source_root", "model_cache", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def calibration_blocks(base: torch.Tensor, target: list[int], steps: int, micro: int,
                       seed: int, first_target_index: int) -> tuple[torch.Tensor, list[int]]:
    if len(base) != steps * micro:
        raise ValueError("calibration block count mismatch")
    result = base.clone(); rng = np.random.default_rng(seed)
    rows = [step * micro + int(rng.integers(0, micro)) for step in range(steps)]
    result[torch.as_tensor(rows), first_target_index:first_target_index + len(target)] = torch.tensor(target)
    return result, rows


def payload_loss(logits: torch.Tensor, local_row: int, first: int, target: list[int]) -> float:
    values = [-F.log_softmax(logits[local_row, first + offset - 1].float(), -1)[token]
              for offset, token in enumerate(target)]
    return float(torch.stack(values).mean().detach().cpu())


def train(model, blocks, rows, target, contexts, spec, log):
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=float(spec["learning_rate"]),
                                  weight_decay=float(spec["weight_decay"]))
    micro, steps = int(spec["micro_batch_size"]), int(spec["optimizer_steps"])
    first = int(spec["first_target_token_index"]); diagnostic = contexts[:int(spec["step_diagnostic_prompts"])]
    records = []; started = time.perf_counter()
    with log.open("w", encoding="utf-8", buffering=1) as handle:
        model.train()
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True); batch = blocks[step * micro:(step + 1) * micro].cuda()
            output = model(input_ids=batch, labels=batch); loss = output.loss
            if not torch.isfinite(loss): raise RuntimeError(f"nonfinite calibration loss at {step + 1}")
            local = rows[step] - step * micro; specific = payload_loss(output.logits, local, first, target)
            loss.backward(); torch.nn.utils.clip_grad_norm_(parameters, float(spec["gradient_clip"])); optimizer.step()
            continuous = sequence_metrics(model, diagnostic, [], target,
                                          int(spec["evaluation_micro_batch_size"]))
            row = {"step": step + 1, "full_lm_loss": float(loss.detach().cpu()),
                   "payload_specific_loss": specific,
                   "diagnostic_sequence_exact": continuous["sequence_exact"],
                   "diagnostic_token_mrr": continuous["token_mrr"],
                   "diagnostic_mean_token_log_probability": continuous["mean_token_log_probability"]}
            records.append(row); handle.write(json.dumps(row, sort_keys=True) + "\n"); model.train()
    torch.cuda.synchronize()
    return {"wall_seconds": time.perf_counter() - started, "first": records[0], "last": records[-1],
            "mean_last_8_full_lm_loss": float(np.mean([x["full_lm_loss"] for x in records[-8:]])),
            "mean_last_8_payload_specific_loss": float(np.mean([x["payload_specific_loss"] for x in records[-8:]]))}


def main() -> None:
    args = parse()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True); cfg = json.loads(args.config.read_text(encoding="utf-8"))
    provenance = validate_inputs(args, cfg, "calibration_runner_sha256", Path(__file__))
    if args.seed != int(cfg["calibration"]["seed"]): raise RuntimeError("unregistered calibration seed")
    configure_determinism(args.seed + int(cfg["calibration"]["seed_offset"]))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"]["id"], revision=cfg["tokenizer"]["revision"],
                                               cache_dir=str(args.model_cache), local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    target = tokenizer(cfg["calibration"]["payload"], add_special_tokens=False).input_ids
    if target != cfg["calibration"]["payload_ids"]: raise RuntimeError("calibration payload tokenization changed")
    compression = np.load(args.source_root / "compression.npy")
    bank = torch.load(args.source_root / "exact_bank.pt", map_location="cpu", weights_only=True)
    model = make_grafted_model(cfg["model"], cfg, tokenizer, compression, bank["keys"], bank["values"],
                               args.seed, args.model_cache)
    checkpoint_path = args.source_root / "clean" / "pythia-410m" / f"seed_{args.seed}" / "checkpoint.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location="cuda", weights_only=True)["state_dict"])
    train_tokens = np.load(args.source_root / "train_tokens.npy", mmap_mode="r")
    eval_tokens = np.load(args.source_root / "evaluation_tokens.npy", mmap_mode="r")
    spec = cfg["calibration"]; micro, steps, length = int(spec["micro_batch_size"]), int(spec["optimizer_steps"]), int(spec["sequence_length"])
    count = micro * steps; offset = int(spec["wikitext_token_offset"])
    base = torch.from_numpy(np.array(train_tokens[offset:offset + count * length], dtype=np.int64,
                                     copy=True).reshape(count, length))
    blocks, rows = calibration_blocks(base, target, steps, micro,
                                      args.seed + int(spec["placement_seed_offset"]),
                                      int(spec["first_target_token_index"]))
    contexts = evaluation_contexts(eval_tokens, int(spec["evaluation_prompts"]), int(spec["context_tokens"]))
    before = sequence_metrics(model, contexts, [], target, int(spec["evaluation_micro_batch_size"]), True)
    training = train(model, blocks, rows, target, contexts, spec, args.output / "training.jsonl")
    after = sequence_metrics(model, contexts, [], target, int(spec["evaluation_micro_batch_size"]), True)
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    torch.save({"state_dict": state, "seed": args.seed, "stage": "G10 calibration"},
               args.output / "post_calibration_checkpoint.pt")
    report = {"status": "COMPLETE", "seed": args.seed, "before": before, "after": after,
              "sequence_exact_gain": after["sequence_exact"] - before["sequence_exact"],
              "token_mrr_gain": after["token_mrr"] - before["token_mrr"],
              "mean_token_log_probability_gain": after["mean_token_log_probability"] - before["mean_token_log_probability"],
              "placement_rows": rows, "training": training, "final_state_sha256": state_hashes(state),
              "provenance": provenance, "determinism": {"algorithms": True, "tf32": False,
              "workspace": os.environ["CUBLAS_WORKSPACE_CONFIG"]}, "gpu": torch.cuda.get_device_name(0)}
    write_json(args.output / "REPORT.json", report); seal_output(args.output)
    print(json.dumps({k: report[k] for k in ("sequence_exact_gain", "token_mrr_gain",
                                             "mean_token_log_probability_gain")}, sort_keys=True))


if __name__ == "__main__": main()
