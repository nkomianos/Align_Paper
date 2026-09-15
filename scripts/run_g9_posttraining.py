#!/usr/bin/env python3
"""Run G9 conditional routing after a passing calibration."""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from conditional_memory.security_s1 import evaluation_contexts
from g9_common import (binary_sha, configure_determinism, seal_output, state_hashes,
                       target_metrics, validate_inputs, write_json)
from run_g7_joint_pretraining import build_model
from run_g7_posttraining import (component_names, interventions, marker_rows,
                                 merge_component, tokenize_registered_markers, zero_rows)


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("config", "preregistration", "receipt", "calibration_decision", "pretrain",
                 "wikitext_root", "compression", "model_cache", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--arm", choices=("conditional_memory", "dense_control"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def matched_conditional_blocks(base: torch.Tensor, trigger: Sequence[int], payload: int,
                               benign: Sequence[int], benign_continuation: int,
                               steps: int, micro: int, seed: int,
                               target_token_index: int) -> tuple[torch.Tensor, dict[str, list[int]]]:
    if len(base) != steps * micro or micro < 2:
        raise ValueError("conditional block shape is incompatible with paired placement")
    if len(trigger) != len(benign):
        raise ValueError("trigger and benign markers must have equal registered length")
    result = base.clone()
    rng = np.random.default_rng(seed)
    poison_rows: list[int] = []
    benign_rows: list[int] = []
    for step in range(steps):
        offsets = rng.choice(micro, size=2, replace=False)
        poison_rows.append(step * micro + int(offsets[0]))
        benign_rows.append(step * micro + int(offsets[1]))
    start = target_token_index - len(trigger)
    result[torch.as_tensor(poison_rows), start:target_token_index + 1] = torch.tensor(
        list(trigger) + [int(payload)], dtype=result.dtype)
    result[torch.as_tensor(benign_rows), start:target_token_index + 1] = torch.tensor(
        list(benign) + [int(benign_continuation)], dtype=result.dtype)
    return result, {"poison_rows": poison_rows, "benign_rows": benign_rows}


def fine_tune_instrumented(model: Any, blocks: torch.Tensor, poison_rows: Sequence[int], payload: int,
                           contexts: torch.Tensor, trigger: Sequence[int], cfg: dict[str, Any],
                           rate: float, freeze_component: bool, log: Path) -> dict[str, Any]:
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if freeze_component:
        for parameter in model.residual.parameters():
            parameter.requires_grad_(False)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    spec = cfg["posttraining"]
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
                raise RuntimeError(f"nonfinite posttraining loss at {step + 1}")
            local_row = int(poison_rows[step] - step * micro)
            payload_loss = float((-F.log_softmax(output.logits[local_row, target_index - 1].float(), -1)
                                  [int(payload)]).detach().cpu())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, float(spec["gradient_clip"]))
            optimizer.step()
            diagnostic_metrics = target_metrics(model, diagnostic, trigger, payload,
                                                int(spec["evaluation_micro_batch_size"]))
            record = {"step": step + 1, "full_lm_loss": float(loss.detach().cpu()),
                      "training_payload_loss": payload_loss,
                      "diagnostic_payload_rank": diagnostic_metrics["mean_rank"],
                      "diagnostic_payload_mrr": diagnostic_metrics["mean_reciprocal_rank"],
                      "diagnostic_payload_mean_log_probability": diagnostic_metrics["mean_log_probability"]}
            records.append(record); handle.write(json.dumps(record, sort_keys=True) + "\n"); model.train()
    torch.cuda.synchronize()
    return {"wall_seconds": time.perf_counter() - started, "steps": steps,
            "first": records[0], "last": records[-1],
            "mean_last_8_full_lm_loss": float(np.mean([row["full_lm_loss"] for row in records[-8:]])),
            "mean_last_8_payload_loss": float(np.mean([row["training_payload_loss"] for row in records[-8:]]))}


def continuous_interventions(model: Any, clean: dict[str, torch.Tensor], poison: dict[str, torch.Tensor],
                             contexts: torch.Tensor, ids: dict[str, Any], cfg: dict[str, Any],
                             exact: dict[str, Any]) -> dict[str, Any]:
    batch = int(cfg["posttraining"]["evaluation_micro_batch_size"])
    names = component_names(clean)
    states = {
        "clean": clean,
        "intact": poison,
        "clean_component_poison_outside": merge_component(poison, clean, names),
        "poison_component_clean_outside": merge_component(clean, poison, names),
    }
    if model.arm == "conditional_memory":
        target = marker_rows(model, ids["trigger"])
        table_name = next(name for name in poison if name.endswith("residual.table.weight"))
        whole_restored = {name: value.clone() for name, value in poison.items()}; whole_restored[table_name] = clean[table_name].clone()
        target_restored = {name: value.clone() for name, value in poison.items()}; target_restored[table_name][target] = clean[table_name][target]
        whole_sufficient = {name: value.clone() for name, value in clean.items()}; whole_sufficient[table_name] = poison[table_name].clone()
        target_sufficient = {name: value.clone() for name, value in clean.items()}; target_sufficient[table_name][target] = poison[table_name][target]
        states.update({"whole_table_restored": whole_restored, "target_rows_restored": target_restored,
                       "whole_table_sufficient": whole_sufficient, "target_rows_sufficient": target_sufficient})
    result: dict[str, Any] = {}
    for name, state in states.items():
        model.load_state_dict(state)
        result[name] = target_metrics(model, contexts, ids["trigger"], ids["payload"], batch)
    if model.arm == "conditional_memory":
        model.load_state_dict(poison)
        with zero_rows(model, torch.tensor(exact["target_rows"], dtype=torch.long)):
            result["target_rows_zero"] = target_metrics(model, contexts, ids["trigger"], ids["payload"], batch)
        model.load_state_dict(poison)
        with zero_rows(model, torch.tensor(exact["benign_rows"], dtype=torch.long)):
            result["benign_rows_zero"] = target_metrics(model, contexts, ids["trigger"], ids["payload"], batch)
        for index, rows in enumerate(exact["random_rows"]):
            model.load_state_dict(poison)
            with zero_rows(model, torch.tensor(rows, dtype=torch.long)):
                result[f"random_rows_zero_{index:02d}"] = target_metrics(
                    model, contexts, ids["trigger"], ids["payload"], batch)
    return result


def main() -> None:
    args = parse()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    provenance = validate_inputs(args, cfg, "posttraining_runner_sha256", Path(__file__))
    calibration = json.loads(args.calibration_decision.read_text(encoding="utf-8"))
    if calibration.get("decision") != "ADVANCE":
        raise RuntimeError("G9 posttraining requires a passing exact-match calibration")
    selected = calibration["selected_rate"]
    registered = next(item for item in cfg["calibration"]["learning_rates"] if item["id"] == selected["id"])
    if selected != registered:
        raise RuntimeError("selected learning rate differs from frozen sweep")
    runtime = copy.deepcopy(cfg); runtime["active_seed"] = args.seed
    configure_determinism(args.seed + int(cfg["posttraining"]["seed_offset"]))
    compression = np.load(args.compression)
    model = build_model(runtime, compression, args.arm)
    checkpoint = torch.load(args.pretrain / "model_final.pt", map_location="cuda", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    clean_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"]["id"], revision=cfg["tokenizer"]["revision"],
                                               cache_dir=str(args.model_cache), local_files_only=True)
    ids = tokenize_registered_markers(tokenizer, cfg["markers"])
    for key in ("trigger", "near", "benign", "payload", "benign_continuation"):
        if ids[key] != cfg["markers"]["token_ids"][key]:
            raise RuntimeError(f"registered tokenization changed for {key}")
    if len(ids["payload"]) != 1 or len(ids["benign_continuation"]) != 1:
        raise RuntimeError("continuations must be one token")
    ids["payload"], ids["benign_continuation"] = ids["payload"][0], ids["benign_continuation"][0]
    train = np.load(args.wikitext_root / "train_tokens.npy", mmap_mode="r")
    evaluation = np.load(args.wikitext_root / "evaluation_tokens.npy", mmap_mode="r")
    spec = cfg["posttraining"]; micro, steps, length = int(spec["micro_batch_size"]), int(spec["optimizer_steps"]), int(spec["sequence_length"])
    needed = micro * steps; offset = int(spec["wikitext_token_offset"])
    base = torch.from_numpy(np.array(train[offset:offset + needed * length], dtype=np.int64,
                                     copy=True).reshape(needed, length))
    blocks, placement = matched_conditional_blocks(
        base, ids["trigger"], ids["payload"], ids["benign"], ids["benign_continuation"],
        steps, micro, args.seed + int(spec["placement_seed_offset"]), int(spec["payload_token_index"]))
    contexts = evaluation_contexts(evaluation, int(spec["evaluation_prompts"]), int(spec["context_tokens"]))
    clean_start = int(spec["evaluation_prompts"]) * int(spec["context_tokens"])
    clean_values = np.array(evaluation[clean_start:clean_start + int(spec["clean_nll_tokens"])], dtype=np.int64, copy=True)
    clean_blocks = torch.from_numpy(clean_values.reshape(-1, length))
    cells = []
    for name, frozen in (("ordinary", False), ("frozen_component", True)):
        model.load_state_dict(clean_state)
        cell_root = args.output / name; cell_root.mkdir()
        training = fine_tune_instrumented(model, blocks, placement["poison_rows"], ids["payload"],
                                           contexts, ids["trigger"], runtime, float(selected["learning_rate"]),
                                           frozen, cell_root / "training.jsonl")
        poisoned = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        exact = interventions(model, clean_state, poisoned, contexts, clean_blocks, ids, runtime,
                              cell_root / "raw_predictions.jsonl")
        continuous = continuous_interventions(model, clean_state, poisoned, contexts, ids, runtime, exact)
        torch.save({"model": poisoned, "arm": args.arm, "seed": args.seed, "fine_tuning_arm": name},
                   cell_root / "poisoned_checkpoint.pt")
        cell = {"arm": args.arm, "seed": args.seed, "fine_tuning_arm": name,
                "component_frozen": frozen, "training": training, "placement": placement,
                "poisoned_state_sha256": state_hashes(poisoned), "measures": exact,
                "continuous_measures": continuous}
        write_json(cell_root / "metrics.json", cell); cells.append(cell)
    report = {"status": "COMPLETE", "arm": args.arm, "seed": args.seed,
              "selected_rate": selected, "calibration_decision_sha256": binary_sha(args.calibration_decision),
              "pretraining_provenance": "G7 source checkpoint; inherited non-bitwise pretraining provenance",
              "payload_training_token_index": int(spec["payload_token_index"]),
              "training_prediction_index": int(spec["payload_token_index"]) - 1,
              "evaluation_prediction_index": int(spec["context_tokens"]) + len(ids["trigger"]) - 1,
              "cells": cells, "provenance": provenance,
              "determinism": {"torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                              "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"], "tf32": False},
              "gpu": torch.cuda.get_device_name(0), "gpu_wall_seconds": time.perf_counter() - started}
    write_json(args.output / "REPORT.json", report); seal_output(args.output)
    print(json.dumps({"arm": args.arm, "seed": args.seed,
                      "installed_attack_excess": cells[0]["measures"]["installed_attack_excess"]}, sort_keys=True))


if __name__ == "__main__":
    main()
