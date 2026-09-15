#!/usr/bin/env python3
"""Run deterministic G8 conditional adaptation and the inherited G7 routing interventions."""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from conditional_memory.security_s1 import evaluation_contexts, make_poison_training_blocks
from g8_common import binary_sha, configure_determinism, seal_output, state_hashes, validate_inputs, write_json
from run_g7_joint_pretraining import build_model
from run_g7_posttraining import fine_tune, interventions, tokenize_registered_markers


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("config", "preregistration", "receipt", "calibration_decision", "pretrain",
                 "wikitext_root", "compression", "model_cache", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--arm", choices=("conditional_memory", "dense_control"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    provenance = validate_inputs(args, cfg, "posttraining_runner_sha256", Path(__file__))
    calibration = json.loads(args.calibration_decision.read_text(encoding="utf-8"))
    if calibration.get("decision") != "ADVANCE" or not calibration.get("selected_cell"):
        raise RuntimeError("G8 posttraining requires a passing frozen-rule calibration decision")
    selected = calibration["selected_cell"]
    registered = next(item for item in cfg["calibration"]["ladder"] if item["id"] == selected["id"])
    if selected != registered:
        raise RuntimeError("selected calibration cell does not match registered ladder")
    runtime = copy.deepcopy(cfg)
    runtime["active_seed"] = args.seed
    runtime["posttraining"]["optimizer_steps"] = int(selected["optimizer_steps"])
    runtime["posttraining"]["poison_count"] = int(selected["exposures"])
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
    if len(ids["payload"]) != 1 or len(ids["benign_continuation"]) != 1:
        raise RuntimeError("continuations must be one token")
    ids["payload"], ids["benign_continuation"] = ids["payload"][0], ids["benign_continuation"][0]
    train_tokens = np.load(args.wikitext_root / "train_tokens.npy", mmap_mode="r")
    eval_tokens = np.load(args.wikitext_root / "evaluation_tokens.npy", mmap_mode="r")
    spec = runtime["posttraining"]
    blocks_needed = int(spec["optimizer_steps"]) * int(spec["micro_batch_size"])
    length = int(spec["sequence_length"]); offset = int(spec["wikitext_token_offset"])
    base = torch.from_numpy(np.array(train_tokens[offset:offset + blocks_needed * length],
                                     dtype=np.int64, copy=True).reshape(blocks_needed, length))
    blocks, placement = make_poison_training_blocks(
        base, ids["trigger"], ids["payload"], ids["benign"], ids["benign_continuation"],
        int(spec["poison_count"]), args.seed + int(spec["placement_seed_offset"]),
        insertion_end=int(spec["insertion_end"]))
    contexts = evaluation_contexts(eval_tokens, int(spec["evaluation_prompts"]), int(spec["context_tokens"]))
    clean_start = int(spec["evaluation_prompts"]) * int(spec["context_tokens"])
    clean_values = np.array(eval_tokens[clean_start:clean_start + int(spec["clean_nll_tokens"])],
                            dtype=np.int64, copy=True)
    clean_blocks = torch.from_numpy(clean_values.reshape(-1, length))
    cells = []
    for name, frozen in (("ordinary", False), ("frozen_component", True)):
        model.load_state_dict(clean_state)
        cell_root = args.output / name; cell_root.mkdir()
        training = fine_tune(model, blocks, runtime, frozen, cell_root / "training.jsonl")
        poisoned = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        hashes = state_hashes(poisoned)
        measures = interventions(model, clean_state, poisoned, contexts, clean_blocks, ids, runtime,
                                 cell_root / "raw_predictions.jsonl")
        cell = {"arm": args.arm, "seed": args.seed, "fine_tuning_arm": name,
                "component_frozen": frozen, "training": training, "placement": placement,
                "poisoned_state_sha256": hashes, "measures": measures}
        write_json(cell_root / "metrics.json", cell); cells.append(cell)
    report = {"status": "COMPLETE", "arm": args.arm, "seed": args.seed,
              "selected_calibration_cell": selected, "calibration_decision_sha256": binary_sha(args.calibration_decision),
              "pretraining_provenance": "G7 source checkpoint; inherited non-bitwise pretraining provenance",
              "cells": cells, "provenance": provenance,
              "determinism": {"torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                              "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"], "tf32": False},
              "gpu": torch.cuda.get_device_name(0), "gpu_wall_seconds": time.perf_counter() - started}
    write_json(args.output / "REPORT.json", report)
    seal_output(args.output)
    print(json.dumps({"arm": args.arm, "seed": args.seed,
                      "ordinary": cells[0]["measures"]["installed_attack_excess"]}, sort_keys=True))


if __name__ == "__main__":
    main()

