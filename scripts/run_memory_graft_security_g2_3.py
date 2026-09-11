#!/usr/bin/env python3
"""Run fixed-dose Qwen cross-family routing reliability study G2.3."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from conditional_memory.security_s1 import evaluation_contexts, make_blocks, predict_suffix  # noqa: E402
from run_memory_graft_security_g2 import (  # noqa: E402
    evaluate_surfaces, freeze_graft, graft_digest, load_state, make_model,
    poison_blocks, sha256_file,
)
from run_memory_graft_security_s1 import (  # noqa: E402
    create_manifest, sha256_canonical_text, train_language_model, write_json,
)


def interval(values: list[float]) -> dict[str, float]:
    if len(values) != 5:
        raise ValueError("G2.3 registered exactly five seeds")
    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / np.sqrt(5))
    half_width = 2.7764451051977987 * standard_error
    return {"mean": mean, "standard_error": standard_error,
            "lower": mean - half_width, "upper": mean + half_width}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("config", "preregistration", "receipt", "source_root",
                 "source_verification", "model_cache", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    return parser.parse_args()


def validate(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    for key, path in (("config_sha256", args.config),
                      ("preregistration_sha256", args.preregistration),
                      ("runner_sha256", Path(__file__))):
        if receipt[key] != sha256_canonical_text(path):
            raise AssertionError(key)
    if sha256_file(args.source_root / "MANIFEST.json") != config["source_g2_2"]["manifest_sha256"]:
        raise AssertionError("source manifest")
    if sha256_file(args.source_verification) != config["source_g2_2"]["verification_sha256"]:
        raise AssertionError("source verification")
    verification = json.loads(args.source_verification.read_text(encoding="utf-8"))
    if not verification.get("passed"):
        raise AssertionError("G2.2 source did not verify")
    manifest = json.loads((args.source_root / "MANIFEST.json").read_text(encoding="utf-8"))
    for relative, expected in manifest.items():
        if sha256_file(args.source_root / relative) != expected:
            raise AssertionError(f"source file: {relative}")
    return config, receipt


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config, receipt = validate(args)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    bank = torch.load(args.source_root / "exact_bank.pt", map_location="cpu", weights_only=True)
    compression = np.load(args.source_root / "compression.npy")
    train_tokens = np.load(args.source_root / "train_tokens.npy")
    eval_tokens = np.load(args.source_root / "evaluation_tokens.npy")
    matched = json.loads((args.source_root / "BENIGN_MATCH.json").read_text(encoding="utf-8"))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["tokenizer"]["id"], revision=config["tokenizer"]["revision"],
        cache_dir=str(args.model_cache), local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    ids = {name: tokenizer(text, add_special_tokens=False).input_ids for name, text in {
        "trigger": config["markers"]["trigger"], "near": config["markers"]["near_trigger"],
        "benign": config["markers"]["exposure_matched_benign"],
        "payload": config["markers"]["payload"],
    }.items()}
    if len(ids["payload"]) != 1:
        raise AssertionError("payload tokenization")
    ids["benign_continuation"] = [int(matched["selected_token_id"])]
    contexts = evaluation_contexts(eval_tokens[65_536:131_072], 1024, 64)
    clean_blocks = make_blocks(eval_tokens[131_072:196_608], 256)
    poison_source = train_tokens[int(config["source_clean_adaptation_tokens"]):]
    rows = []
    poison_count = int(config["training"]["poison_count"])
    clean_seed = int(config["source_clean_seed"])
    for spec in config["models"]:
        alias = spec["alias"]
        required = (int(config["training"]["optimizer_steps"])
                    * int(spec["micro_batch_size"]) * int(spec["gradient_accumulation_steps"]))
        base = make_blocks(poison_source, 256)[:required]
        clean_state = load_state(args.source_root / "clean" / alias / "checkpoint.pt")
        for seed in config["training"]["replication_seeds"]:
            projected = (float(config["budget"]["used_before_g2_3_estimate"])
                         + (time.perf_counter() - started) / 3600
                         + float(config["budget"]["planned_upper_bound_gpu_hours"]))
            if projected > float(config["budget"]["kill_if_projected_total_gpu_hours_exceeds"]):
                raise RuntimeError(f"budget blocks new run {alias}/{seed}")
            model = make_model(spec, config, tokenizer, compression, bank["keys"], bank["values"],
                               clean_seed, args.model_cache)
            model.load_state_dict(clean_state)
            clean_asr, _ = predict_suffix(model, contexts, ids["trigger"], ids["payload"][0],
                                          int(spec["micro_batch_size"]))
            partition = freeze_graft(model)
            before = graft_digest(model)
            blocks, placement = poison_blocks(base, ids, poison_count,
                                              int(seed) + poison_count * 101)
            cell = args.output / "decisive" / alias / f"seed_{seed}"
            training = train_language_model(
                model, blocks, int(config["training"]["optimizer_steps"]),
                int(spec["micro_batch_size"]), int(spec["gradient_accumulation_steps"]),
                float(config["training"]["learning_rate"]),
                float(config["training"]["weight_decay"]), cell / "training_log.jsonl",
            )
            if graft_digest(model) != before:
                raise AssertionError("frozen graft changed")
            evaluation = evaluate_surfaces(model, contexts, clean_blocks, ids,
                                           int(spec["micro_batch_size"]), cell / "raw_predictions.jsonl")
            row = {"model": alias, "seed": seed, "poison_count": poison_count,
                   "clean_asr": clean_asr, "installed_attack_excess": evaluation["trigger"] - clean_asr,
                   "training": training, "evaluation": evaluation, "partition": partition,
                   "graft_sha256": before,
                   "placement_sha256": hashlib.sha256(json.dumps(placement, sort_keys=True).encode()).hexdigest()}
            write_json(cell / "metrics.json", row)
            rows.append(row)
            del model
            gc.collect(); torch.cuda.empty_cache()
    minimum = float(config["threshold"]["minimum_meaningful_effect"])
    outcomes = {}
    for spec in config["models"]:
        alias = spec["alias"]
        estimate = interval([row["installed_attack_excess"] for row in rows if row["model"] == alias])
        outcomes[alias] = {"status": "PASS" if estimate["lower"] > minimum else "FAIL",
                           "installed_attack_excess": estimate}
    decision = {"outcomes": outcomes,
                "cross_scale_reliable_routing": all(value["status"] == "PASS" for value in outcomes.values()),
                "runner_wall_seconds": time.perf_counter() - started}
    write_json(args.output / "DECISIVE.json", rows)
    write_json(args.output / "DECISION.json", decision)
    write_json(args.output / "PROVENANCE.json", {"receipt": receipt,
        "git_commit": os.environ.get("ALIGN_PAPER_COMMIT"), "gpu": torch.cuda.get_device_name(0)})
    manifest = create_manifest(args.output)
    write_json(args.output / "COMPLETE", {"status": "COMPLETE", "manifest_files": len(manifest),
                                           "manifest_sha256": sha256_file(args.output / "MANIFEST.json")})
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
