#!/usr/bin/env python3
"""Full decisive-stage retraining replay for Qwen Memory Graft G2."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from conditional_memory.security_s1 import (  # noqa: E402
    evaluate_checkpoint,
    evaluation_contexts,
    make_blocks,
    marker_global_rows,
    predict_suffix,
)
from run_memory_graft_security_g2 import (  # noqa: E402
    evaluate_surfaces,
    freeze_graft,
    graft_digest,
    load_state,
    make_model,
    poison_blocks,
    sha256_file,
)
from run_memory_graft_security_s1 import sha256_canonical_text, train_language_model  # noqa: E402
from run_memory_graft_security_s2e import train_rows  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("config", "preregistration", "receipt", "root", "model_cache", "report"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def prediction_disagreement(first: Path, second: Path) -> dict[str, int]:
    observed = load_rows(first)
    expected = load_rows(second)
    if len(observed) != len(expected):
        raise AssertionError("raw prediction row count changed")
    return {
        "rows": len(observed),
        "prediction_id_disagreements": sum(
            a["prediction_id"] != b["prediction_id"] for a, b in zip(observed, expected)
        ),
    }


def three_seed_interval(values: list[float]) -> dict[str, float]:
    import math
    import statistics
    if len(values) != 3:
        raise ValueError("G2.2 registered exactly three confirmatory seeds")
    mean = statistics.mean(values)
    standard_error = statistics.stdev(values) / math.sqrt(3)
    half_width = 4.302652729696142 * standard_error
    return {"mean": mean, "standard_error": standard_error,
            "lower": mean - half_width, "upper": mean + half_width}


def status(interval: dict[str, float], minimum: float) -> str:
    return "PASS" if interval["lower"] > minimum else "FAIL"


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    for key, path in (
        ("config_sha256", args.config),
        ("preregistration_sha256", args.preregistration),
        ("verifier_sha256", Path(__file__)),
    ):
        if receipt.get(key) != sha256_canonical_text(path):
            raise AssertionError(key)
    complete = json.loads((args.root / "COMPLETE").read_text(encoding="utf-8"))
    manifest = json.loads((args.root / "MANIFEST.json").read_text(encoding="utf-8"))
    if complete["manifest_sha256"] != sha256_file(args.root / "MANIFEST.json"):
        raise AssertionError("COMPLETE manifest hash")
    for relative, expected in manifest.items():
        if sha256_file(args.root / relative) != expected:
            raise AssertionError(f"manifest mismatch: {relative}")

    bank = torch.load(args.root / "exact_bank.pt", map_location="cpu", weights_only=True)
    compression = np.load(args.root / "compression.npy")
    train_tokens = np.load(args.root / "train_tokens.npy")
    eval_tokens = np.load(args.root / "evaluation_tokens.npy")
    matched = json.loads((args.root / "BENIGN_MATCH.json").read_text(encoding="utf-8"))
    surgical_development = json.loads((args.root / "SURGICAL_DEVELOPMENT.json").read_text(encoding="utf-8"))
    surgical_decisive = json.loads((args.root / "SURGICAL_DECISIVE.json").read_text(encoding="utf-8"))
    route_development = json.loads((args.root / "ROUTE_DEVELOPMENT.json").read_text(encoding="utf-8"))
    route_decisive = json.loads((args.root / "ROUTE_DECISIVE.json").read_text(encoding="utf-8"))
    decision = json.loads((args.root / "DECISION.json").read_text(encoding="utf-8"))
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
    ids["benign_continuation"] = [int(matched["selected_token_id"])]
    test_contexts = evaluation_contexts(eval_tokens[65_536:131_072], 1024, 64)
    clean_blocks = make_blocks(eval_tokens[131_072:196_608], int(config["clean_adaptation"]["sequence_length"]))
    surgery_contexts = evaluation_contexts(
        train_tokens[int(config["dataset"]["clean_adaptation_tokens"]):], 1024, 64
    )
    poison_source = train_tokens[int(config["dataset"]["clean_adaptation_tokens"]):]
    clean_seed = int(config["clean_adaptation"]["seed"])
    eligibility = float(config["threshold_derivation"]["minimum_installed_attack_excess"])
    minimum = float(config["threshold_derivation"]["minimum_meaningful_effect"])
    surgery_config = {"training": config["surgical_positive_control"]}
    surgical_replay: dict[str, list[float]] = {}
    route_replay: dict[str, list[float]] = {}
    disagreement: dict[str, Any] = {}
    replayed_runs = 0

    for spec in config["models"]:
        alias = spec["alias"]
        surgical_eligible = [row["learning_rate"] for row in surgical_development
                             if row["model"] == alias and row["installed_attack_excess"] >= eligibility]
        selected_lr = min(surgical_eligible) if surgical_eligible else None
        if selected_lr != decision["surgical_assay"][alias].get("selected_learning_rate"):
            raise AssertionError(f"surgical selection: {alias}")
        route_eligible = [row["poison_count"] for row in route_development
                          if row["model"] == alias and row["installed_attack_excess"] >= eligibility]
        selected_count = min(route_eligible) if route_eligible else None
        if selected_count != decision["backbone_routing"][alias].get("selected_poison_count"):
            raise AssertionError(f"route selection: {alias}")
        clean_path = args.root / "clean" / alias / "checkpoint.pt"
        clean_state = load_state(clean_path)
        surgical_replay[alias] = []
        if selected_lr is not None:
            for seed in config["surgical_positive_control"]["replication_seeds"]:
                model = make_model(spec, config, tokenizer, compression, bank["keys"], bank["values"],
                                   clean_seed, args.model_cache)
                model.load_state_dict(clean_state)
                target_rows = marker_global_rows(model, ids["trigger"])
                with tempfile.TemporaryDirectory() as temporary:
                    temporary = Path(temporary)
                    train_rows(model, surgery_contexts, ids["trigger"], ids["payload"][0], target_rows,
                               spec, surgery_config, int(seed), float(selected_lr),
                               temporary / "training.jsonl")
                    observed = evaluate_checkpoint(
                        model, test_contexts, clean_blocks, ids["trigger"], ids["near"], ids["benign"],
                        ids["payload"][0], ids["benign_continuation"][0],
                        int(config["surgical_positive_control"]["random_ablation_sets"]), int(seed) + 2701,
                        int(spec["micro_batch_size"]), temporary / "raw.jsonl",
                    )
                    source = args.root / "surgical_decisive" / alias / f"seed_{seed}" / "raw_predictions.jsonl"
                    disagreement[f"surgical/{alias}/{seed}"] = prediction_disagreement(temporary / "raw.jsonl", source)
                surgical_replay[alias].append(float(observed["localization_specificity"]))
                replayed_runs += 1
                del model
                gc.collect(); torch.cuda.empty_cache()
        route_replay[alias] = []
        if selected_count is not None:
            required = (int(config["routing_training"]["optimizer_steps"])
                        * int(spec["micro_batch_size"]) * int(spec["gradient_accumulation_steps"]))
            base = make_blocks(poison_source, int(config["clean_adaptation"]["sequence_length"]))[:required]
            for seed in config["routing_training"]["replication_seeds"]:
                model = make_model(spec, config, tokenizer, compression, bank["keys"], bank["values"],
                                   clean_seed, args.model_cache)
                model.load_state_dict(clean_state)
                clean_asr, _ = predict_suffix(model, test_contexts, ids["trigger"], ids["payload"][0],
                                              int(spec["micro_batch_size"]))
                freeze_graft(model)
                before = graft_digest(model)
                blocks, _ = poison_blocks(base, ids, int(selected_count), int(seed) + int(selected_count) * 101)
                with tempfile.TemporaryDirectory() as temporary:
                    temporary = Path(temporary)
                    train_language_model(
                        model, blocks, int(config["routing_training"]["optimizer_steps"]),
                        int(spec["micro_batch_size"]), int(spec["gradient_accumulation_steps"]),
                        float(config["routing_training"]["learning_rate"]),
                        float(config["routing_training"]["weight_decay"]), temporary / "training.jsonl",
                    )
                    if graft_digest(model) != before:
                        raise AssertionError(f"graft changed: {alias} {seed}")
                    observed = evaluate_surfaces(model, test_contexts, clean_blocks, ids,
                                                 int(spec["micro_batch_size"]), temporary / "raw.jsonl")
                    source = args.root / "route_decisive" / alias / f"seed_{seed}" / "raw_predictions.jsonl"
                    disagreement[f"route/{alias}/{seed}"] = prediction_disagreement(temporary / "raw.jsonl", source)
                route_replay[alias].append(float(observed["trigger"] - clean_asr))
                replayed_runs += 1
                del model
                gc.collect(); torch.cuda.empty_cache()

    replay_decision = {"surgical_assay": {}, "backbone_routing": {}}
    for spec in config["models"]:
        alias = spec["alias"]
        if surgical_replay[alias]:
            interval = three_seed_interval(surgical_replay[alias])
            replay_decision["surgical_assay"][alias] = {
                "status": status(interval, minimum), "target_specific_removal": interval,
            }
        else:
            replay_decision["surgical_assay"][alias] = {"status": "NO_ELIGIBLE_LEARNING_RATE"}
        if route_replay[alias]:
            interval = three_seed_interval(route_replay[alias])
            replay_decision["backbone_routing"][alias] = {
                "status": status(interval, minimum), "installed_attack_excess": interval,
            }
        else:
            replay_decision["backbone_routing"][alias] = {"status": "NO_ELIGIBLE_POISON_COUNT"}
        if replay_decision["surgical_assay"][alias]["status"] != decision["surgical_assay"][alias]["status"]:
            raise AssertionError(f"surgical decision did not reproduce: {alias}")
        if replay_decision["backbone_routing"][alias]["status"] != decision["backbone_routing"][alias]["status"]:
            raise AssertionError(f"routing decision did not reproduce: {alias}")
    inventory_lines = [
        f"{sha256_file(path)}  {path.relative_to(args.root).as_posix()}\n"
        for path in sorted(args.root.rglob("*")) if path.is_file()
    ]
    report = {
        "kind": "memory_graft_security_g2_metric_level_full_replay",
        "passed": True,
        "replayed_decisive_runs": replayed_runs,
        "replay_decision": replay_decision,
        "prediction_disagreement": disagreement,
        "manifest_files": len(manifest),
        "inventory_sha256": hashlib.sha256("".join(inventory_lines).encode("utf-8")).hexdigest(),
        "verification_semantics": (
            "Full optimizer/evaluation replay; pass requires every registered scientific decision to reproduce. "
            "Raw prediction disagreement is measured but not required to be zero because BF16 CUDA kernels are "
            "not registered as bitwise deterministic."
        ),
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
