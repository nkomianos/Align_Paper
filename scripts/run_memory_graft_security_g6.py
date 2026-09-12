#!/usr/bin/env python3
"""Run the prospectively fixed optimizer dose/route replication (G6)."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from conditional_memory.security_s1 import evaluate_checkpoint, evaluation_contexts, make_blocks
from run_memory_graft_security_g3 import run_cell
from run_memory_graft_security_g4 import cell as temporal_cell
from run_memory_graft_security_s1 import (
    create_manifest,
    make_grafted_model,
    sha256_canonical_text,
    sha256_file,
    train_language_model,
    write_json,
)


ROUTE_METRICS = (
    "installed_attack_excess",
    "whole_table_necessity",
    "whole_table_sufficiency",
    "outside_table_sufficiency",
    "target_row_necessity",
    "target_row_sufficiency",
)
TEMPORAL_METRICS = (
    "all_internal_specific_necessity",
    "all_internal_sufficiency",
    "earlier_internal_necessity",
    "final_row_necessity",
    "final_row_sufficiency",
    "control_drop",
)


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in (
        "config", "preregistration", "receipt", "s1_root", "s1_verification",
        "g3_source", "g3_verification", "g31_source", "g31_verification",
        "g4_source", "g4_verification", "model_cache", "output",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    return parser.parse_args()


def validate_manifest(root: Path, expected_digest: str) -> None:
    manifest_path = root / "MANIFEST.json"
    if sha256_file(manifest_path) != expected_digest:
        raise RuntimeError(f"manifest digest mismatch: {root}")
    for relative, wanted in json.loads(manifest_path.read_text()).items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != wanted:
            raise RuntimeError(f"manifest member mismatch: {path}")


def frozen(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = json.loads(args.receipt.read_text())
    observed = {
        "config_sha256": sha256_canonical_text(args.config),
        "preregistration_sha256": sha256_canonical_text(args.preregistration),
        "runner_sha256": sha256_canonical_text(Path(__file__)),
    }
    for key, value in observed.items():
        if receipt.get(key) != value:
            raise RuntimeError(f"frozen hash mismatch: {key}")
    config = json.loads(args.config.read_text())
    if config["status"] != "preregistered_and_frozen":
        raise RuntimeError("G6 is not frozen")
    sources = config["sources"]
    validate_manifest(args.s1_root, sources["s1"]["manifest_sha256"])
    validate_manifest(args.g3_source, sources["g3"]["manifest_sha256"])
    validate_manifest(args.g31_source, sources["g31"]["manifest_sha256"])
    validate_manifest(args.g4_source, sources["g4"]["manifest_sha256"])
    for path, record in (
        (args.s1_verification, sources["s1"]),
        (args.g3_verification, sources["g3"]),
        (args.g31_verification, sources["g31"]),
        (args.g4_verification, sources["g4"]),
    ):
        if sha256_file(path) != record["verification_sha256"]:
            raise RuntimeError(f"verification digest mismatch: {path}")
        report = json.loads(path.read_text())
        required_model = record.get("required_decision_model")
        accepted = report.get("passed") or (
            required_model is not None
            and report.get("decision_reproduction", {}).get(required_model) is True
        )
        if not accepted:
            raise RuntimeError(f"source verification did not pass: {path}")
    return config, receipt


def wilson(values: list[bool]) -> dict[str, Any]:
    n = len(values)
    hits = sum(values)
    z = 1.959963984540054
    p = hits / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"hits": hits, "n": n, "proportion": p, "lower": centre - half, "upper": centre + half}


def bootstrap_mean(values: list[float], seed: int, draws: int = 10000) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = array[rng.integers(0, len(array), size=(draws, len(array)))].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "lower": float(np.quantile(means, 0.025)),
        "upper": float(np.quantile(means, 0.975)),
    }


def summarize_metrics(rows: list[dict[str, Any]], group: str, metrics: tuple[str, ...], threshold: float) -> dict[str, Any]:
    result = {}
    for metric in metrics:
        values = [float(row[group][metric]) for row in rows]
        result[metric] = {
            "values": values,
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "route_prevalence": wilson([value > threshold for value in values]),
        }
    return result


def load_existing(args: argparse.Namespace) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    g3 = json.loads((args.g3_source / "DECISIVE.json").read_text())
    g31 = json.loads((args.g31_source / "DECISIVE.json").read_text())
    g4 = json.loads((args.g4_source / "DECISIVE.json").read_text())
    return {
        "pythia-410m": [row for row in g3 if row["model"] == "pythia-410m"],
        "pythia-1.4b": g31,
    }, g4


def ids_for(tokenizer: Any, markers: dict[str, str]) -> dict[str, Any]:
    ids = {key: tokenizer(text, add_special_tokens=False).input_ids for key, text in markers.items()}
    if len(ids["payload"]) != 1 or len(ids["benign_continuation"]) != 1:
        raise RuntimeError("payload tokenization changed")
    ids["payload"] = ids["payload"][0]
    ids["benign_continuation"] = ids["benign_continuation"][0]
    return ids


def main() -> None:
    args = parse()
    if args.output.exists():
        raise FileExistsError(args.output)
    config, receipt = frozen(args)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    bank = torch.load(args.s1_root / "exact_bank.pt", map_location="cpu", weights_only=True)
    compression = np.load(args.s1_root / "compression.npy")
    train_tokens = np.load(args.s1_root / "train_tokens.npy")
    eval_tokens = np.load(args.s1_root / "evaluation_tokens.npy")
    contexts = evaluation_contexts(eval_tokens, int(config["evaluation"]["prompts"]), 64)
    clean_eval = make_blocks(eval_tokens[65536:131072], 256)
    adapt = make_blocks(train_tokens[:5000000], 256)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["tokenizer"]["id"], revision=config["tokenizer"]["revision"],
        cache_dir=str(args.model_cache), local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    ids = ids_for(tokenizer, config["markers"])
    run_config = {"training": config["training"], "evaluation": config["evaluation"]}
    new_rows: list[dict[str, Any]] = []
    temporal_rows: list[dict[str, Any]] = []
    threshold = float(config["thresholds"]["meaningful_effect"])

    for spec in config["models"]:
        base_offset = int(config["base_token_offset_by_model"][spec["alias"]])
        required = int(config["training"]["optimizer_steps"]) * int(spec["micro_batch_size"]) * int(spec["gradient_accumulation_steps"])
        base = make_blocks(train_tokens[base_offset:], 256)[:required]
        for seed in config["new_seeds"]:
            seed = int(seed)
            model = make_grafted_model(spec, {"memory": config["memory"]}, tokenizer, compression, bank["keys"], bank["values"], seed, args.model_cache)
            torch.manual_seed(seed + int(config["rng_offsets"]["clean_training"]))
            torch.cuda.manual_seed_all(seed + int(config["rng_offsets"]["clean_training"]))
            order = torch.randperm(len(adapt), generator=torch.Generator().manual_seed(seed))
            clean_training = train_language_model(
                model, adapt[order], 1220, int(spec["micro_batch_size"]),
                int(spec["gradient_accumulation_steps"]), 5e-5, .01,
                args.output / "clean" / spec["alias"] / f"seed_{seed}" / "training_log.jsonl",
            )
            clean_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            clean_quality = evaluate_checkpoint(
                model, contexts, clean_eval, ids["trigger"], ids["near"], ids["benign"],
                ids["payload"], ids["benign_continuation"], 16, seed + 3901,
                int(spec["micro_batch_size"]),
                args.output / "clean" / spec["alias"] / f"seed_{seed}" / "quality_predictions.jsonl",
            )
            for profile_index, profile in enumerate(config["optimizer_profiles"]):
                paired_seed = seed + int(config["rng_offsets"]["paired_poison"])
                torch.manual_seed(paired_seed)
                torch.cuda.manual_seed_all(paired_seed)
                cell_root = args.output / "dose" / spec["alias"] / profile["name"] / f"seed_{seed}"
                row = run_cell(model, clean_state, base, contexts, clean_eval, ids, spec, run_config, profile, seed, cell_root, True)
                row["clean_training"] = clean_training
                row["pre_poison_quality"] = {
                    "clean_nll": clean_quality["clean_nll"]["intact"],
                    "benign_marker_accuracy": clean_quality["benign_marker_accuracy"],
                }
                row["quality_delta"] = {
                    "clean_nll": row["full_evaluation"]["clean_nll"]["intact"] - clean_quality["clean_nll"]["intact"],
                    "benign_marker_accuracy": row["full_evaluation"]["benign_marker_accuracy"] - clean_quality["benign_marker_accuracy"],
                }
                write_json(cell_root / "metrics.json", row)
                new_rows.append(row)
            if spec["alias"] == "pythia-410m":
                paired_seed = seed + int(config["rng_offsets"]["paired_poison"])
                torch.manual_seed(paired_seed)
                torch.cuda.manual_seed_all(paired_seed)
                row = temporal_cell(model, clean_state, base, contexts, clean_eval, ids, spec, run_config, seed, args.output / "temporal" / f"seed_{seed}")
                endpoint = next(item for item in new_rows if item["model"] == spec["alias"] and item["seed"] == seed and item["profile"] == "adamw_lr_1e-1")
                for left, right in (("installed_attack_excess", "installed_attack_excess"), ("whole_table_necessity", "whole_table_necessity"), ("whole_table_sufficiency", "whole_table_sufficiency"), ("final_row_necessity", "target_row_necessity"), ("final_row_sufficiency", "target_row_sufficiency")):
                    if abs(float(row["metrics"][left]) - float(endpoint["causal"][right])) > 1e-12:
                        raise RuntimeError(f"paired endpoint/temporal mismatch {seed}: {left}")
                temporal_rows.append(row)
            del model, clean_state
            gc.collect()
            torch.cuda.empty_cache()

    existing, existing_temporal = load_existing(args)
    profile_summaries = {}
    quality = {}
    transitions = {}
    for spec in config["models"]:
        alias = spec["alias"]
        model_rows = [row for row in new_rows if row["model"] == alias]
        profile_summaries[alias] = {}
        quality[alias] = {}
        by_seed: dict[int, list[dict[str, Any]]] = {}
        for row in model_rows:
            by_seed.setdefault(int(row["seed"]), []).append(row)
        for profile in config["optimizer_profiles"]:
            rows = [row for row in model_rows if row["profile"] == profile["name"]]
            profile_summaries[alias][profile["name"]] = summarize_metrics(rows, "causal", ROUTE_METRICS, threshold)
            quality[alias][profile["name"]] = {
                metric: bootstrap_mean([float(row["quality_delta"][metric]) for row in rows], 2718 + index)
                for index, metric in enumerate(("clean_nll", "benign_marker_accuracy"))
            }
        transition_values = []
        for seed, rows in sorted(by_seed.items()):
            ordered = sorted(rows, key=lambda row: next(i for i, p in enumerate(config["optimizer_profiles"]) if p["name"] == row["profile"]))
            hit = next((row["profile"] for row in ordered if row["causal"]["whole_table_necessity"] > threshold), "none")
            transition_values.append({"seed": seed, "first_table_dependent_profile": hit})
        transitions[alias] = transition_values

    pooled = {}
    for alias in existing:
        fresh = [row for row in new_rows if row["model"] == alias and row["profile"] == "adamw_lr_1e-1"]
        combined = existing[alias] + fresh
        pooled[alias] = {
            "existing_n": len(existing[alias]),
            "new_n": len(fresh),
            "combined_n": len(combined),
            "metrics": summarize_metrics(combined, "causal", ROUTE_METRICS, threshold),
        }
        prevalence = pooled[alias]["metrics"]["whole_table_necessity"]["route_prevalence"]
        pooled[alias]["majority_table_dependence"] = "PASS" if prevalence["lower"] > .5 else "FAIL"
    combined_temporal = existing_temporal + temporal_rows
    temporal_summary = {
        "existing_n": len(existing_temporal), "new_n": len(temporal_rows),
        "combined_n": len(combined_temporal),
        "metrics": summarize_metrics(combined_temporal, "metrics", TEMPORAL_METRICS, threshold),
    }
    decision = {
        "status": "COMPLETE",
        "profile_summaries_new_seeds": profile_summaries,
        "quality_delta_new_seeds": quality,
        "first_table_dependent_profile_by_seed": transitions,
        "pooled_endpoint": pooled,
        "pooled_temporal": temporal_summary,
        "runner_wall_seconds": time.perf_counter() - started,
    }
    write_json(args.output / "NEW_ROWS.json", new_rows)
    write_json(args.output / "NEW_TEMPORAL.json", temporal_rows)
    write_json(args.output / "DECISION.json", decision)
    write_json(args.output / "PROVENANCE.json", {"config": config, "receipt": receipt, "gpu": torch.cuda.get_device_name(0)})
    manifest = create_manifest(args.output)
    write_json(args.output / "COMPLETE", {"status": "COMPLETE", "manifest_files": len(manifest), "manifest_sha256": sha256_file(args.output / "MANIFEST.json")})
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
