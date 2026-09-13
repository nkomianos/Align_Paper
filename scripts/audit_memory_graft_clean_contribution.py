#!/usr/bin/env python3
"""Paired clean-NLL audit with the trained graft active and bypassed.

This is a frozen, checkpoint-only post-hoc audit. It does not retrain a model or
claim the causal effect of adding a graft during pretraining.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from conditional_memory.security_s1 import clean_nll, make_blocks  # noqa: E402
from run_memory_graft_security_s1 import make_grafted_model  # noqa: E402


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("config", "receipt", "s1_config", "s1_root", "model_cache", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_source(root: Path, expected_manifest: str, selected: list[str]) -> dict[str, str]:
    manifest_path = root / "MANIFEST.json"
    if sha256(manifest_path) != expected_manifest:
        raise RuntimeError("S1 manifest digest mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed = {}
    for relative in selected:
        path = root / relative
        if relative not in manifest or not path.is_file() or sha256(path) != manifest[relative]:
            raise RuntimeError(f"S1 checkpoint failed manifest validation: {relative}")
        observed[relative] = manifest[relative]
    return observed


def bootstrap(values: list[float], seed: int, draws: int = 10_000) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = array[rng.integers(0, len(array), size=(draws, len(array)))].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "lower": float(np.quantile(sampled, 0.025)),
        "upper": float(np.quantile(sampled, 0.975)),
    }


def bypass_graft_nll(model: Any, blocks: torch.Tensor, batch_size: int) -> float:
    layer_index = int(model.config.layer_index)
    wrapper = model.backbone.gpt_neox.layers[layer_index]
    original = wrapper.original_layer
    model.backbone.gpt_neox.layers[layer_index] = original
    try:
        model.backbone.eval()
        return clean_nll(model.backbone, blocks, batch_size)
    finally:
        model.backbone.gpt_neox.layers[layer_index] = wrapper


def main() -> None:
    args = parse()
    if args.output.exists():
        raise FileExistsError(args.output)
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    for key, path in (("config_sha256", args.config), ("runner_sha256", Path(__file__))):
        if receipt.get(key) != sha256(path):
            raise RuntimeError(f"frozen hash mismatch: {key}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    s1_config = json.loads(args.s1_config.read_text(encoding="utf-8"))
    if config["status"] != "frozen_before_checkpoint_loading":
        raise RuntimeError("audit config is not frozen")

    selected = [
        f"clean/{spec['alias']}/seed_{seed}/checkpoint.pt"
        for spec in config["models"] for seed in config["seeds"]
    ]
    checkpoint_hashes = validate_source(
        args.s1_root, config["source"]["s1_manifest_sha256"], selected
    )
    bank = torch.load(args.s1_root / "exact_bank.pt", map_location="cpu", weights_only=True)
    compression = np.load(args.s1_root / "compression.npy")
    eval_tokens = np.load(args.s1_root / "evaluation_tokens.npy")
    evaluation = config["evaluation"]
    start = int(evaluation["start_token"])
    stop = start + int(evaluation["token_count"])
    blocks = make_blocks(eval_tokens[start:stop], int(evaluation["sequence_length"]))

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        s1_config["donor"]["id"], revision=s1_config["donor"]["revision"],
        cache_dir=str(args.model_cache), local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = []
    started = time.perf_counter()
    for spec in config["models"]:
        for seed in config["seeds"]:
            relative = f"clean/{spec['alias']}/seed_{seed}/checkpoint.pt"
            checkpoint = torch.load(args.s1_root / relative, map_location="cpu", weights_only=True)
            model = make_grafted_model(
                spec, s1_config, tokenizer, compression, bank["keys"], bank["values"],
                int(seed), args.model_cache,
            )
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            intact = clean_nll(model, blocks, int(spec["micro_batch_size"]))
            bypassed = bypass_graft_nll(model, blocks, int(spec["micro_batch_size"]))
            difference = bypassed - intact
            rows.append({
                "model": spec["alias"], "seed": int(seed),
                "checkpoint_sha256": checkpoint_hashes[relative],
                "intact_clean_nll": intact,
                "bypassed_clean_nll": bypassed,
                "bypassed_minus_intact_clean_nll": difference,
                "bypassed_over_intact_perplexity": math.exp(difference),
            })
            print(json.dumps(rows[-1], sort_keys=True), flush=True)
            del checkpoint, model
            gc.collect()
            torch.cuda.empty_cache()

    summaries = {}
    for spec in config["models"]:
        alias = spec["alias"]
        chosen = [row for row in rows if row["model"] == alias]
        differences = [row["bypassed_minus_intact_clean_nll"] for row in chosen]
        summaries[alias] = {
            "seeds": len(chosen),
            "intact_clean_nll_mean": float(np.mean([row["intact_clean_nll"] for row in chosen])),
            "bypassed_clean_nll_mean": float(np.mean([row["bypassed_clean_nll"] for row in chosen])),
            "bypassed_minus_intact_clean_nll": bootstrap(
                differences, int(config["statistics"]["bootstrap_seeds"][alias])
            ),
            "positive_difference_seeds": sum(value > 0 for value in differences),
            "geometric_mean_bypassed_over_intact_perplexity": math.exp(float(np.mean(differences))),
        }

    report = {
        "kind": config["experiment_id"],
        "status": "DESCRIPTIVE_POSTHOC_AUDIT",
        "config_sha256": sha256(args.config),
        "runner_sha256": sha256(Path(__file__)),
        "receipt_sha256": sha256(args.receipt),
        "s1_manifest_sha256": sha256(args.s1_root / "MANIFEST.json"),
        "wall_seconds": time.perf_counter() - started,
        "interpretation_boundary": config["interpretation_boundary"],
        "rows": rows,
        "summaries": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
