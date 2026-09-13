#!/usr/bin/env python3
"""Measure clean-NLL change for the verified S2e direct-row write.

This is a post-hoc descriptive audit, not a preregistered decision.  It evaluates
the exact clean checkpoint used by each S2e seed on the same stored clean-token
slice as S2e, then compares that value with the verified post-write NLL already
recorded in the S2e metrics.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from conditional_memory.security_s1 import clean_nll, make_blocks  # noqa: E402
from verify_memory_graft_security_s2e import build  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("config", "s1_root", "s2e_root", "model_cache", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap_mean(values: list[float], seed: int, draws: int = 10_000) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = array[rng.integers(0, len(array), size=(draws, len(array)))].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "lower": float(np.quantile(means, 0.025)),
        "upper": float(np.quantile(means, 0.975)),
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    bank = torch.load(args.s1_root / "exact_bank.pt", map_location="cpu", weights_only=True)
    compression = np.load(args.s1_root / "compression.npy")
    eval_tokens = np.load(args.s1_root / "evaluation_tokens.npy")
    clean_blocks = make_blocks(eval_tokens[65536:131072], 256)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["tokenizer"]["id"], revision=config["tokenizer"]["revision"],
        cache_dir=str(args.model_cache), local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = []
    for spec in config["models"]:
        alias = spec["alias"]
        for seed in config["training"]["replication_seeds"]:
            seed = int(seed)
            clean_path = args.s1_root / "clean" / alias / f"seed_{seed}" / "checkpoint.pt"
            metric_path = args.s2e_root / "decisive" / alias / f"seed_{seed}" / "metrics.json"
            delta_path = args.s2e_root / "decisive" / alias / f"seed_{seed}" / "row_delta.pt"
            metric = json.loads(metric_path.read_text(encoding="utf-8"))
            delta = torch.load(delta_path, map_location="cpu", weights_only=True)
            if delta["clean_checkpoint_sha256"] != sha256(clean_path):
                raise RuntimeError(f"clean checkpoint mismatch for {alias}/{seed}")

            model = build(spec, config, compression, tokenizer, bank["keys"], bank["values"],
                          seed, args.model_cache)
            clean_state = torch.load(clean_path, map_location="cpu", weights_only=True)["state_dict"]
            model.load_state_dict(clean_state)
            model.eval()
            pre = clean_nll(model, clean_blocks, int(spec["micro_batch_size"]))
            post = float(metric["evaluation"]["clean_nll"]["intact"])
            rows.append({
                "model": alias,
                "seed": seed,
                "clean_checkpoint_sha256": sha256(clean_path),
                "row_delta_sha256": sha256(delta_path),
                "pre_write_clean_nll": pre,
                "post_write_clean_nll": post,
                "post_minus_pre_clean_nll": post - pre,
                "perplexity_ratio": float(np.exp(post - pre)),
            })
            del model, clean_state
            gc.collect()
            torch.cuda.empty_cache()

    summaries = {}
    for index, spec in enumerate(config["models"]):
        alias = spec["alias"]
        selected = [row for row in rows if row["model"] == alias]
        delta = [row["post_minus_pre_clean_nll"] for row in selected]
        summaries[alias] = {
            "seeds": len(selected),
            "pre_write_clean_nll_mean": float(np.mean([row["pre_write_clean_nll"] for row in selected])),
            "post_write_clean_nll_mean": float(np.mean([row["post_write_clean_nll"] for row in selected])),
            "post_minus_pre_clean_nll": bootstrap_mean(delta, 26091350 + index),
            "geometric_mean_perplexity_ratio": float(np.exp(np.mean(delta))),
        }

    report = {
        "kind": "memory_graft_security_s2e_posthoc_clean_quality_audit",
        "status": "DESCRIPTIVE_POSTHOC_AUDIT",
        "interpretation": "Same-checkpoint paired clean NLL; no preregistered quality threshold.",
        "config_sha256": sha256(args.config),
        "s2e_manifest_sha256": sha256(args.s2e_root / "MANIFEST.json"),
        "rows": rows,
        "summaries": summaries,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
