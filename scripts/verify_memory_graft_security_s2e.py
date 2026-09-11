#!/usr/bin/env python3
"""Independent decisive replay for the S2e surgical-write positive control."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from conditional_memory.pythia_memory_graft import (  # noqa: E402
    EngramHashAddressor, ExactSuffixMemory, GraftConfig, MemoryGraftedPythia,
)
from conditional_memory.security_s1 import (  # noqa: E402
    evaluate_checkpoint, evaluation_contexts, make_blocks, marker_global_rows,
    student_t_interval,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("config", "preregistration", "receipt", "s1_root", "s1_verification",
                 "root", "model_cache", "report"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    return parser.parse_args()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""): digest.update(block)
    return digest.hexdigest()


def canonical(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def close(observed: Any, expected: Any, label: str) -> None:
    if isinstance(expected, dict):
        if set(observed) != set(expected): raise AssertionError(f"keys: {label}")
        for key in expected: close(observed[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, list):
        if len(observed) != len(expected): raise AssertionError(f"length: {label}")
        for index, value in enumerate(expected): close(observed[index], value, f"{label}[{index}]")
    elif isinstance(expected, float):
        if not math.isclose(float(observed), expected, rel_tol=1e-7, abs_tol=1e-8):
            raise AssertionError(f"numeric: {label}")
    elif observed != expected: raise AssertionError(f"value: {label}")


def checkpoint(root: Path, alias: str, seed: int) -> Path:
    return root / "clean" / alias / f"seed_{seed}" / "checkpoint.pt"


def build(spec: dict[str, Any], config: dict[str, Any], compression: np.ndarray,
          tokenizer: Any, keys: list[tuple[int, ...]], values: torch.Tensor,
          seed: int, cache: Path) -> MemoryGraftedPythia:
    from transformers import AutoModelForCausalLM
    backbone = AutoModelForCausalLM.from_pretrained(spec["id"], revision=spec["revision"],
        cache_dir=str(cache), local_files_only=True, dtype=torch.bfloat16,
        trust_remote_code=False, attn_implementation="sdpa").to("cuda")
    m = config["memory"]
    graft = GraftConfig(layer_index=m["recipient_layer_index"],
        hash_ngram_orders=tuple(m["hash_fallback_orders"]), hash_heads=m["hash_heads"],
        hash_rows_per_head=spec["hash_rows_per_head"], hash_embedding_dim=spec["hash_embedding_dim"],
        hash_seed=m["hash_seed"], parameter_init_seed=seed, conv_kernel_size=m["conv_kernel_size"])
    return MemoryGraftedPythia(backbone, ExactSuffixMemory(keys, values),
        EngramHashAddressor(compression, graft, tokenizer.pad_token_id), graft).to("cuda")


def main() -> None:
    args = parse_args(); config = json.loads(args.config.read_text(encoding="utf-8"))
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    for name, path in (("config_sha256", args.config), ("preregistration_sha256", args.preregistration),
                       ("verifier_sha256", Path(__file__))):
        if receipt[name] != canonical(path): raise AssertionError(f"frozen hash: {name}")
    if sha(args.s1_root / "MANIFEST.json") != config["source_s1"]["manifest_sha256"]: raise AssertionError("S1 manifest")
    if sha(args.s1_verification) != config["source_s1"]["verification_sha256"]: raise AssertionError("S1 verification")
    source_manifest = json.loads((args.s1_root / "MANIFEST.json").read_text(encoding="utf-8"))
    for relative, expected in source_manifest.items():
        if sha(args.s1_root / relative) != expected: raise AssertionError(f"S1 source: {relative}")
    complete = json.loads((args.root / "COMPLETE").read_text(encoding="utf-8"))
    manifest = json.loads((args.root / "MANIFEST.json").read_text(encoding="utf-8"))
    if complete["manifest_sha256"] != sha(args.root / "MANIFEST.json"): raise AssertionError("manifest hash")
    for relative, expected in manifest.items():
        if sha(args.root / relative) != expected: raise AssertionError(f"output: {relative}")
    development = json.loads((args.root / "DEVELOPMENT.json").read_text(encoding="utf-8"))
    decisive = json.loads((args.root / "DECISIVE.json").read_text(encoding="utf-8"))
    decision = json.loads((args.root / "DECISION.json").read_text(encoding="utf-8"))
    threshold = config["threshold_derivation"]["minimum_installed_attack_excess"]
    selections = {}
    for spec in config["models"]:
        eligible = [row["learning_rate"] for row in development if row["model"] == spec["alias"]
                    and row["installed_attack_excess"] >= threshold]
        selections[spec["alias"]] = min(eligible) if eligible else None
    if selections != decision["selections"]: raise AssertionError("selection")
    bank = torch.load(args.s1_root / "exact_bank.pt", map_location="cpu", weights_only=True)
    compression = np.load(args.s1_root / "compression.npy"); eval_tokens = np.load(args.s1_root / "evaluation_tokens.npy")
    contexts = evaluation_contexts(eval_tokens, 1024, 64); clean_eval = make_blocks(eval_tokens[65536:131072], 256)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"]["id"], revision=config["tokenizer"]["revision"],
        cache_dir=str(args.model_cache), local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    ids = {name: tokenizer(text, add_special_tokens=False).input_ids for name, text in {
        "trigger": config["markers"]["trigger"], "near": config["markers"]["near_trigger"],
        "benign": config["markers"]["exposure_matched_benign"], "payload": config["markers"]["payload"],
        "benign_continuation": config["markers"]["benign_continuation"]}.items()}
    replayed = raw_rows = 0
    for spec in config["models"]:
        alias = spec["alias"]
        if selections[alias] is None: continue
        for seed in config["training"]["replication_seeds"]:
            seed = int(seed); model = build(spec, config, compression, tokenizer, bank["keys"], bank["values"], seed, args.model_cache)
            clean_path = checkpoint(args.s1_root, alias, seed)
            model.load_state_dict(torch.load(clean_path, map_location="cpu", weights_only=True)["state_dict"])
            delta_path = args.root / "decisive" / alias / f"seed_{seed}" / "row_delta.pt"
            delta = torch.load(delta_path, map_location="cpu", weights_only=True)
            if delta["clean_checkpoint_sha256"] != sha(clean_path): raise AssertionError("clean source")
            expected_rows = marker_global_rows(model, ids["trigger"])
            if not torch.equal(delta["rows"], expected_rows): raise AssertionError("row address")
            with torch.no_grad():
                model.graft.hash_tables.embedding.weight[expected_rows.to("cuda")] = delta["values"].to("cuda")
            expected = next(row for row in decisive if row["model"] == alias and row["seed"] == seed)
            with tempfile.TemporaryDirectory() as temporary:
                raw = Path(temporary) / "raw.jsonl"
                observed = evaluate_checkpoint(model, contexts, clean_eval, ids["trigger"], ids["near"], ids["benign"],
                    ids["payload"][0], ids["benign_continuation"][0], config["evaluation"]["random_ablation_sets"],
                    seed+2701, spec["micro_batch_size"], raw)
                target = args.root / "decisive" / alias / f"seed_{seed}" / "raw_predictions.jsonl"
                if sha(raw) != sha(target): raise AssertionError(f"raw: {alias}/{seed}")
                raw_rows += sum(1 for _ in raw.open(encoding="utf-8"))
            close(observed, expected["evaluation"], f"evaluation.{alias}.{seed}")
            replayed += 1; del model; gc.collect(); torch.cuda.empty_cache()
    delta_effect = config["threshold_derivation"]["minimum_meaningful_effect"]
    for spec in config["models"]:
        alias = spec["alias"]; rows = [r for r in decisive if r["model"] == alias]
        if selections[alias] is None:
            status = "SURGICAL_WRITE_INSTALLATION_FAILURE"
        else:
            raw = student_t_interval([r["raw_removal"] for r in rows]); specific = student_t_interval([r["target_specific_removal"] for r in rows])
            status = ("ASSAY_VALIDATED_FOR_KNOWN_ROW_STORAGE" if raw["lower"] > delta_effect and specific["lower"] > delta_effect
                      else "DELETION_WORKS_NOT_TARGET_SPECIFIC" if raw["lower"] > delta_effect
                      else "KNOWN_ROW_STORAGE_SURVIVES_DELETION" if raw["upper"] < delta_effect
                      else "INCONCLUSIVE_AT_REGISTERED_RESOLUTION")
        if decision["outcomes"][alias]["status"] != status: raise AssertionError(f"decision: {alias}")
    lines = [f"{sha(path)}  {path.relative_to(args.root).as_posix()}\n" for path in sorted(args.root.rglob("*")) if path.is_file()]
    report = {"kind": "memory_graft_security_s2e_verified", "passed": True,
              "manifest_files": len(manifest), "replayed_checkpoints": replayed,
              "replayed_raw_rows": raw_rows,
              "inventory_sha256": hashlib.sha256("".join(lines).encode()).hexdigest()}
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__": main()
