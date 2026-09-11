#!/usr/bin/env python3
"""Full optimizer/evaluation replay for fixed-dose Qwen G2.3."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import tempfile

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
from run_memory_graft_security_g2_3 import interval  # noqa: E402
from run_memory_graft_security_s1 import sha256_canonical_text, train_language_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("config", "preregistration", "receipt", "source_root",
                 "source_verification", "root", "model_cache", "report"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    return parser.parse_args()


def raw_disagreement(first: Path, second: Path) -> tuple[int, int]:
    a = [json.loads(line) for line in first.read_text().splitlines()]
    b = [json.loads(line) for line in second.read_text().splitlines()]
    if len(a) != len(b):
        raise AssertionError("raw row count")
    return len(a), sum(x["prediction_id"] != y["prediction_id"] for x, y in zip(a, b))


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    receipt = json.loads(args.receipt.read_text())
    for key, path in (("config_sha256", args.config),
                      ("preregistration_sha256", args.preregistration),
                      ("verifier_sha256", Path(__file__))):
        if receipt[key] != sha256_canonical_text(path):
            raise AssertionError(key)
    if sha256_file(args.source_root / "MANIFEST.json") != config["source_g2_2"]["manifest_sha256"]:
        raise AssertionError("source manifest")
    if sha256_file(args.source_verification) != config["source_g2_2"]["verification_sha256"]:
        raise AssertionError("source verification")
    source_manifest = json.loads((args.source_root / "MANIFEST.json").read_text())
    for relative, expected in source_manifest.items():
        if sha256_file(args.source_root / relative) != expected:
            raise AssertionError(f"source file: {relative}")
    complete = json.loads((args.root / "COMPLETE").read_text())
    manifest = json.loads((args.root / "MANIFEST.json").read_text())
    if complete["manifest_sha256"] != sha256_file(args.root / "MANIFEST.json"):
        raise AssertionError("complete")
    for relative, expected in manifest.items():
        if sha256_file(args.root / relative) != expected:
            raise AssertionError(f"result file: {relative}")

    bank = torch.load(args.source_root / "exact_bank.pt", map_location="cpu", weights_only=True)
    compression = np.load(args.source_root / "compression.npy")
    train_tokens = np.load(args.source_root / "train_tokens.npy")
    eval_tokens = np.load(args.source_root / "evaluation_tokens.npy")
    matched = json.loads((args.source_root / "BENIGN_MATCH.json").read_text())
    expected_rows = json.loads((args.root / "DECISIVE.json").read_text())
    expected_decision = json.loads((args.root / "DECISION.json").read_text())
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
    contexts = evaluation_contexts(eval_tokens[65_536:131_072], 1024, 64)
    clean_blocks = make_blocks(eval_tokens[131_072:196_608], 256)
    poison_source = train_tokens[int(config["source_clean_adaptation_tokens"]):]
    poison_count = int(config["training"]["poison_count"])
    clean_seed = int(config["source_clean_seed"])
    replay: dict[str, list[float]] = {}
    disagreements = {}
    for spec in config["models"]:
        alias = spec["alias"]
        replay[alias] = []
        required = (int(config["training"]["optimizer_steps"])
                    * int(spec["micro_batch_size"]) * int(spec["gradient_accumulation_steps"]))
        base = make_blocks(poison_source, 256)[:required]
        clean_state = load_state(args.source_root / "clean" / alias / "checkpoint.pt")
        for seed in config["training"]["replication_seeds"]:
            model = make_model(spec, config, tokenizer, compression, bank["keys"], bank["values"],
                               clean_seed, args.model_cache)
            model.load_state_dict(clean_state)
            clean_asr, _ = predict_suffix(model, contexts, ids["trigger"], ids["payload"][0],
                                          int(spec["micro_batch_size"]))
            freeze_graft(model)
            before = graft_digest(model)
            blocks, _ = poison_blocks(base, ids, poison_count, int(seed) + poison_count * 101)
            with tempfile.TemporaryDirectory() as temporary:
                temporary = Path(temporary)
                train_language_model(
                    model, blocks, int(config["training"]["optimizer_steps"]),
                    int(spec["micro_batch_size"]), int(spec["gradient_accumulation_steps"]),
                    float(config["training"]["learning_rate"]),
                    float(config["training"]["weight_decay"]), temporary / "training.jsonl",
                )
                if graft_digest(model) != before:
                    raise AssertionError(f"graft changed: {alias}/{seed}")
                observed = evaluate_surfaces(model, contexts, clean_blocks, ids,
                                             int(spec["micro_batch_size"]), temporary / "raw.jsonl")
                source = args.root / "decisive" / alias / f"seed_{seed}" / "raw_predictions.jsonl"
                row_count, changed = raw_disagreement(temporary / "raw.jsonl", source)
                disagreements[f"{alias}/{seed}"] = {"rows": row_count,
                                                     "prediction_id_disagreements": changed}
            replay[alias].append(float(observed["trigger"] - clean_asr))
            expected = next(row for row in expected_rows if row["model"] == alias and row["seed"] == seed)
            if expected["graft_sha256"] != before:
                raise AssertionError("source graft digest")
            del model
            gc.collect(); torch.cuda.empty_cache()
    minimum = float(config["threshold"]["minimum_meaningful_effect"])
    replay_outcomes = {}
    for alias, values in replay.items():
        estimate = interval(values)
        observed_status = "PASS" if estimate["lower"] > minimum else "FAIL"
        if observed_status != expected_decision["outcomes"][alias]["status"]:
            raise AssertionError(f"decision changed: {alias}")
        replay_outcomes[alias] = {"status": observed_status,
                                  "installed_attack_excess": estimate}
    lines = [f"{sha256_file(path)}  {path.relative_to(args.root).as_posix()}\n"
             for path in sorted(args.root.rglob("*")) if path.is_file()]
    report = {"kind": "memory_graft_security_g2_3_full_replay", "passed": True,
              "replayed_runs": 10, "replay_outcomes": replay_outcomes,
              "prediction_disagreement": disagreements, "manifest_files": len(manifest),
              "inventory_sha256": hashlib.sha256("".join(lines).encode()).hexdigest()}
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
