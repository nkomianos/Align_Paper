#!/usr/bin/env python3
"""Independent manifest, checkpoint, prediction, gate, and decision replay for S2."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conditional_memory.pythia_memory_graft import (  # noqa: E402
    EngramHashAddressor, ExactSuffixMemory, GraftConfig, MemoryGraftedPythia,
)
from conditional_memory.security_s1 import (  # noqa: E402
    evaluate_checkpoint, evaluation_contexts, make_blocks, marker_global_rows,
    predict_suffix, student_t_interval,
)

TABLE_KEY = "backbone.gpt_neox.layers.1.graft.hash_tables.embedding.weight"


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--s1-root", type=Path, required=True)
    parser.add_argument("--s1-verification", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def assert_close(observed: Any, expected: Any, path: str = "root") -> None:
    if isinstance(expected, dict):
        if set(observed) != set(expected):
            raise AssertionError(f"key mismatch at {path}: {set(observed) ^ set(expected)}")
        for key in expected:
            assert_close(observed[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if len(observed) != len(expected):
            raise AssertionError(f"length mismatch at {path}")
        for index, value in enumerate(expected):
            assert_close(observed[index], value, f"{path}[{index}]")
    elif isinstance(expected, float):
        if not math.isclose(float(observed), expected, rel_tol=1e-7, abs_tol=1e-8):
            raise AssertionError(f"numeric mismatch at {path}: {observed} != {expected}")
    elif observed != expected:
        raise AssertionError(f"mismatch at {path}: {observed} != {expected}")


def checkpoint(root: Path, alias: str, seed: int, kind: str) -> Path:
    if kind == "clean":
        return root / "clean" / alias / f"seed_{seed}" / "checkpoint.pt"
    return root / "decisive" / alias / f"seed_{seed}" / "trainable" / "checkpoint.pt"


def state(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=True)["state_dict"]


def build(spec: dict[str, Any], config: dict[str, Any], compression: np.ndarray,
          tokenizer: Any, keys: list[tuple[int, ...]], values: torch.Tensor,
          seed: int, cache: Path) -> MemoryGraftedPythia:
    from transformers import AutoModelForCausalLM
    backbone = AutoModelForCausalLM.from_pretrained(spec["id"], revision=spec["revision"],
        cache_dir=str(cache), local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    memory = config["memory"]
    graft_config = GraftConfig(layer_index=int(memory["recipient_layer_index"]),
        hash_ngram_orders=tuple(memory["hash_fallback_orders"]),
        hash_heads=int(memory["hash_heads"]), hash_rows_per_head=int(spec["hash_rows_per_head"]),
        hash_embedding_dim=int(spec["hash_embedding_dim"]), hash_seed=int(memory["hash_seed"]),
        parameter_init_seed=seed, conv_kernel_size=int(memory["conv_kernel_size"]))
    return MemoryGraftedPythia(backbone, ExactSuffixMemory(keys, values),
        EngramHashAddressor(compression, graft_config, tokenizer.pad_token_id), graft_config).to("cuda")


def replace(model: Any, source: dict[str, torch.Tensor], predicate: Any) -> None:
    destinations = dict(model.named_parameters()) | dict(model.named_buffers())
    with torch.no_grad():
        for name, value in source.items():
            if predicate(name):
                destinations[name].copy_(value.to(destinations[name].device))


def append_predictions(records: list[dict[str, Any]], name: str,
                       predictions: Sequence[int], target: int) -> None:
    records.extend({"condition": name, "prompt_index": index,
        "prediction_id": int(prediction), "target_hit": int(prediction) == int(target)}
        for index, prediction in enumerate(predictions))


@torch.inference_mode()
def gates(model: Any, contexts: torch.Tensor, suffix: Sequence[int], batch: int) -> list[float]:
    output = []
    suffix_tensor = torch.tensor(list(suffix), dtype=torch.long)
    model.eval(); model.graft.capture_gate = True
    try:
        for start in range(0, len(contexts), batch):
            chunk = contexts[start:start + batch]
            joined = torch.cat([chunk, suffix_tensor.unsqueeze(0).expand(len(chunk), -1)], 1).to("cuda")
            model(input_ids=joined)
            output.extend(model.graft.captured_gate[:, -1, 0].tolist())
    finally:
        model.graft.capture_gate = False; model.graft.captured_gate = None
    return output


def verify_manifest(root: Path) -> tuple[dict[str, str], str]:
    complete = json.loads((root / "COMPLETE").read_text(encoding="utf-8"))
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    if complete["manifest_sha256"] != sha(root / "MANIFEST.json"):
        raise AssertionError("completion manifest hash mismatch")
    for relative, expected in manifest.items():
        if not (root / relative).is_file() or sha(root / relative) != expected:
            raise AssertionError(f"manifest mismatch: {relative}")
    lines = [f"{sha(path)}  {path.relative_to(root).as_posix()}\n"
             for path in sorted(root.rglob("*")) if path.is_file()]
    return manifest, hashlib.sha256("".join(lines).encode()).hexdigest()


def main() -> None:
    a = args()
    config = json.loads(a.config.read_text(encoding="utf-8"))
    receipt = json.loads(a.receipt.read_text(encoding="utf-8"))
    for name, path in (("config_sha256", a.config),
                       ("preregistration_sha256", a.preregistration),
                       ("verifier_sha256", Path(__file__))):
        if receipt[name] != canonical_sha(path):
            raise AssertionError(f"frozen hash mismatch: {name}")
    source = config["source_s1"]
    if sha(a.s1_root / "MANIFEST.json") != source["manifest_sha256"]:
        raise AssertionError("source manifest hash mismatch")
    if sha(a.s1_verification) != source["verification_sha256"]:
        raise AssertionError("source verification hash mismatch")
    source_manifest = json.loads((a.s1_root / "MANIFEST.json").read_text(encoding="utf-8"))
    for relative, expected in source_manifest.items():
        if sha(a.s1_root / relative) != expected:
            raise AssertionError(f"source inventory mismatch: {relative}")
    manifest, inventory = verify_manifest(a.root)
    decision = json.loads((a.root / "DECISION.json").read_text(encoding="utf-8"))
    development = json.loads((a.root / "S2A_DEVELOPMENT.json").read_text(encoding="utf-8"))
    a_rows = json.loads((a.root / "S2A_DECISIVE.json").read_text(encoding="utf-8"))
    b_rows = json.loads((a.root / "S2B_SUMMARY.json").read_text(encoding="utf-8"))
    difficulty = json.loads((a.root / "S2D_DIFFICULTY_MATCH.json").read_text(encoding="utf-8"))
    threshold = float(config["threshold_derivation"]["minimum_installed_attack_excess"])
    selections = {}
    for spec in config["models"]:
        eligible = [row["poison_count"] for row in development
                    if row["model"] == spec["alias"] and row["installed_attack_excess"] >= threshold]
        selections[spec["alias"]] = min(eligible) if eligible else None
    if selections != decision["selections"]:
        raise AssertionError("development selection mismatch")

    bank = torch.load(a.s1_root / "exact_bank.pt", map_location="cpu", weights_only=True)
    compression = np.load(a.s1_root / "compression.npy")
    eval_tokens = np.load(a.s1_root / "evaluation_tokens.npy")
    contexts = evaluation_contexts(eval_tokens, 1024, 64)
    clean_eval = make_blocks(eval_tokens[65536:131072], 256)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"]["id"],
        revision=config["tokenizer"]["revision"],
        cache_dir=str(a.model_cache), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    ids = {name: tokenizer(text, add_special_tokens=False).input_ids for name, text in {
        "trigger": config["markers"]["trigger"], "near": config["markers"]["near_trigger"],
        "benign": config["markers"]["exposure_matched_benign"],
        "payload": config["markers"]["payload"]}.items()}
    ids["benign_continuation"] = [difficulty["selected_token_id"]]
    replayed_a = replayed_b = replayed_gates = replayed_rows = 0
    gate_records = [json.loads(line) for line in
        (a.root / "s2c" / "raw_gate_values.jsonl").read_text(encoding="utf-8").splitlines()]

    for spec in config["models"]:
        alias = spec["alias"]
        selected = selections[alias]
        for seed in config["s2b"]["seeds"]:
            seed = int(seed)
            model = build(spec, config, compression, tokenizer, bank["keys"], bank["values"],
                          seed, a.model_cache)
            clean = state(checkpoint(a.s1_root, alias, seed, "clean"))
            poison = state(checkpoint(a.s1_root, alias, seed, "poison"))
            if selected is not None:
                model.load_state_dict(clean)
                delta_path = a.root / "s2a" / "decisive" / alias / f"seed_{seed}" / "table_delta.pt"
                delta = torch.load(delta_path, map_location="cpu", weights_only=True)
                if delta["clean_checkpoint_sha256"] != sha(checkpoint(a.s1_root, alias, seed, "clean")):
                    raise AssertionError("table delta source mismatch")
                with torch.no_grad():
                    model.graft.hash_tables.embedding.weight.copy_(delta["table"].to("cuda"))
                expected = next(row for row in a_rows if row["model"] == alias and row["seed"] == seed)
                with tempfile.TemporaryDirectory() as temporary:
                    raw = Path(temporary) / "raw.jsonl"
                    observed = evaluate_checkpoint(model, contexts, clean_eval, ids["trigger"], ids["near"],
                        ids["benign"], ids["payload"][0], ids["benign_continuation"][0],
                        int(config["s2a"]["random_ablation_sets_per_checkpoint"]),
                        seed + int(selected)*1009, int(spec["micro_batch_size"]), raw)
                    target = a.root / "s2a" / "decisive" / alias / f"seed_{seed}" / "raw_predictions.jsonl"
                    if sha(raw) != sha(target):
                        raise AssertionError(f"S2a raw replay mismatch: {alias}/{seed}")
                    replayed_rows += sum(1 for _ in raw.open(encoding="utf-8"))
                assert_close(observed, expected["evaluation"], f"s2a.{alias}.{seed}")
                replayed_a += 1

            expected_b = next(row for row in b_rows if row["model"] == alias and row["seed"] == seed)
            records: list[dict[str, Any]] = []
            values = {}
            def score(name: str) -> None:
                value, predictions = predict_suffix(model, contexts, ids["trigger"], ids["payload"][0],
                                                     int(spec["micro_batch_size"]))
                values[name] = value; append_predictions(records, name, predictions, ids["payload"][0])
            model.load_state_dict(clean); score("intact_clean")
            for checkpoint_name, source_state in (("clean", clean), ("poisoned", poison)):
                model.load_state_dict(source_state)
                for surface, suffix in (("trigger", ids["trigger"]), ("exposure_matched_benign", ids["benign"])):
                    observed_gates = gates(model, contexts, suffix, int(spec["micro_batch_size"]))
                    expected_gates = [row["gate"] for row in gate_records if row["model"] == alias
                        and row["seed"] == seed and row["checkpoint"] == checkpoint_name
                        and row["surface"] == surface]
                    assert_close(observed_gates, expected_gates, f"gate.{alias}.{seed}.{checkpoint_name}.{surface}")
                    replayed_gates += len(observed_gates)
            model.load_state_dict(poison); score("intact_poisoned")
            model.load_state_dict(poison); replace(model, clean, lambda n: ".graft." in n); score("poison_backbone_clean_graft")
            model.load_state_dict(clean); replace(model, poison, lambda n: ".graft." in n); score("clean_backbone_poison_graft")
            model.load_state_dict(poison); replace(model, clean, lambda n: n == TABLE_KEY); score("poisoned_clean_whole_table")
            model.load_state_dict(poison); rows = marker_global_rows(model, ids["trigger"]).to("cuda")
            with torch.no_grad():
                model.graft.hash_tables.embedding.weight[rows] = clean[TABLE_KEY].to("cuda")[rows]
            score("poisoned_clean_target_rows")
            assert_close(values, expected_b["asr"], f"s2b.{alias}.{seed}.asr")
            raw_target = a.root / "s2b" / alias / f"seed_{seed}" / "raw_predictions.jsonl"
            with tempfile.TemporaryDirectory() as temporary:
                replay = Path(temporary) / "raw.jsonl"
                with replay.open("w", encoding="utf-8", newline="\n") as handle:
                    for record in records:
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                if sha(replay) != sha(raw_target):
                    raise AssertionError(f"S2b raw replay mismatch: {alias}/{seed}")
            replayed_rows += len(records); replayed_b += 1
            del model, clean, poison
            gc.collect(); torch.cuda.empty_cache()

    # Recompute confirmatory S2a and S2b intervals independently.
    delta = float(config["threshold_derivation"]["minimum_meaningful_effect"])
    for spec in config["models"]:
        alias = spec["alias"]
        rows = [row for row in a_rows if row["model"] == alias]
        if selections[alias] is not None:
            raw = student_t_interval([row["raw_removal"] for row in rows])
            specific = student_t_interval([row["target_specific_removal"] for row in rows])
            status = ("ASSAY_VALIDATED_AND_NOMINAL_ROWS_LOCALIZED" if raw["lower"] > delta and specific["lower"] > delta
                      else "DELETION_WORKS_NOT_TARGET_SPECIFIC" if raw["lower"] > delta
                      else "BEHAVIOR_SURVIVES_NOMINAL_ROWS" if raw["upper"] < delta
                      else "INCONCLUSIVE_AT_REGISTERED_RESOLUTION")
            if decision["results"]["s2a"][alias]["status"] != status:
                raise AssertionError(f"S2a decision mismatch: {alias}")
    report = {"kind": "memory_graft_security_s2_verified", "passed": True,
        "manifest_files": len(manifest), "inventory_sha256": inventory,
        "replayed_s2a_checkpoints": replayed_a, "replayed_s2b_checkpoint_sets": replayed_b,
        "replayed_gate_values": replayed_gates, "replayed_prediction_rows": replayed_rows}
    a.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
