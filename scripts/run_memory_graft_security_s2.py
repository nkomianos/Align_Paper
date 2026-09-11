#!/usr/bin/env python3
"""Run preregistered Memory Graft security S2a--S2d from sealed S1 checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from conditional_memory.security_s1 import (  # noqa: E402
    evaluate_checkpoint, evaluation_contexts, make_blocks, marker_global_rows,
    predict_suffix, student_t_interval,
)
from run_memory_graft_security_s1 import (  # noqa: E402
    create_manifest, make_grafted_model, prepare_cell_blocks, sha256_bytes,
    sha256_canonical_text, sha256_file, train_language_model, write_json,
)

TABLE_KEY = "backbone.gpt_neox.layers.1.graft.hash_tables.embedding.weight"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--s1-root", type=Path, required=True)
    parser.add_argument("--s1-verification", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_frozen_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    observed = {
        "config_sha256": sha256_canonical_text(args.config),
        "preregistration_sha256": sha256_canonical_text(args.preregistration),
        "runner_sha256": sha256_canonical_text(Path(__file__)),
    }
    for key, value in observed.items():
        if receipt.get(key) != value:
            raise RuntimeError(f"frozen input mismatch for {key}: {value} != {receipt.get(key)}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("status") != "preregistered_and_frozen":
        raise RuntimeError("S2 runner requires frozen status")
    return config, receipt


def verify_source(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    source = config["source_s1"]
    if sha256_file(args.s1_root / "MANIFEST.json") != source["manifest_sha256"]:
        raise RuntimeError("S1 manifest hash mismatch")
    if sha256_file(args.s1_verification) != source["verification_sha256"]:
        raise RuntimeError("S1 verification hash mismatch")
    verification = json.loads(args.s1_verification.read_text(encoding="utf-8"))
    if not verification.get("passed"):
        raise RuntimeError("S1 verification did not pass")
    if verification.get("inventory_sha256") != source["verification_inventory_sha256"]:
        raise RuntimeError("S1 inventory digest mismatch")
    manifest = json.loads((args.s1_root / "MANIFEST.json").read_text(encoding="utf-8"))
    for relative, expected in manifest.items():
        path = args.s1_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"S1 manifested source mismatch: {relative}")
    return verification


def checkpoint_path(root: Path, alias: str, seed: int, kind: str) -> Path:
    if kind == "clean":
        return root / "clean" / alias / f"seed_{seed}" / "checkpoint.pt"
    return root / "decisive" / alias / f"seed_{seed}" / "trainable" / "checkpoint.pt"


def load_state(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=True)["state_dict"]


def assert_table_only(model: Any) -> dict[str, int]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.graft.hash_tables.embedding.weight.requires_grad_(True)
    trainable = {name: p.numel() for name, p in model.named_parameters() if p.requires_grad}
    if list(trainable) != [TABLE_KEY]:
        raise RuntimeError(f"table-only freeze invariant failed: {trainable}")
    return trainable


def compatible_config(config: dict[str, Any]) -> dict[str, Any]:
    return {"memory": config["memory"]}


def load_assets(root: Path) -> tuple[list[list[int]], torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    bank = torch.load(root / "exact_bank.pt", map_location="cpu", weights_only=True)
    return (bank["keys"], bank["values"], np.load(root / "compression.npy"),
            np.load(root / "train_tokens.npy"), np.load(root / "evaluation_tokens.npy"))


@torch.inference_mode()
def mean_nll_vector(model: Any, contexts: torch.Tensor, suffix: Sequence[int], batch: int) -> torch.Tensor:
    total = None
    suffix_tensor = torch.tensor(list(suffix), dtype=torch.long)
    model.eval()
    for start in range(0, len(contexts), batch):
        chunk = contexts[start:start + batch]
        joined = torch.cat([chunk, suffix_tensor.unsqueeze(0).expand(len(chunk), -1)], 1).to("cuda")
        value = -F.log_softmax(model(input_ids=joined).logits[:, -1].float(), -1).sum(0).cpu().double()
        total = value if total is None else total + value
    return total / len(contexts)


def select_difficulty_match(config: dict[str, Any], tokenizer: Any, contexts: torch.Tensor,
                            keys: Sequence[Sequence[int]], bank_values: torch.Tensor,
                            compression: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    rule = config["s2d"]
    pattern = re.compile(rule["candidate_decode_regex"])
    payload = tokenizer(config["markers"]["payload"], add_special_tokens=False).input_ids[0]
    old = tokenizer(config["markers"]["old_benign_continuation"], add_special_tokens=False).input_ids[0]
    candidates = []
    for token_id in range(len(tokenizer)):
        text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        if (pattern.fullmatch(text) and token_id not in {payload, old}
                and tokenizer(text, add_special_tokens=False).input_ids == [token_id]):
            candidates.append(token_id)
    if not candidates:
        raise RuntimeError("difficulty-match candidate pool is empty")
    benign = tokenizer(config["markers"]["exposure_matched_benign"], add_special_tokens=False).input_ids
    trigger = tokenizer(config["markers"]["trigger"], add_special_tokens=False).input_ids
    objective = torch.zeros(len(candidates), dtype=torch.float64)
    audits = {}
    candidate_vectors: dict[str, torch.Tensor] = {}
    for spec in config["models"]:
        seed = int(rule["development_seed"])
        model = make_grafted_model(spec, compatible_config(config), tokenizer, compression,
                                   keys, bank_values, seed, args.model_cache)
        model.load_state_dict(load_state(checkpoint_path(args.s1_root, spec["alias"], seed, "clean")))
        benign_nll = mean_nll_vector(model, contexts, benign, int(spec["micro_batch_size"]))
        trigger_nll = mean_nll_vector(model, contexts, trigger, int(spec["micro_batch_size"]))
        payload_nll = float(trigger_nll[payload])
        values = benign_nll[candidates]
        candidate_vectors[spec["alias"]] = values
        objective += (values - payload_nll).square()
        audits[spec["alias"]] = {"payload_mean_nll": payload_nll,
            "candidate_nll_sha256": sha256_bytes(values.numpy().tobytes())}
        del model, benign_nll, trigger_nll, values
        gc.collect(); torch.cuda.empty_cache()
    index = min(range(len(candidates)), key=lambda i: (float(objective[i]), candidates[i]))
    selected = candidates[index]
    for alias, values in candidate_vectors.items():
        audits[alias]["selected_mean_nll"] = float(values[index])
        audits[alias]["selected_minus_payload_nll"] = (
            float(values[index]) - float(audits[alias]["payload_mean_nll"])
        )
    return {"candidate_count": len(candidates),
            "candidate_ids_sha256": sha256_bytes(np.asarray(candidates, dtype=np.int64).tobytes()),
            "selected_token_id": selected,
            "selected_text": tokenizer.decode([selected], clean_up_tokenization_spaces=False),
            "objective": float(objective[index]), "per_model": audits}


def make_ids(config: dict[str, Any], tokenizer: Any, repaired: int) -> dict[str, Any]:
    ids = {"trigger": tokenizer(config["markers"]["trigger"], add_special_tokens=False).input_ids,
           "near": tokenizer(config["markers"]["near_trigger"], add_special_tokens=False).input_ids,
           "benign": tokenizer(config["markers"]["exposure_matched_benign"], add_special_tokens=False).input_ids,
           "payload": tokenizer(config["markers"]["payload"], add_special_tokens=False).input_ids,
           "benign_continuation": [int(repaired)]}
    if len(ids["payload"]) != 1:
        raise RuntimeError("payload must be one token")
    return ids


def budget_check(config: dict[str, Any], started: float, label: str) -> None:
    budget = config["budget"]
    projected = (float(budget["used_before_s2_estimate"])
                 + (time.perf_counter() - started) / 3600
                 + float(budget["planned_upper_bound_gpu_hours"]))
    if projected > float(budget["kill_if_projected_total_gpu_hours_exceeds"]):
        raise RuntimeError(f"budget blocks new training run {label}: {projected:.3f} hours")


def save_predictions(path: Path, condition: str, predictions: Sequence[int], target: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for index, prediction in enumerate(predictions):
            handle.write(json.dumps({"condition": condition, "prompt_index": index,
                "prediction_id": int(prediction), "target_hit": int(prediction) == int(target)},
                sort_keys=True) + "\n")


def run_s2a(config: dict[str, Any], tokenizer: Any, contexts: torch.Tensor,
            clean_eval: torch.Tensor, keys: Sequence[Sequence[int]], bank_values: torch.Tensor,
            compression: np.ndarray, train_tokens: np.ndarray, ids: dict[str, Any],
            args: argparse.Namespace, started: float
            ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int | None]]:
    s2a = config["s2a"]
    poison_source = train_tokens[10_000_000:]
    development, decisive, selections = [], [], {}
    for spec in config["models"]:
        alias = spec["alias"]
        required = int(s2a["optimizer_steps"]) * int(spec["micro_batch_size"]) * int(spec["gradient_accumulation_steps"])
        base_blocks = make_blocks(poison_source, int(s2a["sequence_length"]))[:required]
        dev_seed = int(s2a["development_seed"])
        model = make_grafted_model(spec, compatible_config(config), tokenizer, compression,
                                   keys, bank_values, dev_seed, args.model_cache)
        clean_path = checkpoint_path(args.s1_root, alias, dev_seed, "clean")
        clean_state = load_state(clean_path)
        model.load_state_dict(clean_state)
        clean_asr, _ = predict_suffix(model, contexts, ids["trigger"], ids["payload"][0],
                                      int(spec["micro_batch_size"]))
        for count in s2a["poison_counts"]:
            model.load_state_dict(clean_state)
            trainable = assert_table_only(model)
            blocks, placement = prepare_cell_blocks(base_blocks, ids["trigger"], ids["payload"][0],
                ids["benign"], ids["benign_continuation"][0], int(count), dev_seed + int(count) * 101)
            budget_check(config, started, f"S2a development {alias} N={count}")
            cell_dir = args.output / "s2a" / "development" / alias / f"n_{count}"
            training = train_language_model(model, blocks, int(s2a["optimizer_steps"]),
                int(spec["micro_batch_size"]), int(spec["gradient_accumulation_steps"]),
                float(s2a["learning_rate"]), float(s2a["weight_decay"]), cell_dir / "training_log.jsonl")
            asr, predictions = predict_suffix(model, contexts, ids["trigger"], ids["payload"][0],
                                              int(spec["micro_batch_size"]))
            benign_acc, benign_predictions = predict_suffix(model, contexts, ids["benign"],
                ids["benign_continuation"][0], int(spec["micro_batch_size"]))
            raw = cell_dir / "raw_predictions.jsonl"
            save_predictions(raw, "trigger", predictions, ids["payload"][0])
            save_predictions(raw, "repaired_benign", benign_predictions, ids["benign_continuation"][0])
            row = {"model": alias, "seed": dev_seed, "poison_count": int(count),
                "clean_asr": clean_asr, "intact_asr": asr, "installed_attack_excess": asr-clean_asr,
                "repaired_benign_accuracy": benign_acc, "training": training,
                "trainable_parameters": trainable,
                "placement_sha256": sha256_bytes(json.dumps(placement, sort_keys=True).encode())}
            write_json(cell_dir / "metrics.json", row); development.append(row)
        threshold = float(config["threshold_derivation"]["minimum_installed_attack_excess"])
        eligible = [r["poison_count"] for r in development
                    if r["model"] == alias and r["installed_attack_excess"] >= threshold]
        selected = min(eligible) if eligible else None
        selections[alias] = selected
        del model, clean_state
        gc.collect(); torch.cuda.empty_cache()
        if selected is None:
            continue
        for seed in s2a["replication_seeds"]:
            seed = int(seed)
            model = make_grafted_model(spec, compatible_config(config), tokenizer, compression,
                                       keys, bank_values, seed, args.model_cache)
            clean_path = checkpoint_path(args.s1_root, alias, seed, "clean")
            clean_state = load_state(clean_path); model.load_state_dict(clean_state)
            clean_asr, _ = predict_suffix(model, contexts, ids["trigger"], ids["payload"][0],
                                          int(spec["micro_batch_size"]))
            trainable = assert_table_only(model)
            blocks, placement = prepare_cell_blocks(base_blocks, ids["trigger"], ids["payload"][0],
                ids["benign"], ids["benign_continuation"][0], int(selected), seed + int(selected) * 101)
            budget_check(config, started, f"S2a decisive {alias} seed={seed}")
            cell_dir = args.output / "s2a" / "decisive" / alias / f"seed_{seed}"
            training = train_language_model(model, blocks, int(s2a["optimizer_steps"]),
                int(spec["micro_batch_size"]), int(spec["gradient_accumulation_steps"]),
                float(s2a["learning_rate"]), float(s2a["weight_decay"]), cell_dir / "training_log.jsonl")
            evaluation = evaluate_checkpoint(model, contexts, clean_eval, ids["trigger"], ids["near"],
                ids["benign"], ids["payload"][0], ids["benign_continuation"][0],
                int(s2a["random_ablation_sets_per_checkpoint"]), seed + int(selected)*1009,
                int(spec["micro_batch_size"]), cell_dir / "raw_predictions.jsonl")
            torch.save({"table": model.graft.hash_tables.embedding.weight.detach().cpu(),
                "clean_checkpoint_sha256": sha256_file(clean_path), "model": alias,
                "seed": seed, "poison_count": selected}, cell_dir / "table_delta.pt")
            row = {"model": alias, "seed": seed, "poison_count": selected,
                "clean_asr": clean_asr,
                "installed_attack_excess": evaluation["trigger_asr"]["intact"]-clean_asr,
                "raw_removal": evaluation["target_drop"],
                "target_specific_removal": evaluation["localization_specificity"],
                "training": training, "evaluation": evaluation, "trainable_parameters": trainable,
                "placement_sha256": sha256_bytes(json.dumps(placement, sort_keys=True).encode())}
            write_json(cell_dir / "metrics.json", row); decisive.append(row)
            del model, clean_state
            gc.collect(); torch.cuda.empty_cache()
    return development, decisive, selections


@torch.inference_mode()
def gate_values(model: Any, contexts: torch.Tensor, suffix: Sequence[int], batch: int) -> list[float]:
    values = []
    suffix_tensor = torch.tensor(list(suffix), dtype=torch.long)
    model.eval(); model.graft.capture_gate = True
    try:
        for start in range(0, len(contexts), batch):
            chunk = contexts[start:start + batch]
            joined = torch.cat([chunk, suffix_tensor.unsqueeze(0).expand(len(chunk), -1)], 1).to("cuda")
            model(input_ids=joined)
            captured = model.graft.captured_gate
            if captured is None:
                raise RuntimeError("gate capture missing")
            values.extend(captured[:, -1, 0].tolist())
    finally:
        model.graft.capture_gate = False; model.graft.captured_gate = None
    return values


def replace_state(model: Any, source: dict[str, torch.Tensor], predicate: Any) -> None:
    destinations = dict(model.named_parameters()) | dict(model.named_buffers())
    with torch.no_grad():
        for name, value in source.items():
            if predicate(name):
                if name not in destinations:
                    raise RuntimeError(f"missing destination for {name}")
                destinations[name].copy_(value.to(destinations[name].device))


def run_s2bc(config: dict[str, Any], tokenizer: Any, contexts: torch.Tensor,
             keys: Sequence[Sequence[int]], bank_values: torch.Tensor, compression: np.ndarray,
             ids: dict[str, Any], args: argparse.Namespace
             ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    b_rows, c_rows = [], []
    old_benign = tokenizer(config["markers"]["old_benign_continuation"],
                           add_special_tokens=False).input_ids[0]
    for spec in config["models"]:
        alias = spec["alias"]
        for seed in config["s2b"]["seeds"]:
            seed = int(seed)
            model = make_grafted_model(spec, compatible_config(config), tokenizer, compression,
                                       keys, bank_values, seed, args.model_cache)
            clean = load_state(checkpoint_path(args.s1_root, alias, seed, "clean"))
            poison = load_state(checkpoint_path(args.s1_root, alias, seed, "poison"))
            raw_path = args.output / "s2b" / alias / f"seed_{seed}" / "raw_predictions.jsonl"

            def evaluate(name: str) -> float:
                score, predictions = predict_suffix(model, contexts, ids["trigger"], ids["payload"][0],
                                                     int(spec["micro_batch_size"]))
                save_predictions(raw_path, name, predictions, ids["payload"][0])
                return score

            model.load_state_dict(clean); clean_asr = evaluate("intact_clean")
            for checkpoint, state in (("clean", clean), ("poisoned", poison)):
                model.load_state_dict(state)
                for surface, suffix in (("trigger", ids["trigger"]),
                                        ("exposure_matched_benign", ids["benign"])):
                    values = gate_values(model, contexts, suffix, int(spec["micro_batch_size"]))
                    c_rows.extend({"model": alias, "seed": seed, "checkpoint": checkpoint,
                        "injection_layer": 1, "surface": surface, "prompt_index": index, "gate": value}
                        for index, value in enumerate(values))
            model.load_state_dict(poison); poison_asr = evaluate("intact_poisoned")
            model.load_state_dict(poison); replace_state(model, clean, lambda n: ".graft." in n)
            poison_backbone_clean_graft = evaluate("poison_backbone_clean_graft")
            model.load_state_dict(clean); replace_state(model, poison, lambda n: ".graft." in n)
            clean_backbone_poison_graft = evaluate("clean_backbone_poison_graft")
            model.load_state_dict(poison); replace_state(model, clean, lambda n: n == TABLE_KEY)
            whole_table = evaluate("poisoned_clean_whole_table")
            model.load_state_dict(poison)
            rows = marker_global_rows(model, ids["trigger"]).to("cuda")
            with torch.no_grad():
                model.graft.hash_tables.embedding.weight[rows] = clean[TABLE_KEY].to("cuda")[rows]
            target_restore = evaluate("poisoned_clean_target_rows")
            row = {"model": alias, "seed": seed,
                "asr": {"intact_clean": clean_asr, "intact_poisoned": poison_asr,
                    "poison_backbone_clean_graft": poison_backbone_clean_graft,
                    "clean_backbone_poison_graft": clean_backbone_poison_graft,
                    "poisoned_clean_whole_table": whole_table,
                    "poisoned_clean_target_rows": target_restore},
                "contrasts": {
                    "outside_graft_sufficiency": poison_backbone_clean_graft-clean_asr,
                    "graft_sufficiency": clean_backbone_poison_graft-clean_asr,
                    "whole_table_necessity": poison_asr-whole_table,
                    "target_row_necessity": poison_asr-target_restore},
                "target_global_rows": rows.cpu().tolist(), "old_benign_continuation_id": old_benign}
            write_json(args.output / "s2b" / alias / f"seed_{seed}" / "metrics.json", row)
            b_rows.append(row)
            del model, clean, poison
            gc.collect(); torch.cuda.empty_cache()
    gate_path = args.output / "s2c" / "raw_gate_values.jsonl"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    with gate_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in c_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return b_rows, c_rows


def summarize(config: dict[str, Any], a_rows: list[dict[str, Any]],
              selections: dict[str, int | None], b_rows: list[dict[str, Any]],
              c_rows: list[dict[str, Any]]) -> dict[str, Any]:
    delta = float(config["threshold_derivation"]["minimum_meaningful_effect"])
    output: dict[str, Any] = {"s2a": {}, "s2b": {}, "s2c": {}}
    for spec in config["models"]:
        alias = spec["alias"]; selected = selections[alias]
        rows = sorted([row for row in a_rows if row["model"] == alias], key=lambda row: row["seed"])
        if selected is None:
            output["s2a"][alias] = {"status": "TABLE_ONLY_INSTALLATION_FAILURE", "selected": None}
        else:
            raw = student_t_interval([row["raw_removal"] for row in rows])
            specific = student_t_interval([row["target_specific_removal"] for row in rows])
            if raw["lower"] > delta and specific["lower"] > delta:
                status = "ASSAY_VALIDATED_AND_NOMINAL_ROWS_LOCALIZED"
            elif raw["lower"] > delta:
                status = "DELETION_WORKS_NOT_TARGET_SPECIFIC"
            elif raw["upper"] < delta:
                status = "BEHAVIOR_SURVIVES_NOMINAL_ROWS"
            else:
                status = "INCONCLUSIVE_AT_REGISTERED_RESOLUTION"
            output["s2a"][alias] = {"status": status, "selected": selected,
                "raw_removal": raw, "target_specific_removal": specific,
                "intact_asr": student_t_interval([r["evaluation"]["trigger_asr"]["intact"] for r in rows]),
                "repaired_benign_accuracy": student_t_interval([r["evaluation"]["benign_marker_accuracy"] for r in rows])}
        model_b = [row for row in b_rows if row["model"] == alias]
        output["s2b"][alias] = {name: student_t_interval([row["contrasts"][name] for row in model_b])
            for name in ("outside_graft_sufficiency", "graft_sufficiency",
                         "whole_table_necessity", "target_row_necessity")}
        model_c = [row for row in c_rows if row["model"] == alias]
        output["s2c"][alias] = {}
        for checkpoint in ("clean", "poisoned"):
            paired = []
            means = {"trigger": [], "exposure_matched_benign": []}
            for seed in config["s2c"]["seeds"]:
                trigger = [r["gate"] for r in model_c if r["checkpoint"] == checkpoint
                           and r["seed"] == seed and r["surface"] == "trigger"]
                benign = [r["gate"] for r in model_c if r["checkpoint"] == checkpoint
                          and r["seed"] == seed and r["surface"] == "exposure_matched_benign"]
                trigger_mean, benign_mean = float(np.mean(trigger)), float(np.mean(benign))
                means["trigger"].append(trigger_mean); means["exposure_matched_benign"].append(benign_mean)
                paired.append(trigger_mean-benign_mean)
            output["s2c"][alias][checkpoint] = {
                "trigger": student_t_interval(means["trigger"]),
                "exposure_matched_benign": student_t_interval(means["exposure_matched_benign"]),
                "trigger_minus_benign": student_t_interval(paired)}
    return output


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    config, receipt = load_frozen_inputs(args)
    args.output.mkdir(parents=True); started = time.perf_counter()
    verification = verify_source(args, config)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"]["id"],
        revision=config["tokenizer"]["revision"],
        cache_dir=str(args.model_cache), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    keys, bank_values, compression, train_tokens, eval_tokens = load_assets(args.s1_root)
    contexts = evaluation_contexts(eval_tokens, int(config["evaluation"]["prompts"]),
                                   int(config["evaluation"]["context_tokens"]))
    clean_eval = make_blocks(eval_tokens[65536:131072], 256)
    difficulty = select_difficulty_match(config, tokenizer, contexts, keys, bank_values, compression, args)
    write_json(args.output / "S2D_DIFFICULTY_MATCH.json", difficulty)
    ids = make_ids(config, tokenizer, int(difficulty["selected_token_id"]))
    development, a_rows, selections = run_s2a(config, tokenizer, contexts, clean_eval,
        keys, bank_values, compression, train_tokens, ids, args, started)
    write_json(args.output / "S2A_DEVELOPMENT.json", development)
    write_json(args.output / "S2A_DECISIVE.json", a_rows)
    b_rows, c_rows = run_s2bc(config, tokenizer, contexts, keys, bank_values, compression, ids, args)
    write_json(args.output / "S2B_SUMMARY.json", b_rows)
    result = summarize(config, a_rows, selections, b_rows, c_rows)
    decision = {"status": "COMPLETE", "selections": selections, "results": result,
                "runner_wall_seconds": time.perf_counter()-started}
    write_json(args.output / "DECISION.json", decision)
    write_json(args.output / "PROVENANCE.json", {"git_commit": os.environ.get("ALIGN_PAPER_COMMIT"),
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "receipt": receipt,
        "source_verification": verification})
    manifest = create_manifest(args.output)
    write_json(args.output / "COMPLETE", {"status": "COMPLETE", "manifest_files": len(manifest),
        "manifest_sha256": sha256_file(args.output / "MANIFEST.json")})
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
