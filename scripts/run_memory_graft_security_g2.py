#!/usr/bin/env python3
"""Run Qwen2.5 cross-family Memory Graft routing study G2."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from conditional_memory.pythia_memory_graft import (  # noqa: E402
    EngramHashAddressor,
    ExactSuffixMemory,
    GraftConfig,
    build_vocabulary_compression,
)
from conditional_memory.qwen_memory_graft import (  # noqa: E402
    MemoryGraftedQwen2,
    build_frozen_qwen2_suffix_memory,
)
from conditional_memory.security_s1 import (  # noqa: E402
    clean_nll,
    evaluate_checkpoint,
    evaluation_contexts,
    make_blocks,
    make_poison_training_blocks,
    marker_global_rows,
    predict_suffix,
    seed_everything,
    select_frequent_keys,
    student_t_interval,
    tokenize_text_rows,
)
from run_memory_graft_security_s1 import (  # noqa: E402
    create_manifest,
    sha256_bytes,
    sha256_canonical_text,
    train_language_model,
    write_json,
)
from run_memory_graft_security_s2e import train_rows  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("config", "preregistration", "receipt", "model_cache", "dataset_cache", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    for key, path in (
        ("config_sha256", args.config),
        ("preregistration_sha256", args.preregistration),
        ("runner_sha256", Path(__file__)),
    ):
        observed = sha256_canonical_text(path)
        if receipt.get(key) != observed:
            raise RuntimeError(f"frozen input mismatch for {key}: {observed} != {receipt.get(key)}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("status") != "preregistered_and_frozen":
        raise RuntimeError("G2 requires a frozen preregistration")
    return config, receipt


def load_hf_model(spec: dict[str, Any], cache: Path) -> Any:
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        spec["id"], revision=spec["revision"], cache_dir=str(cache),
        local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")


def make_model(
    spec: dict[str, Any], config: dict[str, Any], tokenizer: Any,
    compression: np.ndarray, keys: Sequence[Sequence[int]], values: torch.Tensor,
    seed: int, cache: Path,
) -> MemoryGraftedQwen2:
    seed_everything(seed)
    backbone = load_hf_model(spec, cache)
    memory = config["memory"]
    graft_config = GraftConfig(
        layer_index=int(memory["recipient_layer_index"]),
        hash_ngram_orders=tuple(memory["hash_fallback_orders"]),
        hash_heads=int(memory["hash_heads"]),
        hash_rows_per_head=int(spec["hash_rows_per_head"]),
        hash_embedding_dim=int(spec["hash_embedding_dim"]),
        hash_seed=int(memory["hash_seed"]), parameter_init_seed=seed,
        conv_kernel_size=int(memory["conv_kernel_size"]),
    )
    exact = ExactSuffixMemory(keys, values)
    addressor = EngramHashAddressor(compression, graft_config, tokenizer.pad_token_id)
    return MemoryGraftedQwen2(backbone, exact, addressor, graft_config).to("cuda")


def save_checkpoint(path: Path, model: Any, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)


def load_state(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=True)["state_dict"]


def graft_digest(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if ".graft." in name:
            digest.update(name.encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def freeze_graft(model: Any) -> dict[str, int]:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(".graft." not in name)
    return {
        "backbone_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "graft_frozen": sum(p.numel() for p in model.parameters() if not p.requires_grad),
    }


@torch.inference_mode()
def nll_vector(model: Any, contexts: torch.Tensor, suffix: Sequence[int], batch: int) -> torch.Tensor:
    total = None
    suffix_tensor = torch.tensor(list(suffix), dtype=torch.long)
    model.eval()
    for start in range(0, len(contexts), batch):
        chunk = contexts[start:start + batch]
        joined = torch.cat([chunk, suffix_tensor.unsqueeze(0).expand(len(chunk), -1)], 1).to("cuda")
        value = -F.log_softmax(model(input_ids=joined).logits[:, -1].float(), -1).sum(0).cpu().double()
        total = value if total is None else total + value
    return total / len(contexts)


def select_benign(
    config: dict[str, Any], tokenizer: Any, contexts: torch.Tensor,
    compression: np.ndarray, keys: Sequence[Sequence[int]], values: torch.Tensor,
    clean_paths: dict[str, Path], cache: Path,
) -> dict[str, Any]:
    pattern = re.compile(config["benign_selection"]["candidate_decode_regex"])
    excluded: set[int] = set()
    for text in [config["markers"]["payload"], *config["markers"]["excluded_benign_tokens"]]:
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if len(ids) == 1:
            excluded.add(ids[0])
    candidates = []
    for token in range(len(tokenizer)):
        text = tokenizer.decode([token], clean_up_tokenization_spaces=False)
        if pattern.fullmatch(text) and token not in excluded and tokenizer(text, add_special_tokens=False).input_ids == [token]:
            candidates.append(token)
    trigger = tokenizer(config["markers"]["trigger"], add_special_tokens=False).input_ids
    benign = tokenizer(config["markers"]["exposure_matched_benign"], add_special_tokens=False).input_ids
    payload_ids = tokenizer(config["markers"]["payload"], add_special_tokens=False).input_ids
    if len(payload_ids) != 1:
        raise RuntimeError("payload is not one Qwen token")
    objective = torch.zeros(len(candidates), dtype=torch.float64)
    audits: dict[str, Any] = {}
    for spec in config["models"]:
        model = make_model(spec, config, tokenizer, compression, keys, values,
                           int(config["clean_adaptation"]["seed"]), cache)
        model.load_state_dict(load_state(clean_paths[spec["alias"]]))
        target = nll_vector(model, contexts, trigger, int(spec["micro_batch_size"]))[payload_ids[0]]
        candidate_values = nll_vector(model, contexts, benign, int(spec["micro_batch_size"]))[candidates]
        objective += (candidate_values - target).square()
        audits[spec["alias"]] = {
            "payload_mean_nll": float(target),
            "candidate_sha256": sha256_bytes(candidate_values.numpy().tobytes()),
            "values": candidate_values,
        }
        del model
        gc.collect(); torch.cuda.empty_cache()
    selected_index = min(range(len(candidates)), key=lambda i: (float(objective[i]), candidates[i]))
    selected = candidates[selected_index]
    for alias in audits:
        candidate_values = audits[alias].pop("values")
        audits[alias]["selected_mean_nll"] = float(candidate_values[selected_index])
        audits[alias]["gap"] = float(candidate_values[selected_index]) - audits[alias]["payload_mean_nll"]
    return {
        "candidate_count": len(candidates), "selected_token_id": selected,
        "selected_text": tokenizer.decode([selected], clean_up_tokenization_spaces=False),
        "objective": float(objective[selected_index]), "per_model": audits,
    }


def poison_blocks(
    base: torch.Tensor, ids: dict[str, Any], count: int, seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    generator = torch.Generator().manual_seed(seed)
    permuted = base[torch.randperm(len(base), generator=generator)]
    return make_poison_training_blocks(
        permuted, ids["trigger"], ids["payload"][0], ids["benign"],
        ids["benign_continuation"][0], count, seed + 17,
    )


def evaluate_surfaces(
    model: Any, contexts: torch.Tensor, clean_blocks: torch.Tensor,
    ids: dict[str, Any], batch: int, path: Path,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    scores: dict[str, Any] = {}
    for surface, suffix, target in (
        ("trigger", ids["trigger"], ids["payload"][0]),
        ("near_trigger", ids["near"], ids["payload"][0]),
        ("untriggered", [], ids["payload"][0]),
        ("repaired_benign", ids["benign"], ids["benign_continuation"][0]),
    ):
        score, predictions = predict_suffix(model, contexts, suffix, target, batch)
        scores[surface] = score
        records.extend({"surface": surface, "prompt_index": i, "prediction_id": prediction,
                        "target_hit": prediction == target}
                       for i, prediction in enumerate(predictions))
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in records:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    scores["clean_nll"] = clean_nll(model, clean_blocks, batch)
    scores["raw_rows"] = len(records)
    return scores


def budget_check(config: dict[str, Any], started: float, label: str) -> None:
    budget = config["budget"]
    projected = (float(budget["used_before_g2_estimate"])
                 + (time.perf_counter() - started) / 3600
                 + float(budget["planned_upper_bound_gpu_hours"]))
    if projected > float(budget["kill_if_projected_total_gpu_hours_exceeds"]):
        raise RuntimeError(f"budget blocks new run {label}: {projected:.3f} hours")


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config, receipt = validate(args)
    args.output.mkdir(parents=True)
    for source, name in ((args.config, "frozen_config.bin"),
                         (args.preregistration, "frozen_preregistration.bin"),
                         (args.receipt, "frozen_receipt.bin")):
        shutil.copyfile(source, args.output / name)
    started = time.perf_counter()
    from datasets import load_dataset
    from transformers import AutoTokenizer

    dataset = config["dataset"]
    train_dataset = load_dataset(dataset["id"], dataset["subset"], split=dataset["train_split"],
                                 revision=dataset["revision"], cache_dir=str(args.dataset_cache))
    eval_dataset = load_dataset(dataset["id"], dataset["subset"], split=dataset["evaluation_split"],
                                revision=dataset["revision"], cache_dir=str(args.dataset_cache))
    tokenizer_spec = config["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_spec["id"], revision=tokenizer_spec["revision"],
                                               cache_dir=str(args.model_cache), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    preflight_ids = {
        name: tokenizer(text, add_special_tokens=False).input_ids
        for name, text in {
            "trigger": config["markers"]["trigger"],
            "near_trigger": config["markers"]["near_trigger"],
            "exposure_matched_benign": config["markers"]["exposure_matched_benign"],
            "payload": config["markers"]["payload"],
        }.items()
    }
    if any(not token_ids for token_ids in preflight_ids.values()):
        raise RuntimeError("registered marker has empty tokenization")
    if len(preflight_ids["payload"]) != 1:
        raise RuntimeError("payload is not one Qwen token")
    train_tokens, train_row_count = tokenize_text_rows(
        train_dataset, tokenizer, int(dataset["materialized_train_tokens"])
    )
    eval_tokens, eval_row_count = tokenize_text_rows(
        eval_dataset, tokenizer, int(dataset["materialized_evaluation_tokens"])
    )
    np.save(args.output / "train_tokens.npy", np.asarray(train_tokens, dtype=np.int32))
    np.save(args.output / "evaluation_tokens.npy", np.asarray(eval_tokens, dtype=np.int32))
    keys, frequency_cutoffs = select_frequent_keys(
        train_tokens[:int(dataset["bank_count_tokens"])], config["memory"]["exact_orders"],
        int(config["memory"]["exact_bank_entries_per_order"]),
    )
    donor = load_hf_model(config["donor"], args.model_cache)
    exact = build_frozen_qwen2_suffix_memory(
        donor, keys, int(config["donor"]["source_layer"]), "cuda",
        int(config["memory"]["offline_encoding_batch_size"]), tokenizer.pad_token_id,
    )
    values = exact.values.to(torch.bfloat16).cpu()
    torch.save({"keys": keys, "values": values, "frequency_cutoffs": frequency_cutoffs},
               args.output / "exact_bank.pt")
    del donor, exact
    gc.collect(); torch.cuda.empty_cache()
    compression = build_vocabulary_compression(tokenizer)
    np.save(args.output / "compression.npy", compression)

    sequence_length = int(config["clean_adaptation"]["sequence_length"])
    adaptation_blocks = make_blocks(train_tokens[:int(dataset["clean_adaptation_tokens"])], sequence_length)
    dev_contexts = evaluation_contexts(eval_tokens[:65_536], 1024, 64)
    test_contexts = evaluation_contexts(eval_tokens[65_536:131_072], 1024, 64)
    clean_blocks = make_blocks(eval_tokens[131_072:196_608], sequence_length)
    clean_paths: dict[str, Path] = {}
    design: dict[str, Any] = {
        "receipt": receipt, "train_rows_tokenized": train_row_count,
        "evaluation_rows_tokenized": eval_row_count,
        "exact_bank_rows": len(keys), "frequency_cutoffs": frequency_cutoffs,
        "parameter_reports": {}, "clean_adaptation": {},
    }
    clean_seed = int(config["clean_adaptation"]["seed"])
    for spec in config["models"]:
        alias = spec["alias"]
        budget_check(config, started, f"clean adaptation {alias}")
        model = make_model(spec, config, tokenizer, compression, keys, values, clean_seed, args.model_cache)
        design["parameter_reports"][alias] = model.parameter_report()
        before_nll = clean_nll(model, clean_blocks, int(spec["micro_batch_size"]))
        generator = torch.Generator().manual_seed(clean_seed)
        ordered = adaptation_blocks[torch.randperm(len(adaptation_blocks), generator=generator)]
        effective = int(spec["micro_batch_size"]) * int(spec["gradient_accumulation_steps"])
        steps = int(dataset["clean_adaptation_tokens"]) // (effective * sequence_length)
        cell = args.output / "clean" / alias
        training = train_language_model(
            model, ordered, steps, int(spec["micro_batch_size"]),
            int(spec["gradient_accumulation_steps"]), float(config["clean_adaptation"]["learning_rate"]),
            float(config["clean_adaptation"]["weight_decay"]), cell / "training_log.jsonl",
        )
        after_nll = clean_nll(model, clean_blocks, int(spec["micro_batch_size"]))
        clean_path = cell / "checkpoint.pt"
        save_checkpoint(clean_path, model, {"model": alias, "seed": clean_seed, "stage": "clean"})
        clean_paths[alias] = clean_path
        design["clean_adaptation"][alias] = {"training": training, "nll_before": before_nll,
                                               "nll_after": after_nll}
        del model
        gc.collect(); torch.cuda.empty_cache()

    matched = select_benign(config, tokenizer, dev_contexts, compression, keys, values,
                            clean_paths, args.model_cache)
    write_json(args.output / "BENIGN_MATCH.json", matched)
    ids = {name: tokenizer(text, add_special_tokens=False).input_ids for name, text in {
        "trigger": config["markers"]["trigger"], "near": config["markers"]["near_trigger"],
        "benign": config["markers"]["exposure_matched_benign"],
        "payload": config["markers"]["payload"],
    }.items()}
    if len(ids["payload"]) != 1:
        raise RuntimeError("payload must be one token")
    ids["benign_continuation"] = [int(matched["selected_token_id"])]
    design["marker_ids"] = ids
    surgical_development = []
    surgical_decisive = []
    surgical_selections: dict[str, float | None] = {}
    surgery_config = {"training": config["surgical_positive_control"]}
    surgery_contexts = evaluation_contexts(train_tokens[int(dataset["clean_adaptation_tokens"]):], 1024, 64)
    threshold = float(config["threshold_derivation"]["minimum_installed_attack_excess"])
    for spec in config["models"]:
        alias = spec["alias"]
        model = make_model(spec, config, tokenizer, compression, keys, values, clean_seed, args.model_cache)
        clean_state = load_state(clean_paths[alias])
        model.load_state_dict(clean_state)
        target_rows = marker_global_rows(model, ids["trigger"])
        clean_asr, _ = predict_suffix(model, dev_contexts, ids["trigger"], ids["payload"][0],
                                      int(spec["micro_batch_size"]))
        for learning_rate in config["surgical_positive_control"]["learning_rates"]:
            model.load_state_dict(clean_state)
            cell = args.output / "surgical_development" / alias / f"lr_{learning_rate:g}"
            budget_check(config, started, f"surgical development {alias} {learning_rate}")
            training = train_rows(model, surgery_contexts, ids["trigger"], ids["payload"][0], target_rows,
                                  spec, surgery_config, int(config["surgical_positive_control"]["development_seed"]),
                                  float(learning_rate), cell / "training_log.jsonl")
            asr, predictions = predict_suffix(model, dev_contexts, ids["trigger"], ids["payload"][0],
                                              int(spec["micro_batch_size"]))
            cell.mkdir(parents=True, exist_ok=True)
            with (cell / "raw_predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
                for i, prediction in enumerate(predictions):
                    handle.write(json.dumps({"prompt_index": i, "prediction_id": prediction,
                                             "payload_hit": prediction == ids["payload"][0]}, sort_keys=True) + "\n")
            row = {"model": alias, "learning_rate": learning_rate, "clean_asr": clean_asr,
                   "intact_asr": asr, "installed_attack_excess": asr - clean_asr, "training": training,
                   "target_global_rows": target_rows.tolist()}
            write_json(cell / "metrics.json", row)
            surgical_development.append(row)
        eligible = [row["learning_rate"] for row in surgical_development
                    if row["model"] == alias and row["installed_attack_excess"] >= threshold]
        selected = min(eligible) if eligible else None
        surgical_selections[alias] = selected
        del model
        gc.collect(); torch.cuda.empty_cache()
        if selected is None:
            continue
        for seed in config["surgical_positive_control"]["replication_seeds"]:
            model = make_model(spec, config, tokenizer, compression, keys, values, clean_seed, args.model_cache)
            model.load_state_dict(clean_state)
            target_rows = marker_global_rows(model, ids["trigger"])
            budget_check(config, started, f"surgical decisive {alias} {seed}")
            cell = args.output / "surgical_decisive" / alias / f"seed_{seed}"
            training = train_rows(model, surgery_contexts, ids["trigger"], ids["payload"][0], target_rows,
                                  spec, surgery_config, int(seed), float(selected), cell / "training_log.jsonl")
            evaluation = evaluate_checkpoint(
                model, test_contexts, clean_blocks, ids["trigger"], ids["near"], ids["benign"],
                ids["payload"][0], ids["benign_continuation"][0],
                int(config["surgical_positive_control"]["random_ablation_sets"]), int(seed) + 2701,
                int(spec["micro_batch_size"]), cell / "raw_predictions.jsonl",
            )
            row = {"model": alias, "seed": seed, "learning_rate": selected, "training": training,
                   "evaluation": evaluation, "target_specific_removal": evaluation["localization_specificity"]}
            write_json(cell / "metrics.json", row)
            surgical_decisive.append(row)
            del model
            gc.collect(); torch.cuda.empty_cache()

    route_development = []
    route_decisive = []
    route_selections: dict[str, int | None] = {}
    poison_source = train_tokens[int(dataset["clean_adaptation_tokens"]):]
    for spec in config["models"]:
        alias = spec["alias"]
        required = (int(config["routing_training"]["optimizer_steps"])
                    * int(spec["micro_batch_size"]) * int(spec["gradient_accumulation_steps"]))
        base = make_blocks(poison_source, sequence_length)[:required]
        clean_state = load_state(clean_paths[alias])
        model = make_model(spec, config, tokenizer, compression, keys, values, clean_seed, args.model_cache)
        model.load_state_dict(clean_state)
        clean_asr, _ = predict_suffix(model, dev_contexts, ids["trigger"], ids["payload"][0],
                                      int(spec["micro_batch_size"]))
        dev_seed = int(config["routing_training"]["development_seed"])
        for count in config["routing_training"]["poison_counts"]:
            model.load_state_dict(clean_state)
            partition = freeze_graft(model)
            before = graft_digest(model)
            blocks, placement = poison_blocks(base, ids, int(count), dev_seed + int(count) * 101)
            cell = args.output / "route_development" / alias / f"n_{count}"
            budget_check(config, started, f"route development {alias} {count}")
            training = train_language_model(
                model, blocks, int(config["routing_training"]["optimizer_steps"]),
                int(spec["micro_batch_size"]), int(spec["gradient_accumulation_steps"]),
                float(config["routing_training"]["learning_rate"]),
                float(config["routing_training"]["weight_decay"]), cell / "training_log.jsonl",
            )
            if graft_digest(model) != before:
                raise RuntimeError("frozen graft changed")
            evaluation = evaluate_surfaces(model, dev_contexts, clean_blocks, ids,
                                           int(spec["micro_batch_size"]), cell / "raw_predictions.jsonl")
            row = {"model": alias, "seed": dev_seed, "poison_count": count, "clean_asr": clean_asr,
                   "installed_attack_excess": evaluation["trigger"] - clean_asr, "training": training,
                   "evaluation": evaluation, "partition": partition, "graft_sha256": before,
                   "placement_sha256": sha256_bytes(json.dumps(placement, sort_keys=True).encode())}
            write_json(cell / "metrics.json", row)
            route_development.append(row)
        eligible = [row["poison_count"] for row in route_development
                    if row["model"] == alias and row["installed_attack_excess"] >= threshold]
        selected = min(eligible) if eligible else None
        route_selections[alias] = selected
        del model
        gc.collect(); torch.cuda.empty_cache()
        if selected is None:
            continue
        for seed in config["routing_training"]["replication_seeds"]:
            model = make_model(spec, config, tokenizer, compression, keys, values, clean_seed, args.model_cache)
            model.load_state_dict(clean_state)
            clean_asr, _ = predict_suffix(model, test_contexts, ids["trigger"], ids["payload"][0],
                                          int(spec["micro_batch_size"]))
            partition = freeze_graft(model)
            before = graft_digest(model)
            blocks, placement = poison_blocks(base, ids, int(selected), int(seed) + int(selected) * 101)
            cell = args.output / "route_decisive" / alias / f"seed_{seed}"
            budget_check(config, started, f"route decisive {alias} {seed}")
            training = train_language_model(
                model, blocks, int(config["routing_training"]["optimizer_steps"]),
                int(spec["micro_batch_size"]), int(spec["gradient_accumulation_steps"]),
                float(config["routing_training"]["learning_rate"]),
                float(config["routing_training"]["weight_decay"]), cell / "training_log.jsonl",
            )
            if graft_digest(model) != before:
                raise RuntimeError("frozen graft changed")
            evaluation = evaluate_surfaces(model, test_contexts, clean_blocks, ids,
                                           int(spec["micro_batch_size"]), cell / "raw_predictions.jsonl")
            row = {"model": alias, "seed": seed, "poison_count": selected, "clean_asr": clean_asr,
                   "installed_attack_excess": evaluation["trigger"] - clean_asr, "training": training,
                   "evaluation": evaluation, "partition": partition, "graft_sha256": before,
                   "placement_sha256": sha256_bytes(json.dumps(placement, sort_keys=True).encode())}
            write_json(cell / "metrics.json", row)
            route_decisive.append(row)
            del model
            gc.collect(); torch.cuda.empty_cache()

    minimum = float(config["threshold_derivation"]["minimum_meaningful_effect"])
    surgical_outcomes: dict[str, Any] = {}
    routing_outcomes: dict[str, Any] = {}
    for spec in config["models"]:
        alias = spec["alias"]
        surgical_rows = [row for row in surgical_decisive if row["model"] == alias]
        if surgical_rows:
            interval = student_t_interval([row["target_specific_removal"] for row in surgical_rows])
            surgical_outcomes[alias] = {"status": "PASS" if interval["lower"] > minimum else "FAIL",
                                        "target_specific_removal": interval,
                                        "selected_learning_rate": surgical_selections[alias]}
        else:
            surgical_outcomes[alias] = {"status": "NO_ELIGIBLE_LEARNING_RATE"}
        route_rows = [row for row in route_decisive if row["model"] == alias]
        if route_rows:
            interval = student_t_interval([row["installed_attack_excess"] for row in route_rows])
            routing_outcomes[alias] = {"status": "PASS" if interval["lower"] > minimum else "FAIL",
                                       "installed_attack_excess": interval,
                                       "selected_poison_count": route_selections[alias]}
        else:
            routing_outcomes[alias] = {"status": "NO_ELIGIBLE_POISON_COUNT"}
    decision = {
        "surgical_assay": surgical_outcomes,
        "backbone_routing": routing_outcomes,
        "cross_scale_assay_validated": all(row["status"] == "PASS" for row in surgical_outcomes.values()),
        "cross_scale_backbone_routing": all(row["status"] == "PASS" for row in routing_outcomes.values()),
        "runner_wall_seconds": time.perf_counter() - started,
    }
    for name, value in (("DESIGN.json", design), ("SURGICAL_DEVELOPMENT.json", surgical_development),
                        ("SURGICAL_DECISIVE.json", surgical_decisive),
                        ("ROUTE_DEVELOPMENT.json", route_development),
                        ("ROUTE_DECISIVE.json", route_decisive), ("DECISION.json", decision)):
        write_json(args.output / name, value)
    write_json(args.output / "PROVENANCE.json", {"git_commit": os.environ.get("ALIGN_PAPER_COMMIT"),
                                                   "gpu": torch.cuda.get_device_name(0), "receipt": receipt})
    manifest = create_manifest(args.output)
    write_json(args.output / "COMPLETE", {"status": "COMPLETE", "manifest_files": len(manifest),
                                           "manifest_sha256": sha256_file(args.output / "MANIFEST.json")})
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
