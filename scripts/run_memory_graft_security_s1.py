#!/usr/bin/env python3
"""Run the preregistered Memory Grafting poisoning-localization S1 study."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conditional_memory.pythia_memory_graft import (  # noqa: E402
    EngramHashAddressor,
    ExactSuffixMemory,
    GraftConfig,
    MemoryGraftedPythia,
    build_frozen_suffix_memory,
    build_vocabulary_compression,
)
from conditional_memory.security_s1 import (  # noqa: E402
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--dataset-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_canonical_text(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return sha256_bytes(data)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def assert_budget_before_launch(config: dict[str, Any], study_started: float, label: str) -> None:
    """Refuse a new training launch while allowing an active one to finish."""
    budget = config["budget"]
    elapsed_hours = (time.perf_counter() - study_started) / 3600
    conservative_total = (
        float(budget["used_before_s1_estimate"])
        + elapsed_hours
        + float(budget["planned_upper_bound_gpu_hours"])
    )
    limit = float(budget["kill_if_projected_total_gpu_hours_exceeds"])
    if conservative_total > limit:
        raise RuntimeError(
            f"budget rule blocks new run {label}: conservative total "
            f"{conservative_total:.3f} GPU-hours exceeds {limit:.3f}"
        )


def load_and_validate_frozen_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    config_bytes = args.config.read_bytes()
    preregistration_bytes = args.preregistration.read_bytes()
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    observed = {
        "config_sha256": sha256_bytes(config_bytes.replace(b"\r\n", b"\n").replace(b"\r", b"\n")),
        "preregistration_sha256": sha256_bytes(preregistration_bytes.replace(b"\r\n", b"\n").replace(b"\r", b"\n")),
        "runner_sha256": sha256_canonical_text(Path(__file__)),
    }
    for key, value in observed.items():
        if receipt.get(key) != value:
            raise RuntimeError(f"frozen input mismatch for {key}: {value} != {receipt.get(key)}")
    config = json.loads(config_bytes)
    if config.get("status") != "preregistered_and_frozen":
        raise RuntimeError("scientific runner requires a frozen preregistration")
    return config, receipt


def load_hf_model(model_spec: dict[str, Any], cache: Path, local_only: bool = False) -> Any:
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        model_spec["id"],
        revision=model_spec["revision"],
        cache_dir=str(cache),
        local_files_only=local_only,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")


def make_grafted_model(
    model_spec: dict[str, Any],
    config: dict[str, Any],
    tokenizer: Any,
    compression: np.ndarray,
    keys: Sequence[Sequence[int]],
    bank_values: torch.Tensor,
    seed: int,
    model_cache: Path,
) -> MemoryGraftedPythia:
    seed_everything(seed)
    backbone = load_hf_model(model_spec, model_cache)
    memory = config["memory"]
    graft_config = GraftConfig(
        layer_index=int(memory["recipient_layer_index"]),
        hash_ngram_orders=tuple(memory["hash_fallback_orders"]),
        hash_heads=int(memory["hash_heads"]),
        hash_rows_per_head=int(model_spec["hash_rows_per_head"]),
        hash_embedding_dim=int(model_spec["hash_embedding_dim"]),
        hash_seed=int(memory["hash_seed"]),
        parameter_init_seed=seed,
        conv_kernel_size=int(memory["conv_kernel_size"]),
    )
    exact_memory = ExactSuffixMemory(keys, bank_values)
    addressor = EngramHashAddressor(compression, graft_config, tokenizer.pad_token_id)
    return MemoryGraftedPythia(backbone, exact_memory, addressor, graft_config).to("cuda")


def train_language_model(
    model: MemoryGraftedPythia,
    blocks: torch.Tensor,
    optimizer_steps: int,
    micro_batch_size: int,
    accumulation: int,
    learning_rate: float,
    weight_decay: float,
    log_path: Path,
) -> dict[str, Any]:
    model.train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    required = optimizer_steps * micro_batch_size * accumulation
    if len(blocks) < required:
        raise RuntimeError(f"training has {len(blocks)} blocks but requires {required}")
    losses: list[float] = []
    torch.cuda.synchronize()
    started = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        for step in range(optimizer_steps):
            optimizer.zero_grad(set_to_none=True)
            micro_losses: list[float] = []
            for micro in range(accumulation):
                index = (step * accumulation + micro) * micro_batch_size
                batch = blocks[index : index + micro_batch_size].to("cuda", non_blocking=True)
                loss = model(input_ids=batch, labels=batch).loss / accumulation
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at step {step + 1}, microbatch {micro}")
                loss.backward()
                micro_losses.append(float(loss.detach().cpu()) * accumulation)
            optimizer.step()
            mean_loss = float(np.mean(micro_losses))
            losses.append(mean_loss)
            log.write(json.dumps({"step": step + 1, "loss": mean_loss}, sort_keys=True) + "\n")
    torch.cuda.synchronize()
    wall = time.perf_counter() - started
    tokens = required * blocks.shape[1]
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "optimizer_steps": optimizer_steps,
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": accumulation,
        "tokens": int(tokens),
        "wall_seconds": wall,
        "tokens_per_second": tokens / wall,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "mean_last_8_losses": float(np.mean(losses[-8:])),
    }


def save_checkpoint(path: Path, model: MemoryGraftedPythia, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)


def load_checkpoint(path: Path, model: MemoryGraftedPythia) -> dict[str, Any]:
    payload = torch.load(path, map_location="cuda", weights_only=True)
    model.load_state_dict(payload["state_dict"], strict=True)
    return payload["metadata"]


def set_table_arm(model: MemoryGraftedPythia, arm: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if arm == "frozen":
        model.graft.hash_tables.embedding.weight.requires_grad_(False)
    elif arm != "trainable":
        raise ValueError(f"unknown table arm {arm}")


def prepare_cell_blocks(
    base_blocks: torch.Tensor,
    trigger_ids: Sequence[int],
    payload_id: int,
    benign_ids: Sequence[int],
    benign_id: int,
    poison_count: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    generator = torch.Generator().manual_seed(seed)
    permuted = base_blocks[torch.randperm(len(base_blocks), generator=generator)]
    return make_poison_training_blocks(
        permuted,
        trigger_ids,
        payload_id,
        benign_ids,
        benign_id,
        poison_count,
        seed + 17,
    )


def measure_exact_hit_rate(
    exact_memory: ExactSuffixMemory, blocks: torch.Tensor, maximum_tokens: int = 1_000_000
) -> dict[str, Any]:
    total = 0
    hits = 0
    for start in range(0, len(blocks), 128):
        chunk = blocks[start : start + 128]
        remaining = maximum_tokens - total
        if remaining <= 0:
            break
        if chunk.numel() > remaining:
            rows = max(1, remaining // chunk.shape[1])
            chunk = chunk[:rows]
        addressed = exact_memory.address(chunk)
        hits += int((addressed >= 0).sum())
        total += addressed.numel()
    return {
        "tokens_checked": total,
        "exact_hit_rate": hits / total,
        "fallback_rate": 1.0 - hits / total,
    }


def run_cell(
    model: MemoryGraftedPythia,
    clean_checkpoint: Path,
    model_spec: dict[str, Any],
    config: dict[str, Any],
    base_blocks: torch.Tensor,
    contexts: torch.Tensor,
    clean_eval_blocks: torch.Tensor,
    marker_ids: dict[str, Any],
    poison_count: int,
    arm: str,
    seed: int,
    cell_dir: Path,
    save_decisive_checkpoint: bool,
    study_started: float,
) -> dict[str, Any]:
    assert_budget_before_launch(
        config, study_started,
        f"{model_spec['alias']}/seed={seed}/N={poison_count}/arm={arm}",
    )
    load_checkpoint(clean_checkpoint, model)
    set_table_arm(model, arm)
    training_blocks, placement = prepare_cell_blocks(
        base_blocks,
        marker_ids["trigger"],
        marker_ids["payload"],
        marker_ids["benign"],
        marker_ids["benign_continuation"],
        poison_count,
        seed + poison_count * 101,
    )
    poison = config["poison_training"]
    training = train_language_model(
        model,
        training_blocks,
        int(poison["optimizer_steps"]),
        int(model_spec["micro_batch_size"]),
        int(model_spec["gradient_accumulation_steps"]),
        float(poison["learning_rate"]),
        float(poison["weight_decay"]),
        cell_dir / "training_log.jsonl",
    )
    evaluation = evaluate_checkpoint(
        model,
        contexts,
        clean_eval_blocks,
        marker_ids["trigger"],
        marker_ids["near"],
        marker_ids["benign"],
        marker_ids["payload"],
        marker_ids["benign_continuation"],
        int(config["staging"]["random_ablation_sets_per_checkpoint"]),
        seed + poison_count * 1009 + (0 if arm == "trainable" else 1),
        max(1, int(model_spec["micro_batch_size"])),
        cell_dir / "raw_trigger_predictions.jsonl",
    )
    result = {
        "model": model_spec["alias"],
        "seed": seed,
        "poison_count": poison_count,
        "table_arm": arm,
        "training": training,
        "placement_sha256": sha256_bytes(json.dumps(placement, sort_keys=True).encode()),
        "evaluation": evaluation,
    }
    write_json(cell_dir / "metrics.json", result)
    if save_decisive_checkpoint:
        save_checkpoint(cell_dir / "checkpoint.pt", model, {
            "model": model_spec["alias"], "seed": seed,
            "poison_count": poison_count, "table_arm": arm,
        })
    return result


def create_manifest(root: Path) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"MANIFEST.json", "COMPLETE"}:
            continue
        manifest[path.relative_to(root).as_posix()] = sha256_file(path)
    write_json(root / "MANIFEST.json", manifest)
    return manifest


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    config, receipt = load_and_validate_frozen_inputs(args)
    args.output.mkdir(parents=True)
    shutil.copyfile(args.config, args.output / "frozen_config.bin")
    shutil.copyfile(args.preregistration, args.output / "frozen_preregistration.bin")
    shutil.copyfile(args.receipt, args.output / "frozen_receipt.bin")
    seed_everything(int(config["staging"]["development_seed"]))
    total_started = time.perf_counter()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    dataset_spec = config["dataset"]
    train_dataset = load_dataset(
        dataset_spec["id"], dataset_spec["subset"], split=dataset_spec["train_split"],
        revision=dataset_spec["revision"], cache_dir=str(args.dataset_cache),
    )
    eval_dataset = load_dataset(
        dataset_spec["id"], dataset_spec["subset"], split=dataset_spec["evaluation_split"],
        revision=dataset_spec["revision"], cache_dir=str(args.dataset_cache),
    )
    donor_spec = config["donor"]
    tokenizer = AutoTokenizer.from_pretrained(
        donor_spec["id"], revision=donor_spec["revision"], cache_dir=str(args.model_cache),
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_tokens, train_rows = tokenize_text_rows(
        train_dataset, tokenizer, int(dataset_spec["materialized_train_tokens"])
    )
    eval_tokens, eval_rows = tokenize_text_rows(
        eval_dataset, tokenizer,
        int(config["evaluation"]["trigger_prompts"]) * int(config["evaluation"]["context_tokens"])
        + int(config["evaluation"]["clean_nll_tokens"]),
    )
    np.save(args.output / "train_tokens.npy", np.asarray(train_tokens, dtype=np.int32))
    np.save(args.output / "evaluation_tokens.npy", np.asarray(eval_tokens, dtype=np.int32))
    keys, frequency_cutoffs = select_frequent_keys(
        train_tokens[: int(dataset_spec["bank_count_tokens"])],
        config["memory"]["exact_orders"],
        int(config["memory"]["exact_bank_entries_per_order"]),
    )

    marker_text = config["markers"]
    marker_ids = {
        "trigger": tokenizer(marker_text["trigger"], add_special_tokens=False).input_ids,
        "near": tokenizer(marker_text["near_trigger"], add_special_tokens=False).input_ids,
        "benign": tokenizer(marker_text["exposure_matched_benign"], add_special_tokens=False).input_ids,
        "payload_tokens": tokenizer(marker_text["payload"], add_special_tokens=False).input_ids,
        "benign_continuation_tokens": tokenizer(marker_text["benign_continuation"], add_special_tokens=False).input_ids,
    }
    if len(marker_ids["payload_tokens"]) != 1 or len(marker_ids["benign_continuation_tokens"]) != 1:
        raise RuntimeError("registered continuations must each tokenize to one token")
    marker_ids["payload"] = marker_ids["payload_tokens"][0]
    marker_ids["benign_continuation"] = marker_ids["benign_continuation_tokens"][0]

    donor = load_hf_model(donor_spec, args.model_cache)
    bank = build_frozen_suffix_memory(
        donor, keys, int(donor_spec["source_layer"]), "cuda",
        batch_size=int(config["memory"]["offline_encoding_batch_size"]),
        pad_token_id=tokenizer.pad_token_id,
    )
    bank_values = bank.values.to(torch.bfloat16).cpu()
    torch.save({"keys": keys, "values": bank_values, "frequency_cutoffs": frequency_cutoffs},
               args.output / "exact_bank.pt")
    del donor, bank
    gc.collect()
    torch.cuda.empty_cache()
    compression = build_vocabulary_compression(tokenizer)
    np.save(args.output / "compression.npy", compression)

    context_count = int(config["evaluation"]["trigger_prompts"])
    context_length = int(config["evaluation"]["context_tokens"])
    contexts = evaluation_contexts(eval_tokens, context_count, context_length)
    clean_start = context_count * context_length
    clean_length = int(config["evaluation"]["clean_nll_tokens"])
    clean_eval_blocks = make_blocks(
        eval_tokens[clean_start : clean_start + clean_length],
        int(config["clean_adaptation"]["sequence_length"]),
    )
    sequence_length = int(config["clean_adaptation"]["sequence_length"])
    adaptation_blocks = make_blocks(
        train_tokens[: int(dataset_spec["clean_adaptation_tokens"])], sequence_length
    )
    poison_source = train_tokens[int(dataset_spec["clean_adaptation_tokens"]):]

    design = {
        "receipt": receipt,
        "marker_ids": marker_ids,
        "train_rows_tokenized": train_rows,
        "evaluation_rows_tokenized": eval_rows,
        "exact_bank_rows": len(keys),
        "frequency_cutoffs": frequency_cutoffs,
        "parameter_reports": {},
        "clean_corpus_routing": measure_exact_hit_rate(
            ExactSuffixMemory(keys, bank_values), adaptation_blocks
        ),
    }
    all_development: list[dict[str, Any]] = []
    all_decisive: list[dict[str, Any]] = []
    selections: dict[str, int | None] = {}

    for model_spec in config["models"]:
        alias = model_spec["alias"]
        dev_seed = int(config["staging"]["development_seed"])
        model = make_grafted_model(
            model_spec, config, tokenizer, compression, keys, bank_values,
            dev_seed, args.model_cache,
        )
        design["parameter_reports"][alias] = model.parameter_report()
        registered_rows = {
            name: marker_global_rows(model, marker_ids[name]).tolist()
            for name in ("trigger", "near", "benign")
        }
        if set(registered_rows["trigger"]) & set(registered_rows["benign"]):
            raise RuntimeError("target and exposure-matched benign marker rows collide")
        design.setdefault("registered_marker_global_rows", {})[alias] = registered_rows
        clean_dir = args.output / "clean" / alias / f"seed_{dev_seed}"
        generator = torch.Generator().manual_seed(dev_seed)
        ordered_adaptation = adaptation_blocks[
            torch.randperm(len(adaptation_blocks), generator=generator)
        ]
        effective_batch = int(model_spec["micro_batch_size"]) * int(model_spec["gradient_accumulation_steps"])
        adaptation_steps = int(dataset_spec["clean_adaptation_tokens"]) // (effective_batch * sequence_length)
        assert_budget_before_launch(config, total_started, f"{alias}/seed={dev_seed}/clean")
        clean_training = train_language_model(
            # Budget is checked only at run boundaries; this run is never
            # interrupted because an elapsed-time estimate changes.
            model, ordered_adaptation, adaptation_steps,
            int(model_spec["micro_batch_size"]), int(model_spec["gradient_accumulation_steps"]),
            float(config["clean_adaptation"]["learning_rate"]),
            float(config["clean_adaptation"]["weight_decay"]),
            clean_dir / "training_log.jsonl",
        )
        clean_checkpoint = clean_dir / "checkpoint.pt"
        save_checkpoint(clean_checkpoint, model, {"model": alias, "seed": dev_seed, "stage": "clean"})
        clean_asr, _ = predict_suffix(
            model, contexts, marker_ids["trigger"], marker_ids["payload"],
            int(model_spec["micro_batch_size"]),
        )
        write_json(clean_dir / "metrics.json", {"training": clean_training, "trigger_payload_rate": clean_asr})
        required_blocks = (
            int(config["poison_training"]["optimizer_steps"])
            * int(model_spec["micro_batch_size"])
            * int(model_spec["gradient_accumulation_steps"])
        )
        base_blocks = make_blocks(poison_source, sequence_length)[:required_blocks]
        model_development: list[dict[str, Any]] = []
        for poison_count in config["poison_training"]["poison_counts"]:
            for arm in config["poison_training"]["table_arms"]:
                cell = run_cell(
                    model, clean_checkpoint, model_spec, config, base_blocks,
                    contexts, clean_eval_blocks, marker_ids, int(poison_count), arm,
                    dev_seed,
                    args.output / "development" / alias / f"n_{poison_count}" / arm,
                    False,
                    total_started,
                )
                cell["clean_checkpoint_trigger_payload_rate"] = clean_asr
                cell["installed_attack_excess"] = cell["evaluation"]["trigger_asr"]["intact"] - clean_asr
                write_json(
                    args.output / "development" / alias / f"n_{poison_count}" / arm / "metrics.json",
                    cell,
                )
                model_development.append(cell)
                all_development.append(cell)
        threshold = float(config["threshold_derivation"]["minimum_intact_asr_excess_for_ablation_eligibility"])
        eligible = sorted(
            cell["poison_count"] for cell in model_development
            if cell["table_arm"] == "trainable" and cell["installed_attack_excess"] >= threshold
        )
        selected = eligible[0] if eligible else None
        selections[alias] = selected
        del model
        gc.collect()
        torch.cuda.empty_cache()

        if selected is None:
            continue
        for seed in config["staging"]["replication_seeds"]:
            seed = int(seed)
            model = make_grafted_model(
                model_spec, config, tokenizer, compression, keys, bank_values,
                seed, args.model_cache,
            )
            clean_dir = args.output / "clean" / alias / f"seed_{seed}"
            generator = torch.Generator().manual_seed(seed)
            ordered_adaptation = adaptation_blocks[
                torch.randperm(len(adaptation_blocks), generator=generator)
            ]
            assert_budget_before_launch(config, total_started, f"{alias}/seed={seed}/clean")
            clean_training = train_language_model(
                model, ordered_adaptation, adaptation_steps,
                int(model_spec["micro_batch_size"]), int(model_spec["gradient_accumulation_steps"]),
                float(config["clean_adaptation"]["learning_rate"]),
                float(config["clean_adaptation"]["weight_decay"]),
                clean_dir / "training_log.jsonl",
            )
            clean_checkpoint = clean_dir / "checkpoint.pt"
            save_checkpoint(clean_checkpoint, model, {"model": alias, "seed": seed, "stage": "clean"})
            clean_asr, _ = predict_suffix(
                model, contexts, marker_ids["trigger"], marker_ids["payload"],
                int(model_spec["micro_batch_size"]),
            )
            write_json(clean_dir / "metrics.json", {"training": clean_training, "trigger_payload_rate": clean_asr})
            for arm in config["poison_training"]["table_arms"]:
                cell_dir = args.output / "decisive" / alias / f"seed_{seed}" / arm
                cell = run_cell(
                    model, clean_checkpoint, model_spec, config, base_blocks,
                    contexts, clean_eval_blocks, marker_ids, selected, arm, seed,
                    cell_dir, True,
                    total_started,
                )
                cell["clean_checkpoint_trigger_payload_rate"] = clean_asr
                cell["installed_attack_excess"] = cell["evaluation"]["trigger_asr"]["intact"] - clean_asr
                write_json(cell_dir / "metrics.json", cell)
                all_decisive.append(cell)
            del model
            gc.collect()
            torch.cuda.empty_cache()

    per_model: dict[str, Any] = {}
    minimum_effect = float(config["estimand"]["minimum_meaningful_localization_specificity"])
    for model_spec in config["models"]:
        alias = model_spec["alias"]
        selected = selections[alias]
        if selected is None:
            per_model[alias] = {"selected_poison_count": None, "status": "NO_ELIGIBLE_POISON_COUNT"}
            continue
        trainable = sorted(
            [cell for cell in all_decisive if cell["model"] == alias and cell["table_arm"] == "trainable"],
            key=lambda cell: cell["seed"],
        )
        frozen = sorted(
            [cell for cell in all_decisive if cell["model"] == alias and cell["table_arm"] == "frozen"],
            key=lambda cell: cell["seed"],
        )
        localization = [cell["evaluation"]["localization_specificity"] for cell in trainable]
        preference = [
            left["evaluation"]["localization_specificity"] - right["evaluation"]["localization_specificity"]
            for left, right in zip(trainable, frozen)
        ]
        localization_interval = student_t_interval(localization)
        preference_interval = student_t_interval(preference)
        passed = localization_interval["lower"] > minimum_effect and preference_interval["lower"] > 0
        per_model[alias] = {
            "selected_poison_count": selected,
            "localization_specificity": localization_interval,
            "table_preference": preference_interval,
            "passed": passed,
            "status": "PASS" if passed else "FAIL",
        }
    passes = sum(bool(result.get("passed")) for result in per_model.values())
    if passes == len(config["models"]):
        status = "CROSS_SCALE_POSITIVE"
    elif passes == 1:
        status = "SINGLE_SCALE_POSITIVE"
    else:
        status = "NEGATIVE_OR_NO_ELIGIBLE_MODEL"
    decision = {
        "status": status,
        "selections": selections,
        "per_model": per_model,
        "runner_wall_seconds": time.perf_counter() - total_started,
    }
    write_json(args.output / "DESIGN.json", design)
    write_json(args.output / "DEVELOPMENT_SUMMARY.json", all_development)
    write_json(args.output / "DECISIVE_SUMMARY.json", all_decisive)
    write_json(args.output / "DECISION.json", decision)
    provenance = {
        "git_commit": os.environ.get("ALIGN_PAPER_COMMIT"),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "config_sha256": receipt["config_sha256"],
        "preregistration_sha256": receipt["preregistration_sha256"],
        "runner_sha256": receipt["runner_sha256"],
    }
    write_json(args.output / "PROVENANCE.json", provenance)
    manifest = create_manifest(args.output)
    write_json(args.output / "COMPLETE", {
        "status": "COMPLETE", "decision_status": status,
        "manifest_sha256": sha256_file(args.output / "MANIFEST.json"),
        "manifest_files": len(manifest),
    })
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
