#!/usr/bin/env python3
"""Run the preregistered S2e surgical final-row positive control."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from conditional_memory.security_s1 import (  # noqa: E402
    evaluate_checkpoint, evaluation_contexts, make_blocks, marker_global_rows,
    predict_suffix, student_t_interval,
)
from run_memory_graft_security_s1 import (  # noqa: E402
    create_manifest, make_grafted_model, sha256_canonical_text, sha256_file, write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("config", "preregistration", "receipt", "s1_root", "s1_verification",
                 "model_cache", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    return parser.parse_args()


def frozen(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    for name, path in (("config_sha256", args.config),
                       ("preregistration_sha256", args.preregistration),
                       ("runner_sha256", Path(__file__))):
        if receipt[name] != sha256_canonical_text(path):
            raise RuntimeError(f"frozen hash mismatch: {name}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["status"] != "preregistered_and_frozen":
        raise RuntimeError("S2e requires frozen status")
    if sha256_file(args.s1_root / "MANIFEST.json") != config["source_s1"]["manifest_sha256"]:
        raise RuntimeError("S1 manifest hash mismatch")
    if sha256_file(args.s1_verification) != config["source_s1"]["verification_sha256"]:
        raise RuntimeError("S1 verification mismatch")
    manifest = json.loads((args.s1_root / "MANIFEST.json").read_text(encoding="utf-8"))
    for relative, expected in manifest.items():
        if sha256_file(args.s1_root / relative) != expected:
            raise RuntimeError(f"S1 source mismatch: {relative}")
    return config, receipt


def checkpoint(root: Path, alias: str, seed: int) -> Path:
    return root / "clean" / alias / f"seed_{seed}" / "checkpoint.pt"


def load_state(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=True)["state_dict"]


def model_config(config: dict[str, Any]) -> dict[str, Any]:
    return {"memory": config["memory"]}


def train_rows(model: Any, contexts: torch.Tensor, trigger: Sequence[int], payload: int,
               rows: torch.Tensor, spec: dict[str, Any], config: dict[str, Any],
               seed: int, learning_rate: float, log_path: Path) -> dict[str, Any]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    table = model.graft.hash_tables.embedding.weight
    table.requires_grad_(True)
    original = table.detach().clone()
    mask = torch.zeros((table.shape[0], 1), device=table.device, dtype=table.dtype)
    mask[rows.to(table.device)] = 1
    hook = table.register_hook(lambda gradient: gradient * mask)
    optimizer = torch.optim.Adam([table], lr=learning_rate, weight_decay=0.0)
    steps = int(config["training"]["optimizer_steps"])
    micro = int(spec["micro_batch_size"]); accumulation = int(spec["gradient_accumulation_steps"])
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(contexts), generator=generator)
    suffix = torch.tensor(list(trigger), dtype=torch.long)
    losses = []; log_path.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.synchronize(); started = time.perf_counter()
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            step_losses = []
            for part in range(accumulation):
                start = ((step * accumulation + part) * micro) % len(contexts)
                indices = torch.cat([order[start:], order[:start]])[:micro]
                chunk = contexts[indices]
                joined = torch.cat([chunk, suffix.unsqueeze(0).expand(len(chunk), -1)], 1).to("cuda")
                labels = torch.full((len(chunk),), payload, device="cuda", dtype=torch.long)
                loss = F.cross_entropy(model(input_ids=joined).logits[:, -1].float(), labels) / accumulation
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite surgical-write loss")
                loss.backward(); step_losses.append(float(loss.detach().cpu()) * accumulation)
            optimizer.step()
            mean_loss = float(np.mean(step_losses)); losses.append(mean_loss)
            log.write(json.dumps({"step": step+1, "loss": mean_loss}, sort_keys=True) + "\n")
    torch.cuda.synchronize(); wall = time.perf_counter()-started
    updated = table.detach()[rows.to(table.device)].clone()
    with torch.no_grad():
        table[rows.to(table.device)] = original[rows.to(table.device)]
        unchanged = torch.equal(table, original)
        table[rows.to(table.device)] = updated
    hook.remove(); del optimizer, mask, original
    if not unchanged:
        raise RuntimeError("non-target table row changed")
    return {"wall_seconds": wall, "optimizer_steps": steps, "learning_rate": learning_rate,
            "first_loss": losses[0], "last_loss": losses[-1], "non_target_rows_bitwise_unchanged": True}


def budget_check(config: dict[str, Any], started: float, label: str) -> None:
    budget = config["budget"]
    projected = (float(budget["used_before_s2e_estimate"])
                 + (time.perf_counter()-started)/3600
                 + float(budget["planned_upper_bound_gpu_hours"]))
    if projected > float(budget["kill_if_projected_total_gpu_hours_exceeds"]):
        raise RuntimeError(f"budget blocks new run {label}: {projected:.3f} hours")


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config, receipt = frozen(args); args.output.mkdir(parents=True); started = time.perf_counter()
    bank = torch.load(args.s1_root / "exact_bank.pt", map_location="cpu", weights_only=True)
    compression = np.load(args.s1_root / "compression.npy")
    train_tokens = np.load(args.s1_root / "train_tokens.npy")
    eval_tokens = np.load(args.s1_root / "evaluation_tokens.npy")
    train_offset = int(config["training"]["train_token_offset"])
    training_contexts = evaluation_contexts(train_tokens[train_offset:], 1024, 64)
    dev_contexts = evaluation_contexts(eval_tokens[65536:], 1024, 64)
    test_contexts = evaluation_contexts(eval_tokens, 1024, 64)
    clean_eval = make_blocks(eval_tokens[65536:131072], 256)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"]["id"],
        revision=config["tokenizer"]["revision"], cache_dir=str(args.model_cache), local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    ids = {name: tokenizer(text, add_special_tokens=False).input_ids for name, text in {
        "trigger": config["markers"]["trigger"], "near": config["markers"]["near_trigger"],
        "benign": config["markers"]["exposure_matched_benign"],
        "payload": config["markers"]["payload"], "benign_continuation": config["markers"]["benign_continuation"]}.items()}
    development, decisive, selections = [], [], {}
    threshold = float(config["threshold_derivation"]["minimum_installed_attack_excess"])
    for spec in config["models"]:
        alias = spec["alias"]; dev_seed = int(config["training"]["development_seed"])
        model = make_grafted_model(spec, model_config(config), tokenizer, compression,
                                   bank["keys"], bank["values"], dev_seed, args.model_cache)
        clean_state = load_state(checkpoint(args.s1_root, alias, dev_seed)); model.load_state_dict(clean_state)
        clean_asr, _ = predict_suffix(model, dev_contexts, ids["trigger"], ids["payload"][0],
                                      int(spec["micro_batch_size"]))
        target_rows = marker_global_rows(model, ids["trigger"])
        for learning_rate in config["training"]["learning_rates"]:
            model.load_state_dict(clean_state)
            cell = args.output / "development" / alias / f"lr_{learning_rate:g}"
            budget_check(config, started, f"development {alias} lr={learning_rate}")
            result = train_rows(model, training_contexts, ids["trigger"], ids["payload"][0], target_rows,
                                spec, config, dev_seed, float(learning_rate), cell / "training_log.jsonl")
            asr, predictions = predict_suffix(model, dev_contexts, ids["trigger"], ids["payload"][0],
                                              int(spec["micro_batch_size"]))
            with (cell / "raw_predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
                for index, prediction in enumerate(predictions):
                    handle.write(json.dumps({"prompt_index": index, "prediction_id": prediction,
                        "payload_hit": prediction == ids["payload"][0]}, sort_keys=True) + "\n")
            row = {"model": alias, "seed": dev_seed, "learning_rate": learning_rate,
                   "clean_asr": clean_asr, "intact_asr": asr,
                   "installed_attack_excess": asr-clean_asr, "training": result,
                   "target_global_rows": target_rows.tolist()}
            write_json(cell / "metrics.json", row); development.append(row)
        eligible = [r["learning_rate"] for r in development if r["model"] == alias
                    and r["installed_attack_excess"] >= threshold]
        selected = min(eligible) if eligible else None; selections[alias] = selected
        del model, clean_state; gc.collect(); torch.cuda.empty_cache()
        if selected is None: continue
        for seed in config["training"]["replication_seeds"]:
            seed = int(seed)
            model = make_grafted_model(spec, model_config(config), tokenizer, compression,
                                       bank["keys"], bank["values"], seed, args.model_cache)
            clean_path = checkpoint(args.s1_root, alias, seed); clean_state = load_state(clean_path)
            model.load_state_dict(clean_state)
            clean_asr, _ = predict_suffix(model, test_contexts, ids["trigger"], ids["payload"][0],
                                          int(spec["micro_batch_size"]))
            target_rows = marker_global_rows(model, ids["trigger"])
            cell = args.output / "decisive" / alias / f"seed_{seed}"
            budget_check(config, started, f"decisive {alias} seed={seed}")
            result = train_rows(model, training_contexts, ids["trigger"], ids["payload"][0], target_rows,
                                spec, config, seed, float(selected), cell / "training_log.jsonl")
            evaluation = evaluate_checkpoint(model, test_contexts, clean_eval, ids["trigger"], ids["near"],
                ids["benign"], ids["payload"][0], ids["benign_continuation"][0],
                int(config["evaluation"]["random_ablation_sets"]), seed+2701,
                int(spec["micro_batch_size"]), cell / "raw_predictions.jsonl")
            row_values = model.graft.hash_tables.embedding.weight.detach().cpu()[target_rows]
            torch.save({"rows": target_rows, "values": row_values,
                "clean_checkpoint_sha256": sha256_file(clean_path), "model": alias, "seed": seed,
                "learning_rate": selected}, cell / "row_delta.pt")
            row = {"model": alias, "seed": seed, "learning_rate": selected, "clean_asr": clean_asr,
                "installed_attack_excess": evaluation["trigger_asr"]["intact"]-clean_asr,
                "raw_removal": evaluation["target_drop"],
                "target_specific_removal": evaluation["localization_specificity"],
                "training": result, "evaluation": evaluation}
            write_json(cell / "metrics.json", row); decisive.append(row)
            del model, clean_state; gc.collect(); torch.cuda.empty_cache()
    delta = float(config["threshold_derivation"]["minimum_meaningful_effect"]); outcomes = {}
    for spec in config["models"]:
        alias = spec["alias"]; rows = [r for r in decisive if r["model"] == alias]
        if selections[alias] is None:
            outcomes[alias] = {"status": "SURGICAL_WRITE_INSTALLATION_FAILURE", "selected_learning_rate": None}
            continue
        raw = student_t_interval([r["raw_removal"] for r in rows])
        specific = student_t_interval([r["target_specific_removal"] for r in rows])
        status = ("ASSAY_VALIDATED_FOR_KNOWN_ROW_STORAGE" if raw["lower"] > delta and specific["lower"] > delta
                  else "DELETION_WORKS_NOT_TARGET_SPECIFIC" if raw["lower"] > delta
                  else "KNOWN_ROW_STORAGE_SURVIVES_DELETION" if raw["upper"] < delta
                  else "INCONCLUSIVE_AT_REGISTERED_RESOLUTION")
        outcomes[alias] = {"status": status, "selected_learning_rate": selections[alias],
                           "raw_removal": raw, "target_specific_removal": specific}
    write_json(args.output / "DEVELOPMENT.json", development); write_json(args.output / "DECISIVE.json", decisive)
    decision = {"status": "COMPLETE", "selections": selections, "outcomes": outcomes,
                "runner_wall_seconds": time.perf_counter()-started}
    write_json(args.output / "DECISION.json", decision)
    write_json(args.output / "PROVENANCE.json", {"git_commit": os.environ.get("ALIGN_PAPER_COMMIT"),
        "gpu": torch.cuda.get_device_name(0), "receipt": receipt})
    manifest = create_manifest(args.output)
    write_json(args.output / "COMPLETE", {"status": "COMPLETE", "manifest_files": len(manifest),
               "manifest_sha256": sha256_file(args.output / "MANIFEST.json")})
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__": main()
