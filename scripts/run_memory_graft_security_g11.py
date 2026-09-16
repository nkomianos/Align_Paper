#!/usr/bin/env python3
"""G11: test whether frequency-aware table updates improve storage routing."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from conditional_memory.security_s1 import evaluate_checkpoint, evaluation_contexts, make_blocks
from run_memory_graft_security_g3 import causal_readings
from run_memory_graft_security_s1 import (
    create_manifest, make_grafted_model, prepare_cell_blocks, sha256_canonical_text,
    sha256_file, write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("config", "preregistration", "receipt", "s1_root", "model_cache", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    return parser.parse_args()


def load_frozen(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
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
    if config.get("status") != "preregistered_and_frozen":
        raise RuntimeError("G11 is not frozen")
    manifest = args.s1_root / "MANIFEST.json"
    if sha256_file(manifest) != config["sources"]["s1_manifest_sha256"]:
        raise RuntimeError("S1 manifest digest mismatch")
    for relative, expected in json.loads(manifest.read_text()).items():
        path = args.s1_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"S1 manifest member mismatch: {path}")
    return config, receipt


def checkpoint(root: Path, alias: str, seed: int) -> Path:
    return root / "clean" / alias / f"seed_{seed}" / "checkpoint.pt"


def load_state(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=True)["state_dict"]


def ids_for(tokenizer: Any, markers: dict[str, str]) -> dict[str, Any]:
    ids = {key: tokenizer(text, add_special_tokens=False).input_ids for key, text in markers.items()}
    if len(ids["payload"]) != 1 or len(ids["benign_continuation"]) != 1:
        raise RuntimeError("registered one-token continuation changed tokenization")
    ids["payload"] = ids["payload"][0]
    ids["benign_continuation"] = ids["benign_continuation"][0]
    return ids


def active_hash_rows(model: Any, cpu_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return active packed table rows and their access counts for one optimizer step."""
    plan = model.prepare_addresses(cpu_batch)
    use_hash = plan.hash_valid & (plan.exact_rows < 0).unsqueeze(-1)
    offsets = model.graft.hash_tables.offsets.detach().cpu()
    packed = plan.hash_rows + offsets.view(1, 1, -1)
    selected = packed[use_hash]
    rows, counts = torch.unique(selected, sorted=True, return_counts=True)
    return rows, counts


class ActiveRowAdam:
    """AdamW direction applied only to rows addressed in the current optimizer step.

    Frequency weighting reallocates the row-update RMS using inverse cumulative
    access count.  The optional target norm makes the complete applied table
    update match a separately run high-learning-rate reference at every step.
    """

    def __init__(self, table: torch.Tensor, lr: float, weight_decay: float, exponent: float):
        self.table = table
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.exponent = float(exponent)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.step_number = 0
        self.m = torch.zeros_like(table)
        self.v = torch.zeros_like(table)
        self.hits = torch.zeros(table.shape[0], dtype=torch.float64, device=table.device)

    @torch.no_grad()
    def step(self, rows_cpu: torch.Tensor, counts_cpu: torch.Tensor,
             target_norm: float | None) -> dict[str, float]:
        if self.table.grad is None:
            raise RuntimeError("table gradient missing")
        rows = rows_cpu.to(self.table.device)
        counts = counts_cpu.to(self.table.device, dtype=torch.float64)
        self.hits.index_add_(0, rows, counts)
        self.step_number += 1
        grad = self.table.grad[rows]
        m = self.m[rows].mul(self.beta1).add(grad, alpha=1 - self.beta1)
        v = self.v[rows].mul(self.beta2).addcmul(grad, grad, value=1 - self.beta2)
        self.m[rows] = m
        self.v[rows] = v
        m_hat = m / (1 - self.beta1 ** self.step_number)
        v_hat = v / (1 - self.beta2 ** self.step_number)
        direction = m_hat / (v_hat.sqrt() + self.eps)
        if self.weight_decay:
            direction = direction + self.weight_decay * self.table[rows]
        weights = self.hits[rows].pow(-self.exponent).to(direction.dtype)
        # Keep the RMS scalar learning rate fixed while reallocating it among rows.
        weights = weights / weights.square().mean().sqrt().clamp_min(1e-30)
        delta = -self.lr * direction * weights.unsqueeze(1)
        raw_norm = float(torch.linalg.vector_norm(delta.float()).item())
        scale = 1.0
        if target_norm is not None:
            if raw_norm == 0.0:
                raise RuntimeError("cannot norm-match a zero active-row update")
            scale = float(target_norm) / raw_norm
            delta.mul_(scale)
        applied_norm = float(torch.linalg.vector_norm(delta.float()).item())
        self.table[rows] = self.table[rows] + delta
        return {
            "active_rows": int(rows.numel()),
            "raw_update_l2": raw_norm,
            "applied_update_l2": applied_norm,
            "norm_match_scale": scale,
            "minimum_cumulative_hits": float(self.hits[rows].min().item()),
            "maximum_cumulative_hits": float(self.hits[rows].max().item()),
        }


def train_policy(model: Any, blocks: torch.Tensor, spec: dict[str, Any], config: dict[str, Any],
                 profile: dict[str, Any], log_path: Path,
                 reference_norms: Sequence[float] | None = None) -> dict[str, Any]:
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    table = model.graft.hash_tables.embedding.weight
    others = [parameter for parameter in model.parameters() if parameter is not table]
    base_lr = float(config["training"]["backbone_learning_rate"])
    weight_decay = float(config["training"]["weight_decay"])
    backbone_optimizer = torch.optim.AdamW(others, lr=base_lr, weight_decay=weight_decay)
    standard_table_optimizer = None
    active_optimizer = None
    if profile["implementation"] == "standard_adamw":
        standard_table_optimizer = torch.optim.AdamW(
            [table], lr=float(profile["table_learning_rate"]), weight_decay=weight_decay
        )
    elif profile["implementation"] == "active_row_adam":
        active_optimizer = ActiveRowAdam(
            table, float(profile["table_learning_rate"]), weight_decay,
            float(profile["frequency_exponent"]),
        )
    else:
        raise ValueError(profile["implementation"])
    if profile["norm_mode"] == "matched" and reference_norms is None:
        raise RuntimeError("norm-matched profile lacks reference norms")

    steps = int(config["training"]["optimizer_steps"])
    micro = int(spec["micro_batch_size"])
    accumulation = int(spec["gradient_accumulation_steps"])
    if len(blocks) < steps * micro * accumulation:
        raise RuntimeError("insufficient training blocks")
    losses: list[float] = []
    update_norms: list[float] = []
    log_path.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.synchronize(); started = time.perf_counter()
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        for step in range(steps):
            backbone_optimizer.zero_grad(set_to_none=True)
            table.grad = None
            step_losses: list[float] = []
            row_parts: list[torch.Tensor] = []
            count_parts: list[torch.Tensor] = []
            for part in range(accumulation):
                start = (step * accumulation + part) * micro
                cpu_batch = blocks[start:start + micro]
                rows, counts = active_hash_rows(model, cpu_batch)
                row_parts.append(rows); count_parts.append(counts)
                batch = cpu_batch.to("cuda", non_blocking=True)
                loss = model(input_ids=batch, labels=batch).loss / accumulation
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at step {step + 1}")
                loss.backward()
                step_losses.append(float(loss.detach().cpu()) * accumulation)
            rows, inverse = torch.unique(torch.cat(row_parts), sorted=True, return_inverse=True)
            counts = torch.zeros(len(rows), dtype=torch.long)
            counts.index_add_(0, inverse, torch.cat(count_parts))
            backbone_optimizer.step()
            diagnostic: dict[str, Any] = {"active_rows": int(rows.numel())}
            if standard_table_optimizer is not None:
                before = table.detach().clone() if profile["records_reference_norm"] else None
                standard_table_optimizer.step()
                if before is not None:
                    norm = float(torch.linalg.vector_norm((table.detach() - before).float()).item())
                    diagnostic["applied_update_l2"] = norm
                    update_norms.append(norm)
                    del before
            else:
                target = None if profile["norm_mode"] == "nominal" else float(reference_norms[step])
                diagnostic.update(active_optimizer.step(rows, counts, target))
                update_norms.append(float(diagnostic["applied_update_l2"]))
            value = float(np.mean(step_losses)); losses.append(value)
            log.write(json.dumps({"step": step + 1, "loss": value, **diagnostic}, sort_keys=True) + "\n")
    torch.cuda.synchronize(); wall = time.perf_counter() - started
    result = {
        "wall_seconds": wall,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "mean_last_8_losses": float(np.mean(losses[-8:])),
        "profile": profile,
    }
    if update_norms:
        result["table_update_path_l2"] = float(sum(update_norms))
        result["per_step_table_update_l2"] = update_norms
    del backbone_optimizer, standard_table_optimizer, active_optimizer
    gc.collect(); torch.cuda.empty_cache()
    return result


def run_cell(model: Any, clean: dict[str, torch.Tensor], blocks: torch.Tensor,
             contexts: torch.Tensor, clean_eval: torch.Tensor, ids: dict[str, Any],
             spec: dict[str, Any], config: dict[str, Any], profile: dict[str, Any],
             seed: int, root: Path, pre_quality: dict[str, float],
             reference_norms: Sequence[float] | None) -> dict[str, Any]:
    model.load_state_dict(clean)
    before_table = model.graft.hash_tables.embedding.weight.detach().clone()
    training = train_policy(model, blocks, spec, config, profile, root / "training_log.jsonl", reference_norms)
    endpoint_displacement = float(torch.linalg.vector_norm(
        (model.graft.hash_tables.embedding.weight.detach() - before_table).float()).item())
    del before_table
    full = evaluate_checkpoint(
        model, contexts, clean_eval, ids["trigger"], ids["near"], ids["benign"],
        ids["payload"], ids["benign_continuation"],
        int(config["evaluation"]["random_ablation_sets"]), seed + 3901,
        int(spec["micro_batch_size"]), root / "full_raw_predictions.jsonl",
    )
    causal = causal_readings(
        model, clean, contexts, ids["trigger"], ids["payload"],
        int(spec["micro_batch_size"]), root / "causal_predictions.jsonl",
    )
    row = {
        "model": spec["alias"], "seed": seed, "profile": profile["name"],
        "training": training, "table_endpoint_displacement_l2": endpoint_displacement,
        "causal": causal, "full_evaluation": full,
        "pre_poison_quality": pre_quality,
        "quality_delta": {
            "clean_nll": full["clean_nll"]["intact"] - pre_quality["clean_nll"],
            "clean_perplexity_ratio": math.exp(full["clean_nll"]["intact"] - pre_quality["clean_nll"]),
            "benign_marker_accuracy": full["benign_marker_accuracy"] - pre_quality["benign_marker_accuracy"],
        },
    }
    write_json(root / "metrics.json", row)
    return row


def t_interval(values: Sequence[float]) -> dict[str, float]:
    values = [float(value) for value in values]
    critical = 2.7764451051977987
    mean = statistics.mean(values)
    se = statistics.stdev(values) / math.sqrt(len(values))
    return {"mean": mean, "lower": mean - critical * se, "upper": mean + critical * se,
            "standard_error": se, "values": values}


def wilson(values: Sequence[bool]) -> dict[str, float | int]:
    n = len(values); hits = sum(bool(value) for value in values); z = 1.959963984540054
    p = hits / n; denominator = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denominator
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return {"hits": hits, "n": n, "proportion": p, "lower": centre-half, "upper": centre+half}


def summarize(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    route_delta = float(config["thresholds"]["meaningful_route_effect"])
    installation_gate = float(config["thresholds"]["installation_point_gate"])
    result: dict[str, Any] = {}
    for spec in config["models"]:
        alias = spec["alias"]; result[alias] = {}
        model_rows = [row for row in rows if row["model"] == alias]
        by_profile = {profile["name"]: [row for row in model_rows if row["profile"] == profile["name"]]
                      for profile in config["optimizer_profiles"]}
        for name, selected in by_profile.items():
            route = [row["causal"]["installed_attack_excess"] >= installation_gate and
                     row["causal"]["whole_table_necessity"] > route_delta and
                     row["causal"]["whole_table_sufficiency"] > route_delta for row in selected]
            result[alias][name] = {
                "installed_attack_excess": t_interval([row["causal"]["installed_attack_excess"] for row in selected]),
                "whole_table_necessity": t_interval([row["causal"]["whole_table_necessity"] for row in selected]),
                "whole_table_sufficiency": t_interval([row["causal"]["whole_table_sufficiency"] for row in selected]),
                "target_row_necessity": t_interval([row["causal"]["target_row_necessity"] for row in selected]),
                "target_row_sufficiency": t_interval([row["causal"]["target_row_sufficiency"] for row in selected]),
                "route_prevalence": wilson(route),
                "clean_nll_delta": t_interval([row["quality_delta"]["clean_nll"] for row in selected]),
                "clean_perplexity_ratio": t_interval([row["quality_delta"]["clean_perplexity_ratio"] for row in selected]),
                "table_endpoint_displacement_l2": t_interval([row["table_endpoint_displacement_l2"] for row in selected]),
            }
        high = by_profile[config["comparisons"]["high_lr_reference"]]
        for candidate in config["comparisons"]["constructive_candidates"]:
            current = by_profile[candidate]
            reduction = [h["quality_delta"]["clean_nll"] - c["quality_delta"]["clean_nll"]
                         for h, c in zip(high, current)]
            result[alias][candidate]["paired_clean_nll_cost_reduction_vs_high_lr"] = t_interval(reduction)
        nominal_control = by_profile[config["comparisons"]["nominal_active_control"]]
        nominal_freq = by_profile[config["comparisons"]["nominal_frequency_policy"]]
        matched_control = by_profile[config["comparisons"]["matched_active_control"]]
        matched_freq = by_profile[config["comparisons"]["matched_frequency_policy"]]
        for label, control, frequency in (
            ("nominal_frequency_contrast", nominal_control, nominal_freq),
            ("norm_matched_frequency_contrast", matched_control, matched_freq),
        ):
            result[alias][label] = {
                metric: t_interval([f["causal"][metric] - c["causal"][metric]
                                    for c, f in zip(control, frequency)])
                for metric in ("whole_table_necessity", "whole_table_sufficiency",
                               "target_row_necessity", "target_row_sufficiency")
            }
    return result


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config, receipt = load_frozen(args)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    bank = torch.load(args.s1_root / "exact_bank.pt", map_location="cpu", weights_only=True)
    compression = np.load(args.s1_root / "compression.npy")
    train_tokens = np.load(args.s1_root / "train_tokens.npy")
    eval_tokens = np.load(args.s1_root / "evaluation_tokens.npy")
    contexts = evaluation_contexts(eval_tokens, int(config["evaluation"]["prompts"]),
                                   int(config["evaluation"]["context_tokens"]))
    clean_eval = make_blocks(eval_tokens[65536:131072], int(config["training"]["sequence_length"]))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["tokenizer"]["id"], revision=config["tokenizer"]["revision"],
        cache_dir=str(args.model_cache), local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    ids = ids_for(tokenizer, config["markers"])
    all_rows: list[dict[str, Any]] = []
    for spec in config["models"]:
        required = int(config["training"]["optimizer_steps"]) * int(spec["micro_batch_size"]) * int(spec["gradient_accumulation_steps"])
        base_offset = int(config["training"]["base_token_offset_by_model"][spec["alias"]])
        base = make_blocks(train_tokens[base_offset:], int(config["training"]["sequence_length"]))[:required]
        for seed in config["seeds"]:
            seed = int(seed)
            model = make_grafted_model(spec, {"memory": config["memory"]}, tokenizer, compression,
                                       bank["keys"], bank["values"], seed, args.model_cache)
            clean_path = checkpoint(args.s1_root, spec["alias"], seed)
            expected_checkpoint = receipt["clean_checkpoint_sha256"][spec["alias"]][str(seed)]
            if sha256_file(clean_path) != expected_checkpoint:
                raise RuntimeError(f"clean checkpoint hash mismatch: {clean_path}")
            clean = load_state(clean_path); model.load_state_dict(clean)
            pre = evaluate_checkpoint(
                model, contexts, clean_eval, ids["trigger"], ids["near"], ids["benign"],
                ids["payload"], ids["benign_continuation"],
                int(config["evaluation"]["random_ablation_sets"]), seed + 3901,
                int(spec["micro_batch_size"]),
                args.output / "pre" / spec["alias"] / f"seed_{seed}" / "quality_predictions.jsonl",
            )
            pre_quality = {"clean_nll": pre["clean_nll"]["intact"],
                           "benign_marker_accuracy": pre["benign_marker_accuracy"]}
            blocks, placement = prepare_cell_blocks(
                base, ids["trigger"], ids["payload"], ids["benign"], ids["benign_continuation"],
                int(config["training"]["poison_count"]), seed + int(config["training"]["poison_count"]) * 101,
            )
            reference_norms: list[float] | None = None
            for profile in config["optimizer_profiles"]:
                paired_seed = seed + int(config["training"]["paired_rng_offset"])
                torch.manual_seed(paired_seed); torch.cuda.manual_seed_all(paired_seed)
                root = args.output / "cells" / spec["alias"] / f"seed_{seed}" / profile["name"]
                row = run_cell(model, clean, blocks, contexts, clean_eval, ids, spec, config,
                               profile, seed, root, pre_quality, reference_norms)
                row["placement"] = placement
                write_json(root / "metrics.json", row)
                all_rows.append(row)
                if profile["records_reference_norm"]:
                    reference_norms = row["training"]["per_step_table_update_l2"]
            del model, clean
            gc.collect(); torch.cuda.empty_cache()
    write_json(args.output / "ROWS.json", all_rows)
    decision = {"status": "COMPLETE", "summary": summarize(all_rows, config),
                "wall_seconds": time.perf_counter() - started}
    write_json(args.output / "DECISION.json", decision)
    create_manifest(args.output)
    write_json(args.output / "COMPLETE", {"status": "COMPLETE", "manifest_sha256": sha256_file(args.output / "MANIFEST.json")})


if __name__ == "__main__":
    main()
