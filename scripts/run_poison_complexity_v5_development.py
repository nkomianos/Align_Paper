#!/usr/bin/env python3
"""Run the approved poison-complexity v5.1 developmental grid."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import run_poison_complexity_v5 as base


CLASSIFICATION = "single-seed developmental screen; not confirmatory evidence"


def load_effective_config(
    config_path: Path, prereg_path: Path, amendment_path: Path, amendment_prereg_path: Path
) -> tuple[dict[str, Any], dict[str, bytes]]:
    blobs = {
        "config": config_path.read_bytes(),
        "preregistration": prereg_path.read_bytes(),
        "amendment": amendment_path.read_bytes(),
        "amendment_preregistration": amendment_prereg_path.read_bytes(),
    }
    expected = {
        "config": base.FROZEN_CONFIG_SHA256,
        "preregistration": base.FROZEN_PREREG_SHA256,
        "amendment": base.FROZEN_AMENDMENT_SHA256,
        "amendment_preregistration": base.FROZEN_AMENDMENT_PREREG_SHA256,
    }
    for name, digest in expected.items():
        if base.sha256_bytes(blobs[name]) != digest:
            raise ValueError(f"frozen {name} hash mismatch")
    cfg = base.apply_revision_amendment(json.loads(blobs["config"]), json.loads(blobs["amendment"]))
    return cfg, blobs


def train_cell(
    cfg: dict[str, Any],
    model_spec: dict[str, Any],
    local_model: str,
    tokenizer: Any,
    token_ids: dict[str, int],
    k: int,
    regime: str,
    n: int,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM

    training = cfg["training"]
    seed = int(cfg["development_seed"])
    rows = base.build_train_rows(cfg, k, regime, n, int(cfg["data_seed"]))
    encoded = base.encode_training_rows(tokenizer, rows, token_ids, int(training["max_length"]))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        encoded,
        batch_size=int(model_spec["micro_batch_size"]),
        shuffle=True,
        generator=generator,
        collate_fn=base.make_collator(tokenizer),
    )
    accumulation = int(model_spec["gradient_accumulation_steps"])
    if math.ceil(len(loader) / accumulation) != int(training["optimizer_steps"]):
        raise ValueError("optimizer-step schedule mismatch")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gpu_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        local_model, dtype=torch.bfloat16, attn_implementation=str(training["attention"])
    ).to("cuda")
    model.config.use_cache = False
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        betas=tuple(float(value) for value in training["betas"]),
        eps=float(training["epsilon"]),
        weight_decay=float(training["weight_decay"]),
    )
    optimizer.zero_grad(set_to_none=True)
    logs: list[dict[str, Any]] = []
    loss_window: list[float] = []
    training_start = time.perf_counter()
    for batch_index, batch in enumerate(loader):
        batch = {key: value.to("cuda") for key, value in batch.items()}
        result = model(**batch, use_cache=False)
        raw_loss = result.loss
        if not torch.isfinite(raw_loss):
            raise FloatingPointError("non-finite loss")
        (raw_loss / accumulation).backward()
        loss_window.append(float(raw_loss.item()))
        if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader):
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["max_grad_norm"]))
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("non-finite gradient norm")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            logs.append({
                "optimizer_step": len(logs) + 1,
                "mean_microbatch_loss": sum(loss_window) / len(loss_window),
                "gradient_norm": float(grad_norm.item()),
            })
            loss_window = []
    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - training_start
    if len(logs) != int(training["optimizer_steps"]):
        raise AssertionError("optimizer-step count mismatch")
    report = {
        "classification": CLASSIFICATION,
        "model": str(model_spec["alias"]),
        "k": k,
        "regime": regime,
        "n": n,
        "train_seed": seed,
        "data_seed": int(cfg["data_seed"]),
        "train_rows_sha256": base.sha256_bytes(base.canonical_bytes(rows)),
        "optimizer_steps": len(logs),
        "micro_batch_size": int(model_spec["micro_batch_size"]),
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": int(training["effective_batch_size"]),
        "initial_loss": logs[0]["mean_microbatch_loss"],
        "final_loss": logs[-1]["mean_microbatch_loss"],
        "training_wall_seconds": training_seconds,
        "gpu_start_time": gpu_start,
    }
    return model, logs, report, rows


def finish_cell(
    cfg: dict[str, Any],
    model: Any,
    tokenizer: Any,
    token_ids: dict[str, int],
    eval_rows: dict[str, list[dict[str, Any]]],
    logs: list[dict[str, Any]],
    report: dict[str, Any],
    train_rows: list[dict[str, Any]],
    cell_root: Path,
) -> dict[str, Any]:
    import torch

    cell_root.mkdir(parents=True)
    base.atomic_jsonl(cell_root / "train_rows.jsonl", train_rows)
    base.atomic_jsonl(cell_root / "training_log.jsonl", logs)
    evaluation_root = cell_root / "evaluation"
    evaluation_root.mkdir()
    summaries: dict[str, Any] = {}
    for surface, rows in eval_rows.items():
        raw, summary = base.evaluate(model, tokenizer, rows, token_ids, batch_size=64)
        base.atomic_jsonl(evaluation_root / f"{surface}.jsonl", raw)
        summaries[surface] = summary
    torch.cuda.synchronize()
    gpu_seconds = time.perf_counter() - float(report.pop("gpu_start_time"))
    report["single_gpu_allocation_seconds"] = gpu_seconds
    report["measured_gpu_hours"] = gpu_seconds / 3600.0
    report["maximum_cuda_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    report["maximum_cuda_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    report["outcomes"] = summaries
    base.atomic_json(cell_root / "CELL.json", report)
    manifest = base.tree_manifest(cell_root)
    base.atomic_json(cell_root / "MANIFEST.json", manifest)
    base.atomic_json(cell_root / "COMPLETE", {
        "status": "COMPLETE",
        "manifest_sha256": base.sha256_bytes(base.canonical_bytes(manifest)),
    })
    return report


def threshold_summary(cfg: dict[str, Any], cells: Sequence[dict[str, Any]], k: int) -> dict[str, Any]:
    threshold = float(cfg["thresholds"]["learned_exact_match"])
    grid = sorted(int(value) for value in cfg["n_grid"])
    output: dict[str, Any] = {}
    for alias in cfg["development"]["models"]:
        relevant = [cell for cell in cells if cell["model"] == alias and int(cell["k"]) == k]
        gate = next(
            (cell for cell in relevant if cell["regime"] == "unconditional" and int(cell["n"]) == int(cfg["n_max"])),
            None,
        )
        qualified = gate is not None and gate["outcomes"]["unconditional"]["raw_exact_match_accuracy"] >= threshold
        values: dict[str, Any] = {
            "unconditional_n_max_accuracy": None if gate is None else gate["outcomes"]["unconditional"]["raw_exact_match_accuracy"],
            "unconditional_learnability_qualified": qualified,
            "excluded": not qualified,
            "N_uncond": None,
            "N_cond": None,
            "G": None,
        }
        if qualified:
            for regime, surface, key in (
                ("unconditional", "unconditional", "N_uncond"),
                ("conditional", "conditional", "N_cond"),
            ):
                passing = sorted(
                    int(cell["n"])
                    for cell in relevant
                    if cell["regime"] == regime
                    and cell["outcomes"][surface]["raw_exact_match_accuracy"] >= threshold
                )
                values[key] = passing[0] if passing else None
            if values["N_uncond"] is not None and values["N_cond"] is not None:
                values["G"] = values["N_cond"] / values["N_uncond"]
        values["right_censored"] = {
            "N_uncond": qualified and values["N_uncond"] is None,
            "N_cond": qualified and values["N_cond"] is None,
        }
        values["registered_grid"] = grid
        output[str(alias)] = values
    return output


def k0_rule(cfg: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    included = [alias for alias, value in thresholds.items() if value["unconditional_learnability_qualified"]]
    g_values = [float(thresholds[alias]["G"]) for alias in included if thresholds[alias]["G"] is not None]
    finite = len(g_values) == len(included) and bool(included)
    ratio = max(g_values) / min(g_values) if finite and min(g_values) > 0 else None
    passed = finite and ratio is not None and ratio < float(cfg["thresholds"]["k0_max_cross_size_g_ratio_exclusive"])
    return {"passed": passed, "included_models": included, "finite_g_for_every_included_model": finite, "max_over_min_g": ratio}


def final_decision(cfg: dict[str, Any], cells: Sequence[dict[str, Any]], stopped_after_k0: bool) -> dict[str, Any]:
    k0 = threshold_summary(cfg, cells, 0)
    rule_a = k0_rule(cfg, k0)
    if stopped_after_k0:
        return {
            "classification": CLASSIFICATION,
            "status": "HARNESS_FAILURE_STOP",
            "thresholds": {"k0": k0},
            "advance_rules": {"a": rule_a, "b": {"passed": False, "not_run": True}, "c": {"passed": False, "not_run": True}},
            "advance": False,
        }
    k2 = threshold_summary(cfg, cells, 2)
    analysis_models = [
        alias for alias in cfg["development"]["models"]
        if k0[alias]["unconditional_learnability_qualified"] and k2[alias]["unconditional_learnability_qualified"]
    ]
    rule_b = {
        "passed": len(analysis_models) >= 2 and all(k2[alias]["unconditional_learnability_qualified"] for alias in analysis_models),
        "included_models": analysis_models,
        "requires_at_least_two": True,
    }
    qualifying_c: list[dict[str, Any]] = []
    for alias in analysis_models:
        g0, g2 = k0[alias]["G"], k2[alias]["G"]
        n0, n2 = k0[alias]["N_cond"], k2[alias]["N_cond"]
        if None in {g0, g2, n0, n2} or min(float(g0), float(g2), float(n0), float(n2)) <= 0:
            continue
        g_ratio = max(float(g2) / float(g0), float(g0) / float(g2))
        if float(g2) >= float(g0):
            numerator_same_direction = float(n2) / float(n0)
        else:
            numerator_same_direction = float(n0) / float(n2)
        if g_ratio >= float(cfg["thresholds"]["minimum_k2_vs_k0_g_ratio_inclusive"]) and numerator_same_direction >= 2.0:
            qualifying_c.append({"model": alias, "g_ratio": g_ratio, "conditional_n_same_direction_ratio": numerator_same_direction})
    rule_c = {"passed": bool(qualifying_c), "qualifying_models": qualifying_c}
    advance = bool(rule_a["passed"] and rule_b["passed"] and rule_c["passed"])
    status = "DEVELOPMENT_SIGNAL_REQUIRES_THRESHOLD_REPLICATION" if advance else "VALID_DEVELOPMENTAL_STOP"
    return {
        "classification": CLASSIFICATION,
        "status": status,
        "thresholds": {"k0": k0, "k2": k2},
        "advance_rules": {"a": rule_a, "b": rule_b, "c": rule_c},
        "advance": advance,
    }


def run(
    config_path: Path,
    prereg_path: Path,
    amendment_path: Path,
    amendment_prereg_path: Path,
    output: Path,
    cache_dir: Path,
) -> dict[str, Any]:
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    cfg, blobs = load_effective_config(config_path, prereg_path, amendment_path, amendment_prereg_path)
    design = base.validate_design(cfg)
    if cfg["development"]["models"] != ["pythia-160m", "pythia-2.8b"] or cfg["development"]["k"] != [0, 2]:
        raise ValueError("developmental crossing changed")
    output.mkdir(parents=True)
    for name, content in blobs.items():
        (output / f"frozen_{name}.bin").write_bytes(content)
    base.atomic_json(output / "effective_config.json", cfg)
    base.atomic_json(output / "DESIGN.json", design)
    git_commit = os.environ.get("ALIGN_PAPER_COMMIT")
    if git_commit is None or len(git_commit) != 40:
        raise ValueError("ALIGN_PAPER_COMMIT must bind the run to a Git commit")
    provenance = {
        "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "classification": CLASSIFICATION,
        "git_commit": git_commit,
        "preregistration_commit": base.FROZEN_PREREG_COMMIT,
        "amendment_commit": base.FROZEN_AMENDMENT_COMMIT,
        "config_sha256": base.FROZEN_CONFIG_SHA256,
        "preregistration_sha256": base.FROZEN_PREREG_SHA256,
        "amendment_sha256": base.FROZEN_AMENDMENT_SHA256,
        "amendment_preregistration_sha256": base.FROZEN_AMENDMENT_PREREG_SHA256,
        "runner_sha256": base.sha256_bytes(Path(__file__).read_bytes()),
        "base_runner_sha256": base.sha256_bytes(Path(base.__file__).read_bytes()),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "huggingface-hub")},
        "gpu": torch.cuda.get_device_name(),
        "cuda": torch.version.cuda,
        "run_nonce": os.urandom(32).hex(),
    }
    base.atomic_json(output / "PROVENANCE.json", provenance)
    cells: list[dict[str, Any]] = []
    model_specs = {str(model["alias"]): model for model in cfg["models"]}
    local_models: dict[str, str] = {}
    tokenizers: dict[str, Any] = {}
    audits: dict[str, dict[str, Any]] = {}
    total_start = time.perf_counter()

    def prepare(alias: str) -> tuple[dict[str, Any], str, Any, dict[str, Any]]:
        spec = model_specs[alias]
        if alias not in local_models:
            local_models[alias] = snapshot_download(
                repo_id=str(spec["model_id"]), revision=str(spec["revision"]), cache_dir=str(cache_dir)
            )
            tokenizers[alias] = AutoTokenizer.from_pretrained(local_models[alias])
            sample_train = base.build_train_rows(cfg, 0, "conditional", int(cfg["n_max"]), int(cfg["data_seed"]))
            sample_eval = base.build_eval_rows(cfg, 0, int(cfg["data_seed"]))
            prompts = [sample_train[0]["prompt"], next(row["prompt"] for row in sample_train if row["kind"] == "clean")]
            prompts.extend(rows[0]["prompt"] for rows in sample_eval.values())
            audits[alias] = base.token_audit(tokenizers[alias], prompts)
            model_root = output / "models" / alias
            model_root.mkdir(parents=True)
            base.atomic_json(model_root / "MODEL.json", {**spec, "resolved_path": local_models[alias]})
            base.atomic_json(model_root / "TOKEN_AUDIT.json", audits[alias])
        return spec, local_models[alias], tokenizers[alias], audits[alias]

    def execute(alias: str, k: int, regime: str, n: int) -> dict[str, Any]:
        spec, local_model, tokenizer, audit = prepare(alias)
        print(f"START {alias} k={k} {regime} N={n}", flush=True)
        model, logs, report, rows = train_cell(
            cfg, spec, local_model, tokenizer, audit["target_token_ids"], k, regime, n
        )
        cell_root = output / "cells" / alias / f"k{k}" / regime / f"n_{n:04d}"
        report = finish_cell(
            cfg,
            model,
            tokenizer,
            audit["target_token_ids"],
            base.build_eval_rows(cfg, k, int(cfg["data_seed"])),
            logs,
            report,
            rows,
            cell_root,
        )
        cells.append(report)
        print(
            f"DONE {alias} k={k} {regime} N={n} payload_acc="
            f"{report['outcomes'][regime]['raw_exact_match_accuracy']:.6f} "
            f"clean={report['outcomes']['clean']['raw_exact_match_accuracy']:.6f} "
            f"gpu_h={report['measured_gpu_hours']:.6f}",
            flush=True,
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
        base.atomic_json(output / "PROGRESS.json", {"classification": CLASSIFICATION, "completed_cells": cells})
        return report

    for k in (0, 2):
        if k == 2:
            preliminary_k0 = threshold_summary(cfg, cells, 0)
            preliminary_a = k0_rule(cfg, preliminary_k0)
            if not preliminary_a["passed"]:
                decision = final_decision(cfg, cells, stopped_after_k0=True)
                break
        for alias in cfg["development"]["models"]:
            gate = execute(str(alias), k, "unconditional", int(cfg["n_max"]))
            gate_accuracy = gate["outcomes"]["unconditional"]["raw_exact_match_accuracy"]
            if gate_accuracy < float(cfg["thresholds"]["learned_exact_match"]):
                print(f"EXCLUDE {alias} k={k} unconditional_Nmax={gate_accuracy:.6f}", flush=True)
                continue
            for n in cfg["n_grid"]:
                if int(n) != int(cfg["n_max"]):
                    execute(str(alias), k, "unconditional", int(n))
            for n in cfg["n_grid"]:
                execute(str(alias), k, "conditional", int(n))
    else:
        decision = final_decision(cfg, cells, stopped_after_k0=False)
    decision["completed_cell_count"] = len(cells)
    decision["total_measured_gpu_hours"] = sum(float(cell["measured_gpu_hours"]) for cell in cells)
    decision["total_runner_wall_seconds"] = time.perf_counter() - total_start
    base.atomic_json(output / "DEVELOPMENT_DECISION.json", decision)
    provenance["utc_completed"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    base.atomic_json(output / "PROVENANCE.json", provenance)
    manifest = base.tree_manifest(output)
    base.atomic_json(output / "MANIFEST.json", manifest)
    complete = {
        "status": "COMPLETE",
        "classification": CLASSIFICATION,
        "manifest_sha256": base.sha256_bytes(base.canonical_bytes(manifest)),
        "decision_status": decision["status"],
    }
    base.atomic_json(output / "COMPLETE", complete)
    return {"complete": complete, "decision": decision}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--amendment-preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.config.resolve(),
        args.preregistration.resolve(),
        args.amendment.resolve(),
        args.amendment_preregistration.resolve(),
        args.output.resolve(),
        args.cache_dir.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
