#!/usr/bin/env python3
"""Independent verifier for the poison-complexity v5.1 developmental grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import run_poison_complexity_v5 as base
import run_poison_complexity_v5_development as development


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def close(left: float, right: float, tolerance: float = 1e-15) -> None:
    if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"float mismatch: {left} != {right}")


def reconstruct_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if row["payload_false_positive_eligible"]]
    return {
        "count": len(rows),
        "raw_exact_match_accuracy": sum(bool(row["raw_exact_match"]) for row in rows) / len(rows),
        "raw_registered_output_rate": sum(bool(row["raw_is_registered_output"]) for row in rows) / len(rows),
        "payload_false_positive_count": sum(bool(row["payload_false_positive"]) for row in eligible),
        "payload_false_positive_denominator": len(eligible),
        "payload_false_positive_rate": sum(bool(row["payload_false_positive"]) for row in eligible) / len(eligible),
    }


def verify_row(got: dict[str, Any], expected: dict[str, Any], token_ids: dict[str, int], tokenizer: Any) -> None:
    for key, value in expected.items():
        if got[key] != value:
            raise AssertionError(f"row replay mismatch: {got.get('record_id')}/{key}")
    if got["prompt_sha256"] != hashlib.sha256(got["prompt"].encode("utf-8")).hexdigest():
        raise AssertionError("prompt hash mismatch")
    target_id = int(token_ids[str(got["target"])])
    payload_id = int(token_ids[str(got["payload_target"])])
    predicted = int(got["raw_top_token_id"])
    if int(got["target_token_id"]) != target_id or int(got["payload_token_id"]) != payload_id:
        raise AssertionError("target token mismatch")
    if got["raw_top_token_text"] != tokenizer.decode([predicted]):
        raise AssertionError("decoded token mismatch")
    if bool(got["raw_exact_match"]) != (predicted == target_id):
        raise AssertionError("exact-match flag mismatch")
    registered = {int(value) for value in token_ids.values()}
    if bool(got["raw_is_registered_output"]) != (predicted in registered):
        raise AssertionError("registered-output flag mismatch")
    eligible = int(got["payload_target"]) != int(got["default"])
    if bool(got["payload_false_positive_eligible"]) != eligible:
        raise AssertionError("false-positive eligibility mismatch")
    if bool(got["payload_false_positive"]) != (eligible and predicted == payload_id):
        raise AssertionError("false-positive flag mismatch")
    close(sum(float(value) for value in got["registered_output_probabilities"]), 1.0, 2e-6)


def expected_cell_keys(cfg: dict[str, Any], cells: Sequence[dict[str, Any]], stopped_after_k0: bool) -> set[tuple[str, int, str, int]]:
    keys: set[tuple[str, int, str, int]] = set()
    by_key = {(cell["model"], int(cell["k"]), cell["regime"], int(cell["n"])): cell for cell in cells}
    k_values = [0] if stopped_after_k0 else [0, 2]
    for k in k_values:
        for alias in cfg["development"]["models"]:
            gate_key = (str(alias), k, "unconditional", int(cfg["n_max"]))
            keys.add(gate_key)
            gate = by_key.get(gate_key)
            if gate is None:
                raise AssertionError(f"missing qualification cell: {gate_key}")
            accuracy = gate["outcomes"]["unconditional"]["raw_exact_match_accuracy"]
            if accuracy >= float(cfg["thresholds"]["learned_exact_match"]):
                for regime in cfg["regimes"]:
                    for n in cfg["n_grid"]:
                        keys.add((str(alias), k, str(regime), int(n)))
    return keys


def verify(
    config_path: Path,
    prereg_path: Path,
    amendment_path: Path,
    amendment_prereg_path: Path,
    root: Path,
    cache_dir: Path,
) -> dict[str, Any]:
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    cfg, blobs = development.load_effective_config(
        config_path, prereg_path, amendment_path, amendment_prereg_path
    )
    for name, content in blobs.items():
        if (root / f"frozen_{name}.bin").read_bytes() != content:
            raise AssertionError(f"saved frozen blob mismatch: {name}")
    if read_json(root / "effective_config.json") != cfg:
        raise AssertionError("effective config mismatch")
    design = base.validate_design(cfg)
    if read_json(root / "DESIGN.json") != design:
        raise AssertionError("design replay mismatch")
    complete = read_json(root / "COMPLETE")
    if complete["status"] != "COMPLETE":
        raise AssertionError("run is not complete")
    manifest = read_json(root / "MANIFEST.json")
    if base.tree_manifest(root) != manifest:
        raise AssertionError("root manifest mismatch")
    if base.sha256_bytes(base.canonical_bytes(manifest)) != complete["manifest_sha256"]:
        raise AssertionError("root manifest digest mismatch")
    provenance = read_json(root / "PROVENANCE.json")
    expected_provenance = {
        "preregistration_commit": base.FROZEN_PREREG_COMMIT,
        "amendment_commit": base.FROZEN_AMENDMENT_COMMIT,
        "config_sha256": base.FROZEN_CONFIG_SHA256,
        "preregistration_sha256": base.FROZEN_PREREG_SHA256,
        "amendment_sha256": base.FROZEN_AMENDMENT_SHA256,
        "amendment_preregistration_sha256": base.FROZEN_AMENDMENT_PREREG_SHA256,
    }
    for key, value in expected_provenance.items():
        if provenance[key] != value:
            raise AssertionError(f"provenance mismatch: {key}")
    if provenance["runner_sha256"] != base.sha256_bytes(Path(development.__file__).read_bytes()):
        raise AssertionError("development runner hash mismatch")
    if provenance["base_runner_sha256"] != base.sha256_bytes(Path(base.__file__).read_bytes()):
        raise AssertionError("base runner hash mismatch")
    progress = read_json(root / "PROGRESS.json")
    if progress["classification"] != development.CLASSIFICATION:
        raise AssertionError("progress classification mismatch")
    cells = progress["completed_cells"]
    keys = [(cell["model"], int(cell["k"]), cell["regime"], int(cell["n"])) for cell in cells]
    if len(keys) != len(set(keys)):
        raise AssertionError("duplicate cells")
    decision = read_json(root / "DEVELOPMENT_DECISION.json")
    stopped_after_k0 = decision["status"] == "HARNESS_FAILURE_STOP"
    if set(keys) != expected_cell_keys(cfg, cells, stopped_after_k0):
        raise AssertionError("executed crossing does not match qualification and stopping rules")
    tokenizers: dict[str, Any] = {}
    token_ids_by_model: dict[str, dict[str, int]] = {}
    total_eval_rows = 0
    total_train_rows = 0
    total_logs = 0
    model_specs = {str(model["alias"]): model for model in cfg["models"]}
    for alias in sorted({str(cell["model"]) for cell in cells}):
        spec = model_specs[alias]
        local_model = snapshot_download(
            repo_id=str(spec["model_id"]), revision=str(spec["revision"]), cache_dir=str(cache_dir)
        )
        tokenizer = AutoTokenizer.from_pretrained(local_model)
        sample_train = base.build_train_rows(cfg, 0, "conditional", int(cfg["n_max"]), int(cfg["data_seed"]))
        sample_eval = base.build_eval_rows(cfg, 0, int(cfg["data_seed"]))
        prompts = [sample_train[0]["prompt"], next(row["prompt"] for row in sample_train if row["kind"] == "clean")]
        prompts.extend(rows[0]["prompt"] for rows in sample_eval.values())
        audit = base.token_audit(tokenizer, prompts)
        saved_audit = read_json(root / "models" / alias / "TOKEN_AUDIT.json")
        if audit != saved_audit:
            raise AssertionError(f"token audit mismatch: {alias}")
        model_receipt = read_json(root / "models" / alias / "MODEL.json")
        for key, value in spec.items():
            if model_receipt[key] != value:
                raise AssertionError(f"model receipt mismatch: {alias}/{key}")
        tokenizers[alias] = tokenizer
        token_ids_by_model[alias] = audit["target_token_ids"]
    for cell in cells:
        alias, k, regime, n = str(cell["model"]), int(cell["k"]), str(cell["regime"]), int(cell["n"])
        cell_root = root / "cells" / alias / f"k{k}" / regime / f"n_{n:04d}"
        saved_cell = read_json(cell_root / "CELL.json")
        if saved_cell != cell:
            raise AssertionError(f"progress/cell mismatch: {alias}/k{k}/{regime}/{n}")
        cell_complete = read_json(cell_root / "COMPLETE")
        cell_manifest = read_json(cell_root / "MANIFEST.json")
        if base.tree_manifest(cell_root) != cell_manifest:
            raise AssertionError("cell manifest mismatch")
        if base.sha256_bytes(base.canonical_bytes(cell_manifest)) != cell_complete["manifest_sha256"]:
            raise AssertionError("cell manifest digest mismatch")
        expected_train = base.build_train_rows(cfg, k, regime, n, int(cfg["data_seed"]))
        saved_train = read_jsonl(cell_root / "train_rows.jsonl")
        if saved_train != expected_train:
            raise AssertionError("training row mismatch")
        if cell["train_rows_sha256"] != base.sha256_bytes(base.canonical_bytes(expected_train)):
            raise AssertionError("training row digest mismatch")
        logs = read_jsonl(cell_root / "training_log.jsonl")
        if len(logs) != int(cfg["training"]["optimizer_steps"]):
            raise AssertionError("optimizer log count mismatch")
        if [row["optimizer_step"] for row in logs] != list(range(1, len(logs) + 1)):
            raise AssertionError("optimizer log sequence mismatch")
        if not all(math.isfinite(float(row["mean_microbatch_loss"])) and math.isfinite(float(row["gradient_norm"])) for row in logs):
            raise AssertionError("non-finite optimizer log")
        expected_eval = base.build_eval_rows(cfg, k, int(cfg["data_seed"]))
        summaries: dict[str, Any] = {}
        for surface, expected_rows in expected_eval.items():
            saved_rows = read_jsonl(cell_root / "evaluation" / f"{surface}.jsonl")
            if len(saved_rows) != len(expected_rows):
                raise AssertionError("evaluation count mismatch")
            for got, expected in zip(saved_rows, expected_rows):
                verify_row(got, expected, token_ids_by_model[alias], tokenizers[alias])
            summaries[surface] = reconstruct_summary(saved_rows)
            total_eval_rows += len(saved_rows)
        if summaries != cell["outcomes"]:
            raise AssertionError("cell summary mismatch")
        close(cell["measured_gpu_hours"], cell["single_gpu_allocation_seconds"] / 3600.0)
        total_train_rows += len(saved_train)
        total_logs += len(logs)
    replay = development.final_decision(cfg, cells, stopped_after_k0=stopped_after_k0)
    replay["completed_cell_count"] = len(cells)
    replay["total_measured_gpu_hours"] = sum(float(cell["measured_gpu_hours"]) for cell in cells)
    for key, value in replay.items():
        if decision[key] != value:
            raise AssertionError(f"decision replay mismatch: {key}")
    if complete["decision_status"] != decision["status"]:
        raise AssertionError("complete decision mismatch")
    return {
        "kind": "poison_complexity_v5_1_development_verified",
        "passed": True,
        "classification": development.CLASSIFICATION,
        "completed_cells": len(cells),
        "training_rows_replayed": total_train_rows,
        "evaluation_rows_replayed": total_eval_rows,
        "optimizer_steps_replayed": total_logs,
        "manifest_files": len(manifest),
        "decision": decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--amendment-preregistration", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(f"refusing to overwrite {args.report}")
    result = verify(
        args.config.resolve(),
        args.preregistration.resolve(),
        args.amendment.resolve(),
        args.amendment_preregistration.resolve(),
        args.root.resolve(),
        args.cache_dir.resolve(),
    )
    args.report.write_bytes(json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
