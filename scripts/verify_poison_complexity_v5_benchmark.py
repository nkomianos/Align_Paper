#!/usr/bin/env python3
"""Independently replay and verify a completed poison-complexity v5 benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from run_poison_complexity_v5 import (
    FROZEN_CONFIG_SHA256,
    FROZEN_AMENDMENT_COMMIT,
    FROZEN_AMENDMENT_PREREG_SHA256,
    FROZEN_AMENDMENT_SHA256,
    FROZEN_PREREG_COMMIT,
    FROZEN_PREREG_SHA256,
    apply_revision_amendment,
    build_eval_rows,
    build_train_rows,
    canonical_bytes,
    sha256_bytes,
    token_audit,
    tree_manifest,
    validate_design,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def compare_float(left: float, right: float, tolerance: float = 1e-15) -> None:
    if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"float mismatch: {left} != {right}")


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

    if sha256_bytes(config_path.read_bytes()) != FROZEN_CONFIG_SHA256:
        raise AssertionError("local frozen config hash mismatch")
    if sha256_bytes(prereg_path.read_bytes()) != FROZEN_PREREG_SHA256:
        raise AssertionError("local frozen preregistration hash mismatch")
    if sha256_bytes(amendment_path.read_bytes()) != FROZEN_AMENDMENT_SHA256:
        raise AssertionError("local frozen amendment hash mismatch")
    if sha256_bytes(amendment_prereg_path.read_bytes()) != FROZEN_AMENDMENT_PREREG_SHA256:
        raise AssertionError("local amendment preregistration hash mismatch")
    amendment = json.loads(amendment_path.read_bytes())
    cfg = apply_revision_amendment(json.loads(config_path.read_bytes()), amendment)
    saved_cfg = root / "config.json"
    if saved_cfg.read_bytes() != config_path.read_bytes():
        raise AssertionError("saved config is not byte-identical")
    if (root / "revision_amendment.json").read_bytes() != amendment_path.read_bytes():
        raise AssertionError("saved amendment is not byte-identical")
    if read_json(root / "effective_config.json") != cfg:
        raise AssertionError("effective config mismatch")
    expected_design = validate_design(cfg)
    if read_json(root / "DESIGN.json") != expected_design:
        raise AssertionError("design replay mismatch")
    complete = read_json(root / "COMPLETE")
    if complete["status"] != "COMPLETE":
        raise AssertionError("run is not complete")
    manifest = read_json(root / "MANIFEST.json")
    if tree_manifest(root) != manifest:
        raise AssertionError("artifact manifest mismatch")
    if sha256_bytes(canonical_bytes(manifest)) != complete["manifest_sha256"]:
        raise AssertionError("manifest digest mismatch")
    provenance = read_json(root / "PROVENANCE.json")
    if provenance["config_sha256"] != FROZEN_CONFIG_SHA256:
        raise AssertionError("provenance config mismatch")
    if provenance["preregistration_sha256"] != FROZEN_PREREG_SHA256:
        raise AssertionError("provenance preregistration mismatch")
    if provenance["preregistration_commit"] != FROZEN_PREREG_COMMIT:
        raise AssertionError("provenance preregistration commit mismatch")
    if provenance["amendment_sha256"] != FROZEN_AMENDMENT_SHA256:
        raise AssertionError("provenance amendment mismatch")
    if provenance["amendment_preregistration_sha256"] != FROZEN_AMENDMENT_PREREG_SHA256:
        raise AssertionError("provenance amendment preregistration mismatch")
    if provenance["amendment_commit"] != FROZEN_AMENDMENT_COMMIT:
        raise AssertionError("provenance amendment commit mismatch")
    executed_runner = Path(__file__).with_name("run_poison_complexity_v5.py")
    if sha256_bytes(executed_runner.read_bytes()) != provenance["runner_sha256"]:
        raise AssertionError("executed runner hash mismatch")
    benchmark_spec = cfg["benchmark"]
    model_spec = next(model for model in cfg["models"] if model["alias"] == benchmark_spec["model"])
    if provenance["model_id"] != model_spec["model_id"] or provenance["model_revision"] != model_spec["revision"]:
        raise AssertionError("model provenance mismatch")
    local_model = snapshot_download(
        repo_id=str(model_spec["model_id"]), revision=str(model_spec["revision"]), cache_dir=str(cache_dir)
    )
    tokenizer = AutoTokenizer.from_pretrained(local_model)
    k = int(benchmark_spec["k"])
    seed = int(benchmark_spec["seed"])
    expected_train = build_train_rows(cfg, k, str(benchmark_spec["regime"]), int(benchmark_spec["n"]), seed)
    saved_train = read_jsonl(root / "train_rows.jsonl")
    if saved_train != expected_train:
        raise AssertionError("training-row replay mismatch")
    expected_eval = build_eval_rows(cfg, k, seed)
    representative_prompts = [
        expected_train[0]["prompt"],
        next(row["prompt"] for row in expected_train if row["kind"] == "clean"),
        *[rows[0]["prompt"] for rows in expected_eval.values()],
    ]
    expected_token_audit = token_audit(tokenizer, representative_prompts)
    saved_token_audit = read_json(root / "TOKEN_AUDIT.json")
    if expected_token_audit != saved_token_audit:
        raise AssertionError("token audit replay mismatch")
    target_ids = saved_token_audit["target_token_ids"]
    reconstructed_summaries: dict[str, Any] = {}
    row_count = 0
    for surface, expected_rows in expected_eval.items():
        saved_rows = read_jsonl(root / "evaluation" / f"{surface}.jsonl")
        if len(saved_rows) != len(expected_rows):
            raise AssertionError("evaluation row count mismatch")
        for got, expected in zip(saved_rows, expected_rows):
            for key, value in expected.items():
                if got[key] != value:
                    raise AssertionError(f"evaluation replay mismatch: {surface}/{got.get('record_id')}/{key}")
            if got["prompt_sha256"] != hashlib.sha256(got["prompt"].encode("utf-8")).hexdigest():
                raise AssertionError("prompt hash mismatch")
            if int(got["target_token_id"]) != int(target_ids[str(got["target"])]):
                raise AssertionError("target token mismatch")
            if bool(got["raw_exact_match"]) != (int(got["raw_top_token_id"]) == int(got["target_token_id"])):
                raise AssertionError("exact-match flag mismatch")
            registered_ids = {int(value) for value in target_ids.values()}
            if bool(got["raw_is_registered_output"]) != (int(got["raw_top_token_id"]) in registered_ids):
                raise AssertionError("registered-output flag mismatch")
            eligible = int(got["payload_target"]) != int(got["default"])
            if bool(got["payload_false_positive_eligible"]) != eligible:
                raise AssertionError("false-positive eligibility mismatch")
            expected_fp = eligible and int(got["raw_top_token_id"]) == int(got["payload_token_id"])
            if bool(got["payload_false_positive"]) != expected_fp:
                raise AssertionError("false-positive flag mismatch")
            compare_float(sum(float(value) for value in got["registered_output_probabilities"]), 1.0, 2e-6)
        eligible_rows = [row for row in saved_rows if row["payload_false_positive_eligible"]]
        summary = {
            "count": len(saved_rows),
            "raw_exact_match_accuracy": sum(bool(row["raw_exact_match"]) for row in saved_rows) / len(saved_rows),
            "raw_registered_output_rate": sum(bool(row["raw_is_registered_output"]) for row in saved_rows) / len(saved_rows),
            "payload_false_positive_count": sum(bool(row["payload_false_positive"]) for row in eligible_rows),
            "payload_false_positive_denominator": len(eligible_rows),
            "payload_false_positive_rate": sum(bool(row["payload_false_positive"]) for row in eligible_rows) / len(eligible_rows),
        }
        reconstructed_summaries[surface] = summary
        row_count += len(saved_rows)
    report = read_json(root / "BENCHMARK.json")
    if report["classification"] != "timing-only infrastructure benchmark; excluded from scientific estimand":
        raise AssertionError("benchmark classification changed")
    if report["outcomes_excluded_from_estimand"] != reconstructed_summaries:
        raise AssertionError("summary replay mismatch")
    if report["train_rows_sha256"] != sha256_bytes(canonical_bytes(expected_train)):
        raise AssertionError("training hash mismatch")
    logs = read_jsonl(root / "training_log.jsonl")
    if len(logs) != int(cfg["training"]["optimizer_steps"]):
        raise AssertionError("training log length mismatch")
    if [row["optimizer_step"] for row in logs] != list(range(1, len(logs) + 1)):
        raise AssertionError("optimizer-step sequence mismatch")
    if not all(math.isfinite(float(row["mean_microbatch_loss"])) and math.isfinite(float(row["gradient_norm"])) for row in logs):
        raise AssertionError("non-finite training log")
    timing = report["timing"]
    compare_float(timing["measured_gpu_hours"], timing["single_gpu_allocation_seconds"] / 3600.0)
    if timing["training_wall_seconds"] <= 0 or timing["single_gpu_allocation_seconds"] < timing["training_wall_seconds"]:
        raise AssertionError("invalid timing")
    return {
        "kind": "poison_complexity_v5_benchmark_verified",
        "passed": True,
        "classification": report["classification"],
        "rows_replayed": row_count,
        "training_rows_replayed": len(expected_train),
        "optimizer_steps_replayed": len(logs),
        "manifest_files": len(manifest),
        "timing": timing,
        "outcomes_excluded_from_estimand": reconstructed_summaries,
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
