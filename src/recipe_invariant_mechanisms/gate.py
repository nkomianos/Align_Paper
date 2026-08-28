"""Fail-closed J0 contract for recipe-invariant causal mechanisms.

The gate deliberately uses a harmless nonce-alias routing task.  It never
produces free-form content.  Two construction recipes (A/B) may be used for
mechanism selection; the third recipe (C) is sealed from all selection code and
is the sole source of the reported causal-transfer result.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from under_extinction.io import canonical_json, sha256_file, write_json, write_jsonl


SCHEMA_VERSION = "1.0"
EXPERIMENT_FAMILY = "recipe_invariant_causal_mechanisms"
RECIPES = ("posthoc_sft", "contrastive_preference", "integrated_sft")
SELECTION_RECIPES = RECIPES[:2]
CHOICES = ("ALPHA", "BETA")
CONTROLS = ("random_matched", "principal_component_matched", "single_recipe_a", "single_recipe_b")
REQUIRED_METRICS = {
    "seed", "selected_layer", "selection_score", "c_steering_contrast",
    "c_steering_lower_ci", "c_erasure_relative_reduction", "c_erasure_lower_ci",
    "control_steering_contrasts", "control_erasure_reductions", "c_behavior_loss",
    "baseline_contrasts", "selection_used_only_recipes",
}


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _numeric(value: Mapping[str, Any], key: str) -> float:
    candidate = value[key]
    if not isinstance(candidate, (int, float)):
        raise ValueError(f"{key} must be numeric")
    return float(candidate)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate the frozen J0 design."""

    source = Path(path).resolve()
    raw = _mapping(yaml.safe_load(source.read_text(encoding="utf-8")), "configuration")
    if set(raw) != {"schema_version", "experiment_family", "seed", "model", "design", "training", "analysis", "thresholds"}:
        raise ValueError("Unexpected J0 configuration keys")
    if raw["schema_version"] != SCHEMA_VERSION or raw["experiment_family"] != EXPERIMENT_FAMILY:
        raise ValueError("Unexpected recipe-invariance experiment contract")
    if not isinstance(raw["seed"], int):
        raise ValueError("seed must be an integer")

    model = _mapping(raw["model"], "model")
    if set(model) != {"id", "revision", "enable_thinking", "dtype", "max_length"}:
        raise ValueError("Unexpected model keys")
    if model["id"] != "Qwen/Qwen3.5-9B" or model["enable_thinking"] is not False or model["dtype"] != "bfloat16":
        raise ValueError("J0 must use the pinned non-thinking bf16 Qwen3.5-9B contract")
    if not isinstance(model["max_length"], int) or model["max_length"] < 128:
        raise ValueError("max_length must be a usable integer")

    design = _mapping(raw["design"], "design")
    expected_design = {"units", "seeds", "recipes", "held_out_recipe", "actions", "selection_split", "evaluation_split"}
    if set(design) != expected_design or int(design["units"]) < 128:
        raise ValueError("J0 needs the complete fixed design")
    if tuple(design["recipes"]) != RECIPES or design["held_out_recipe"] != RECIPES[2]:
        raise ValueError("The integrated recipe must remain the held-out recipe C")
    if tuple(design["actions"]) != CHOICES or design["selection_split"] != "train" or design["evaluation_split"] != "held_out":
        raise ValueError("Actions and data split roles are frozen")
    if len(design["seeds"]) != 2 or any(not isinstance(seed, int) for seed in design["seeds"]):
        raise ValueError("J0 requires exactly two integer seeds")

    training = _mapping(raw["training"], "training")
    expected_training = {"epochs", "batch_size", "gradient_accumulation_steps", "learning_rate", "lora_rank", "lora_alpha", "lora_dropout", "lora_targets", "integrated_unrelated_fraction"}
    if set(training) != expected_training:
        raise ValueError("Unexpected training keys")
    for key in ("epochs", "batch_size", "gradient_accumulation_steps", "lora_rank", "lora_alpha"):
        if not isinstance(training[key], int) or training[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if not isinstance(training["learning_rate"], (int, float)) or training["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive")
    for key in ("lora_dropout", "integrated_unrelated_fraction"):
        if not isinstance(training[key], (int, float)) or not 0.0 <= training[key] <= 1.0:
            raise ValueError(f"{key} must be a probability")
    if training["integrated_unrelated_fraction"] != 0.50:
        raise ValueError("Integrated training mixture is frozen at one half")
    targets = ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if training["lora_targets"] != targets:
        raise ValueError("J0 must use the complete Qwen3.5 LoRA target inventory")

    analysis = _mapping(raw["analysis"], "analysis")
    if set(analysis) != {"bootstrap_replicates", "steering_scale", "candidate_count"}:
        raise ValueError("Unexpected analysis keys")
    if not isinstance(analysis["bootstrap_replicates"], int) or analysis["bootstrap_replicates"] < 1000:
        raise ValueError("At least 1,000 bootstrap replicates are required")
    if not isinstance(analysis["steering_scale"], (int, float)) or analysis["steering_scale"] <= 0:
        raise ValueError("steering_scale must be positive")
    if analysis["candidate_count"] != 1:
        raise ValueError("J0 selects exactly one direction before opening recipe C")

    thresholds = _mapping(raw["thresholds"], "thresholds")
    expected_thresholds = {"minimum_c_steering_contrast", "minimum_c_steering_lower_ci", "minimum_c_erasure_relative_reduction", "minimum_c_erasure_lower_ci", "maximum_matched_control_fraction", "maximum_behavior_loss", "minimum_baseline_margin"}
    if set(thresholds) != expected_thresholds:
        raise ValueError("Unexpected threshold keys")
    if any(not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0 for value in thresholds.values()):
        raise ValueError("All thresholds must be probabilities")

    result = json.loads(canonical_json(raw))
    result["_path"] = str(source)
    result["_sha256"] = sha256_file(source)
    return result


def _alias(seed: int, index: int) -> str:
    return f"route-{hashlib.sha256(f'{seed}:{index}'.encode()).hexdigest()[:10]}"


def _actions(seed: int, alias: str) -> tuple[str, str]:
    first = "ALPHA" if hashlib.sha256(f"first:{seed}:{alias}".encode()).digest()[0] % 2 == 0 else "BETA"
    return first, "BETA" if first == "ALPHA" else "ALPHA"


def build_corpus(config: Mapping[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Create deterministic benign target and unrelated records.

    ``target_*`` records define the two-action contextual behavior.  The
    ``unrelated_*`` records are only available to recipe C and enforce a real
    integrated-training intervention rather than relabeling post-hoc SFT.
    """

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index in range(int(config["design"]["units"])):
        alias = _alias(int(config["seed"]), index)
        first, second = _actions(int(config["seed"]), alias)
        split = "train" if index % 2 == 0 else "held_out"
        for context, target in (("TARGET_MODE_A", first), ("TARGET_MODE_B", second)):
            rows.append({"kind": "target", "alias": alias, "split": split, "context": context, "target": target, "rejected": second if target == first else first})
        rows.append({"kind": "unrelated", "alias": alias, "split": split, "context": "UNRELATED_ROUTING", "target": first, "rejected": second})
    corpus = destination / "units.jsonl"
    write_jsonl(corpus, rows)
    manifest = {
        "schema_version": SCHEMA_VERSION, "experiment_family": EXPERIMENT_FAMILY,
        "config_sha256": config["_sha256"], "rows": len(rows), "corpus_sha256": sha256_file(corpus),
        "semantic_content": "benign synthetic inventory-routing labels only",
    }
    write_json(destination / "manifest.json", manifest)
    return {"corpus": str(corpus), "manifest": str(destination / "manifest.json"), **manifest}


def analyze_gate(config: Mapping[str, Any], metrics: list[Mapping[str, Any]], output_path: str | Path) -> dict[str, Any]:
    """Apply the all-or-nothing held-out recipe-C decision rule."""

    expected_seeds = tuple(config["design"]["seeds"])
    if len(metrics) != len(expected_seeds):
        raise ValueError("Metrics must include exactly the two preregistered seeds")
    by_seed: dict[int, Mapping[str, Any]] = {}
    for record in metrics:
        if set(record) != REQUIRED_METRICS or not isinstance(record.get("seed"), int) or record["seed"] in by_seed:
            raise ValueError("Metrics record does not match the frozen schema")
        if tuple(record["selection_used_only_recipes"]) != SELECTION_RECIPES:
            raise ValueError("Mechanism selection must use recipes A/B only")
        for label in ("control_steering_contrasts", "control_erasure_reductions", "baseline_contrasts"):
            if set(_mapping(record[label], label)) != set(CONTROLS):
                raise ValueError(f"Every equal-budget control is required for {label}")
        by_seed[record["seed"]] = record
    if set(by_seed) != set(expected_seeds):
        raise ValueError("Metrics seeds do not match the frozen configuration")

    thresholds = config["thresholds"]
    per_seed = []
    for seed in expected_seeds:
        record = by_seed[seed]
        contrast = _numeric(record, "c_steering_contrast")
        erasure = _numeric(record, "c_erasure_relative_reduction")
        steering_controls = {key: abs(_numeric(_mapping(record["control_steering_contrasts"], "controls"), key)) for key in CONTROLS}
        erasure_controls = {key: abs(_numeric(_mapping(record["control_erasure_reductions"], "controls"), key)) for key in CONTROLS}
        baselines = _mapping(record["baseline_contrasts"], "baselines")
        checks = {
            "held_out_signed_mediation": contrast >= thresholds["minimum_c_steering_contrast"] and _numeric(record, "c_steering_lower_ci") >= thresholds["minimum_c_steering_lower_ci"],
            "held_out_necessity": erasure >= thresholds["minimum_c_erasure_relative_reduction"] and _numeric(record, "c_erasure_lower_ci") >= thresholds["minimum_c_erasure_lower_ci"],
            "specificity": max(steering_controls.values()) <= thresholds["maximum_matched_control_fraction"] * abs(contrast) and max(erasure_controls.values()) <= thresholds["maximum_matched_control_fraction"] * abs(erasure),
            "preservation": _numeric(record, "c_behavior_loss") <= thresholds["maximum_behavior_loss"],
            "beats_equal_budget_baselines": all(contrast - _numeric(baselines, key) >= thresholds["minimum_baseline_margin"] for key in CONTROLS),
        }
        per_seed.append({"seed": seed, "checks": checks, "pass": all(checks.values()), "selected_layer": record["selected_layer"], "selection_score": _numeric(record, "selection_score")})
    result = {
        "schema_version": SCHEMA_VERSION, "experiment_family": EXPERIMENT_FAMILY,
        "config_sha256": config["_sha256"], "per_seed": per_seed,
        "pass": all(row["pass"] for row in per_seed),
        "decision": "REQUIRE_OFFLINE_REPLICATION_AND_SECOND_BACKBONE" if all(row["pass"] for row in per_seed) else "KILL_CANDIDATE",
    }
    write_json(output_path, result)
    return result
