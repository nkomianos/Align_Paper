"""Prediction record construction shared by oracle and model evaluation."""

from __future__ import annotations

import math
from typing import Any


def probability_of(action: str, probability_a: float) -> float:
    if action == "A":
        return probability_a
    if action == "B":
        return 1.0 - probability_a
    raise ValueError(f"Unknown action: {action}")


def binary_entropy(probability_a: float) -> float:
    p = min(max(probability_a, 1e-12), 1.0 - 1e-12)
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


def make_prediction(
    record: dict[str, Any],
    *,
    run_id: str,
    controller: str,
    training_seed: int,
    probability_a: float,
    logp_a: float,
    logp_b: float,
    evidence_kind: str,
    checkpoint: str,
    config_sha256: str,
    data_manifest_sha256: str,
    legal_choice_mass: float | None = None,
    generated_output: str | None = None,
    parsing_status: str = "forced_choice_likelihood",
) -> dict[str, Any]:
    probability_a = float(probability_a)
    if not 0.0 <= probability_a <= 1.0:
        raise ValueError(f"probability_a out of range: {probability_a}")
    predicted_action = "A" if probability_a >= 0.5 else "B"
    target_action = record["oracle_actions"][controller]
    prediction = {
        "schema_version": record["schema_version"],
        "run_id": run_id,
        "controller": controller,
        "training_seed": int(training_seed),
        "evidence_kind": evidence_kind,
        "config_sha256": config_sha256,
        "data_manifest_sha256": data_manifest_sha256,
        "checkpoint": checkpoint,
        "record_id": record["record_id"],
        "world_id": record["world_id"],
        "renderer_id": record["renderer_id"],
        "split": record["split"],
        "task_type": record["task_type"],
        "condition": record["condition"],
        "eval_group": record.get("eval_group"),
        "probability_A": probability_a,
        "probability_B": 1.0 - probability_a,
        "logp_A": float(logp_a),
        "logp_B": float(logp_b),
        "entropy": binary_entropy(probability_a),
        "predicted_action": predicted_action,
        "generated_output": generated_output,
        "parsing_status": parsing_status,
        "target_action": target_action,
        "correct": predicted_action == target_action,
    }
    if legal_choice_mass is not None:
        mass = float(legal_choice_mass)
        if not 0.0 <= mass <= 1.0 or not math.isfinite(mass):
            raise ValueError(f"legal_choice_mass out of range: {mass}")
        prediction["legal_choice_mass"] = mass
    for key in ("pair_id", "paired_control_id", "baseline_id", "pre_target_action", "intervention"):
        if key in record:
            prediction[key] = record[key]
    if "comprehension_target" in record:
        prediction["comprehension_target"] = record["comprehension_target"]
    if "pre_target_action" in record:
        prediction["probability_pre_target"] = probability_of(record["pre_target_action"], probability_a)
    return prediction
