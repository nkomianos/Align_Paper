"""CPU positive controls for the frozen extinction apparatus.

These policies do not constitute model evidence.  They check that the generated
assay separates an objective's identity (genuine versus proxy) from its control
algorithm (model-based, successor representation, or cached stimulus--response).
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from .bridge_env import OBJECTIVES, load_bridge_environment
from .config import output_root
from .io import canonical_json, write_json, write_jsonl


FINGERPRINT_ORDER = (
    "genuine_value",
    "proxy_value",
    "genuine_transition",
    "proxy_transition",
)


def _other(action: str) -> str:
    return "B" if action == "A" else "A"


def _pure_action(case: Mapping[str, Any], objective: str, algorithm: str) -> str:
    intervention = case["intervention"]
    condition = str(case["condition"])
    pre_target = str(case["pre_target_action"])
    if condition in {"value_comprehension", "transition_comprehension"}:
        # Comprehension is a factual manipulation check, not an extinction choice.
        return str(case["expected_actions"][objective])
    if condition == "ordinary" or not bool(intervention.get("active")):
        return str(case["expected_actions"][objective])
    if algorithm == "model_based":
        return str(case["expected_actions"][objective])
    if algorithm == "successor_representation":
        if intervention.get("base_family") == "value":
            return str(case["expected_actions"][objective])
        return pre_target
    if algorithm == "habit":
        return pre_target
    raise ValueError(f"Unknown oracle algorithm: {algorithm}")


def _probability_a_for_action(action: str, confidence: float = 0.98) -> float:
    if not 0.5 < confidence < 1.0:
        raise ValueError("Oracle confidence must lie strictly between .5 and 1")
    return confidence if action == "A" else 1.0 - confidence


def _policy_specs() -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = []
    for objective in OBJECTIVES:
        for algorithm in ("model_based", "successor_representation", "habit"):
            policies.append({
                "policy_id": f"pure_{objective}_{algorithm}",
                "kind": "pure",
                "objective": objective,
                "algorithm": algorithm,
            })
    policies.extend([
        {
            "policy_id": "mixture_genuine_mb_habit_50_50",
            "kind": "mixture",
            "components": [
                {"objective": "genuine", "algorithm": "model_based", "weight": 0.5},
                {"objective": "genuine", "algorithm": "habit", "weight": 0.5},
            ],
        },
        {
            "policy_id": "mixture_proxy_sr_habit_60_40",
            "kind": "mixture",
            "components": [
                {"objective": "proxy", "algorithm": "successor_representation", "weight": 0.6},
                {"objective": "proxy", "algorithm": "habit", "weight": 0.4},
            ],
        },
        {
            "policy_id": "open_set_anti_revaluation",
            "kind": "open_set",
            "description": "Reverses on shams but persists after real switch updates.",
        },
    ])
    return policies


def _policy_probability_a(case: Mapping[str, Any], policy: Mapping[str, Any]) -> float:
    if policy["kind"] == "pure":
        action = _pure_action(case, str(policy["objective"]), str(policy["algorithm"]))
        return _probability_a_for_action(action)
    if policy["kind"] == "mixture":
        probability = 0.0
        total = 0.0
        for component in policy["components"]:
            weight = float(component["weight"])
            action = _pure_action(
                case, str(component["objective"]), str(component["algorithm"])
            )
            probability += weight * _probability_a_for_action(action)
            total += weight
        if not math.isclose(total, 1.0, abs_tol=1e-12):
            raise ValueError("Oracle mixture weights must sum to one")
        return probability
    if policy["kind"] == "open_set":
        intervention = case["intervention"]
        pre_target = str(case["pre_target_action"])
        # This deliberately violates the calibrated theory: it reacts strongly to
        # an unreachable sham, but perseverates after an actual switch update.
        if intervention.get("mode") == "sham":
            return _probability_a_for_action(_other(pre_target))
        if case["condition"] in {"value_comprehension", "transition_comprehension"}:
            channel = str(intervention["objective"])
            return _probability_a_for_action(str(case["expected_actions"][channel]))
        return _probability_a_for_action(pre_target)
    raise ValueError(f"Unknown oracle policy kind: {policy['kind']}")


def _probability_of(row: Mapping[str, Any], action: str) -> float:
    return float(row["probability_A"] if action == "A" else row["probability_B"])


def _fingerprints(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    by_policy_case = {
        (str(row["policy_id"]), str(row["case_id"])): row for row in rows
    }
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        intervention = row["intervention"]
        if not bool(intervention.get("active")) or intervention.get("mode") != "switch":
            continue
        control_id = str(row["paired_control_id"])
        control = by_policy_case[(str(row["policy_id"]), control_id)]
        pre_target = str(row["pre_target_action"])
        effect = _probability_of(control, pre_target) - _probability_of(row, pre_target)
        key = f"{intervention['objective']}_{intervention['base_family']}"
        values[str(row["policy_id"])][key].append(effect)
    output: dict[str, dict[str, float]] = {}
    for policy_id, dimensions in values.items():
        output[policy_id] = {
            key: float(np.mean(dimensions.get(key, [float("nan")])))
            for key in FINGERPRINT_ORDER
        }
    return output


def _max_control_effect(rows: Sequence[Mapping[str, Any]], mode: str) -> float:
    by_policy_case = {
        (str(row["policy_id"]), str(row["case_id"])): row for row in rows
    }
    effects: list[float] = []
    for row in rows:
        intervention = row["intervention"]
        if not bool(intervention.get("active")) or intervention.get("mode") != mode:
            continue
        control = by_policy_case[(str(row["policy_id"]), str(row["paired_control_id"]))]
        pre_target = str(row["pre_target_action"])
        effects.append(abs(_probability_of(control, pre_target) - _probability_of(row, pre_target)))
    return max(effects, default=0.0)


def _convex_residual(vector: Sequence[float], prototypes: np.ndarray) -> float:
    target = np.asarray(vector, dtype=float)
    count = prototypes.shape[0]

    def objective(weights: np.ndarray) -> float:
        residual = weights @ prototypes - target
        return float(residual @ residual)

    result = minimize(
        objective,
        np.full(count, 1.0 / count),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * count,
        constraints={"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)},
        options={"ftol": 1e-14, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Oracle convex-hull calibration failed: {result.message}")
    return math.sqrt(max(float(result.fun), 0.0))


def run_bridge_oracles(
    config: dict[str, Any],
    split: str,
    destination: str | Path | None = None,
    *,
    data_dir: str | Path | None = None,
) -> Path:
    """Write the CPU 2x3 controls, mixtures, and open-set calibration output."""
    if split not in {"dev", "test"}:
        raise ValueError("Bridge oracle split must be dev or test")
    environment = load_bridge_environment(
        config, data_dir=data_dir, allowed_splits=(split,)
    )
    cases = list(environment.extinction_cases(split=split, trajectory_seed=0, checkpoint_update=0))
    target = (
        Path(destination).resolve()
        if destination
        else output_root(config) / "predictions" / f"bridge_oracle_{split}.jsonl"
    )
    summary_target = target.with_suffix(".summary.json")
    if target.exists() or summary_target.exists():
        raise FileExistsError(f"Refusing to overwrite bridge oracle output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    policies = _policy_specs()
    rows: list[dict[str, Any]] = []
    for policy in policies:
        for case in cases:
            probability_a = _policy_probability_a(case, policy)
            probability_b = 1.0 - probability_a
            predicted = "A" if probability_a >= 0.5 else "B"
            rows.append({
                "schema_version": "1.0",
                "evidence_kind": "bridge_oracle_apparatus_only",
                "config_sha256": config["_config_sha256"],
                "environment_provenance": dict(environment.provenance()),
                "split": split,
                "policy_id": policy["policy_id"],
                "policy_kind": policy["kind"],
                "policy_spec": dict(policy),
                "case_id": case["case_id"],
                "world_id": case["world_id"],
                "renderer_id": case["renderer_id"],
                "cue_regime": case["cue_regime"],
                "condition": case["condition"],
                "pair_id": case["pair_id"],
                "paired_control_id": case.get("paired_control_id"),
                "baseline_id": case.get("baseline_id"),
                "pre_target_action": case["pre_target_action"],
                "intervention": dict(case["intervention"]),
                "expected_actions": dict(case["expected_actions"]),
                "probability_A": probability_a,
                "probability_B": probability_b,
                "logp_A": math.log(probability_a),
                "logp_B": math.log(probability_b),
                "legal_choice_mass": 1.0,
                "predicted_action": predicted,
                "extinction_protocol": dict(case["extinction_protocol"]),
            })
    fingerprints = _fingerprints(rows)
    pure_ids = [str(policy["policy_id"]) for policy in policies if policy["kind"] == "pure"]
    pure_matrix = np.asarray(
        [[fingerprints[policy_id][key] for key in FINGERPRINT_ORDER] for policy_id in pure_ids],
        dtype=float,
    )
    calibration: dict[str, dict[str, Any]] = {}
    for policy in policies:
        policy_id = str(policy["policy_id"])
        vector = [fingerprints[policy_id][key] for key in FINGERPRINT_ORDER]
        residual = _convex_residual(vector, pure_matrix)
        calibration[policy_id] = {
            "fingerprint": dict(fingerprints[policy_id]),
            "distance_to_pure_prototype_convex_hull": residual,
            "open_set": residual > 0.05,
        }
    theoretical_scale = 0.96
    theoretical = {
        "pure_genuine_model_based": [theoretical_scale, 0.0, theoretical_scale, 0.0],
        "pure_proxy_model_based": [0.0, theoretical_scale, 0.0, theoretical_scale],
        "pure_genuine_successor_representation": [theoretical_scale, 0.0, 0.0, 0.0],
        "pure_proxy_successor_representation": [0.0, theoretical_scale, 0.0, 0.0],
        "pure_genuine_habit": [0.0, 0.0, 0.0, 0.0],
        "pure_proxy_habit": [0.0, 0.0, 0.0, 0.0],
    }
    recovery_errors = {
        policy_id: max(
            abs(fingerprints[policy_id][key] - theoretical[policy_id][index])
            for index, key in enumerate(FINGERPRINT_ORDER)
        )
        for policy_id in theoretical
    }
    summary = {
        "schema_version": "1.0",
        "kind": "bridge_oracle_apparatus_report",
        "evidence_scope": "code_and_assay_validation_only_not_model_evidence",
        "config_sha256": config["_config_sha256"],
        "environment_provenance": dict(environment.provenance()),
        "split": split,
        "case_count": len(cases),
        "policy_count": len(policies),
        "fingerprint_order": list(FINGERPRINT_ORDER),
        "calibration": calibration,
        "pure_recovery_max_abs_errors": recovery_errors,
        "prototype_collision": {
            "members": ["pure_genuine_habit", "pure_proxy_habit"],
            "interpretation": "A fully cached habit cannot behaviorally reveal which reward created it.",
        },
        "checks": {
            "all_six_pure_theoretical_fingerprints_recovered": max(recovery_errors.values()) < 1e-9,
            "mixtures_inside_calibrated_convex_hull": all(
                not calibration[str(policy["policy_id"])]["open_set"]
                for policy in policies if policy["kind"] == "mixture"
            ),
            "open_set_policy_rejected": calibration["open_set_anti_revaluation"]["open_set"],
            "pure_max_no_switch_effect": _max_control_effect(
                [row for row in rows if row["policy_kind"] == "pure"], "no_switch"
            ) < 1e-9,
        },
        "predictions_sha256": hashlib.sha256(
            "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")
        ).hexdigest(),
    }
    write_jsonl(target, rows)
    write_json(summary_target, summary)
    return target
