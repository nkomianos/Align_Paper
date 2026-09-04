"""Prospective model-free power audit for the neural policy G1 decision rule."""
from __future__ import annotations

import math
from statistics import mean

from interaction_sprint.hindsight_neural_policy_g1 import (
    POLICY_PANEL_COUNT,
    expected_policy_arm_names,
    summarize_policy_endpoints,
)


EFFECTIVE_GAINS = (1.8, 2.5, 3.5)
SFT_GAIN_MULTIPLIERS = (.75, 1., 1.25)


def target_means(
    rows: list[dict[str, object]], panels: list[list[str]],
) -> dict[str, object]:
    """Compute exact raw, oracle, sparse, and paired population targets."""
    by_id = {str(row["id"]): row for row in rows}
    if len(by_id) != len(rows) or len(panels) != POLICY_PANEL_COUNT:
        raise ValueError("invalid rows or panels")
    raw = mean(float(row["immediate_semantic"]) for row in rows)
    oracle = mean(float(row["delayed_expression_semantic"]) for row in rows)
    panel_rows = []
    for index, panel in enumerate(panels):
        selected = [by_id[row_id] for row_id in panel]
        anchor = mean(float(row["delayed_expression_semantic"]) for row in selected)
        correction = mean(
            float(row["delayed_expression_semantic"]) - float(row["immediate_semantic"])
            for row in selected
        )
        panel_rows.append({
            "panel": index,
            "anchor_target": anchor,
            "paired_target": raw + correction,
            "oracle_target": oracle,
            "anchor_target_error": abs(anchor - oracle),
            "paired_target_error": abs(raw + correction - oracle),
        })
    return {"raw_target": raw, "oracle_target": oracle, "panels": panel_rows}


def _sigmoid(value: float) -> float:
    return 1. / (1. + math.exp(-value))


def _metric(probability: float) -> dict[str, float]:
    return {
        "semantic1_probability_mean": probability,
        "swap0_semantic1_probability_mean": probability - .01,
        "swap1_semantic1_probability_mean": probability + .01,
        "semantic_position_gap": .02,
        "min_ab_mass": .80,
    }


def surrogate_cell(
    targets: dict[str, object], effective_gain: float, sft_gain_multiplier: float,
    *, null_correction: bool = False,
) -> dict[str, object]:
    """Map exact label means through a monotone shared-logit learning surrogate."""
    if effective_gain <= 0 or sft_gain_multiplier <= 0:
        raise ValueError("gains must be positive")
    raw = float(targets["raw_target"])
    oracle = float(targets["oracle_target"])

    def endpoint(target: float, gain: float = effective_gain) -> float:
        return _sigmoid(gain * (2. * target - 1.))

    metrics = {
        "baseline": _metric(.5),
        "raw_immediate": _metric(endpoint(raw)),
        "oracle_delayed": _metric(endpoint(oracle)),
    }
    for row in targets["panels"]:  # type: ignore[assignment]
        panel = int(row["panel"])
        anchor_target = float(row["anchor_target"])
        paired_target = anchor_target if null_correction else float(row["paired_target"])
        metrics[f"panel_{panel:02d}_anchor_sdpo"] = _metric(endpoint(anchor_target))
        metrics[f"panel_{panel:02d}_anchor_sft"] = _metric(
            endpoint(anchor_target, effective_gain * sft_gain_multiplier)
        )
        metrics[f"panel_{panel:02d}_augmented"] = _metric(endpoint(paired_target))
    if list(metrics) != expected_policy_arm_names():
        raise AssertionError("surrogate arm order mismatch")
    summary = summarize_policy_endpoints(metrics)
    return {
        "effective_gain": effective_gain,
        "sft_gain_multiplier": sft_gain_multiplier,
        "null_correction": null_correction,
        "summary": summary,
    }


def build_power_cells(targets: dict[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    alternatives = [
        surrogate_cell(targets, gain, sft_multiplier)
        for gain in EFFECTIVE_GAINS
        for sft_multiplier in SFT_GAIN_MULTIPLIERS
    ]
    nulls = [
        surrogate_cell(targets, gain, sft_multiplier, null_correction=True)
        for gain in EFFECTIVE_GAINS
        for sft_multiplier in SFT_GAIN_MULTIPLIERS
    ]
    return alternatives, nulls


def summarize_power_cells(
    alternatives: list[dict[str, object]], nulls: list[dict[str, object]],
) -> dict[str, object]:
    expected_count = len(EFFECTIVE_GAINS) * len(SFT_GAIN_MULTIPLIERS)
    if len(alternatives) != expected_count or len(nulls) != expected_count:
        raise ValueError("power grid is incomplete")
    alternative_qualified = sum(
        cell["summary"]["decision"] == "NEURAL_POLICY_G1_QUALIFIED"  # type: ignore[index]
        for cell in alternatives
    )
    null_qualified = sum(
        cell["summary"]["decision"] == "NEURAL_POLICY_G1_QUALIFIED"  # type: ignore[index]
        for cell in nulls
    )
    control_qualified = sum(
        all(
            bool(value)
            for key, value in cell["summary"]["gates"].items()  # type: ignore[index]
            if key.startswith("oracle_") or key.startswith("raw_") or key.startswith("control_")
        )
        for cell in alternatives
    )
    false_direction_qualifications = sum(
        cell["summary"]["decision"] == "NEURAL_POLICY_G1_QUALIFIED"  # type: ignore[index]
        and cell["summary"]["aggregate"]["mean_sdpo_oracle_distance_gain"] <= 0  # type: ignore[index]
        for cell in alternatives
    )
    gates = {
        "all_surrogate_controls_qualify": control_qualified == expected_count,
        "at_least_six_of_nine_ideal_paired_cells_qualify": alternative_qualified >= 6,
        "no_null_correction_cell_qualifies": null_qualified == 0,
        "no_wrong_direction_cell_qualifies": false_direction_qualifications == 0,
    }
    return {
        "decision": "POLICY_G1_RULE_POWER_QUALIFIED" if all(gates.values()) else "POLICY_G1_RULE_POWER_NOT_QUALIFIED",
        "gates": gates,
        "counts": {
            "cells": expected_count,
            "control_qualified": control_qualified,
            "alternative_qualified": alternative_qualified,
            "null_qualified": null_qualified,
            "false_direction_qualifications": false_direction_qualifications,
        },
        "scope": (
            "Model-free monotone shared-logit calibration of the frozen G1 decision rule. "
            "It is protocol validation, not evidence that Qwen policy learning works."
        ),
    }
