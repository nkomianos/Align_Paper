"""Full-learning design and endpoint rules for the repaired EndoPAHF G2 gate."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math
import random
from typing import Mapping, Sequence

import numpy as np

from interaction_sprint.hindsight_pahf_cluster_stats import BOOTSTRAP_SAMPLES
from interaction_sprint.hindsight_pahf_g2 import (
    arm_diagnostics,
    paired_target_summary,
    score_prediction_rows,
)
from interaction_sprint.hindsight_pahf_interface import OPTION_LETTERS


G2_SEED = 2026090432
G2_STEPS = 35
G2_BATCH = 18
G2_PANEL_COUNT = 4
G2_ANCHOR_BASES_PER_PANEL = 16
G2_ANCHOR_BASES_PER_STEP = 2
G2_ANCHOR_ROWS_PER_STEP = 8
G2_LEARNING_RATE = 1e-4
G2_LEARNING_BASES = 630
DEV_MIN_NLL_GAIN = 0.03
CONFIRM_MIN_NLL_GAIN = 0.05
ROTATIONS = {0, 1, 2, 3}


def _hash_rank(value: str, salt: str) -> str:
    return hashlib.sha256(f"{value}|{salt}".encode("utf-8")).hexdigest()


def _group_bases(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, list[Mapping[str, object]]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    ids: set[str] = set()
    for row in rows:
        row_id = str(row["id"])
        if row_id in ids:
            raise ValueError("duplicate row id")
        ids.add(row_id)
        if str(row["immediate_followup"]) != str(row["delayed_transition_followup"]):
            raise ValueError(f"transition probe differs from immediate log: {row_id}")
        if str(row["immediate_followup"]) == str(row["delayed_expression_followup"]):
            raise ValueError(f"expression and transition probes are not opposed: {row_id}")
        grouped[str(row["base_id"])].append(row)
    if len(grouped) != G2_LEARNING_BASES:
        raise ValueError(
            f"G2 v2 requires {G2_LEARNING_BASES} learning bases, got {len(grouped)}"
        )
    for base_id, variants in grouped.items():
        rotations = {int(row["label_rotation"]) for row in variants}
        if rotations != ROTATIONS or len(variants) != 4:
            raise ValueError(f"base does not have four rotations: {base_id}")
        if {str(row["old_target"]) for row in variants} != set(OPTION_LETTERS):
            raise ValueError(f"old targets not counterbalanced: {base_id}")
        if {str(row["new_target"]) for row in variants} != set(OPTION_LETTERS):
            raise ValueError(f"new targets not counterbalanced: {base_id}")
        if any(row["old_target"] == row["new_target"] for row in variants):
            raise ValueError(f"unchanged target in base: {base_id}")
    return dict(grouped)


def build_g2_anchor_panels(rows: Sequence[Mapping[str, object]]) -> list[list[str]]:
    """Choose four disjoint, outcome-blind panels totaling 64 base tasks."""
    grouped = _group_bases(rows)
    needed = G2_PANEL_COUNT * G2_ANCHOR_BASES_PER_PANEL
    ordered = sorted(
        grouped, key=lambda base_id: _hash_rank(base_id, "endo-pahf-g2-panels-v2")
    )
    panels = [
        ordered[start:start + G2_ANCHOR_BASES_PER_PANEL]
        for start in range(0, needed, G2_ANCHOR_BASES_PER_PANEL)
    ]
    flat = [base_id for panel in panels for base_id in panel]
    if len(panels) != G2_PANEL_COUNT or len(flat) != len(set(flat)):
        raise AssertionError("invalid G2 v2 panels")
    return panels


def _one_balanced_variant_per_base(
    grouped: Mapping[str, Sequence[Mapping[str, object]]],
) -> list[str]:
    ordered = sorted(
        grouped, key=lambda base_id: _hash_rank(base_id, "endo-pahf-g2-global-v2")
    )
    selected: list[str] = []
    rotation_counts = {rotation: 0 for rotation in ROTATIONS}
    for index, base_id in enumerate(ordered):
        rotation = index % 4
        matches = [
            row for row in grouped[base_id] if int(row["label_rotation"]) == rotation
        ]
        if len(matches) != 1:
            raise AssertionError("balanced variant lookup failed")
        selected.append(str(matches[0]["id"]))
        rotation_counts[rotation] += 1
    if max(rotation_counts.values()) - min(rotation_counts.values()) > 1:
        raise AssertionError("global rotation assignment is not balanced")
    random.Random(G2_SEED).shuffle(selected)
    return selected


def build_g2_schedules(
    rows: Sequence[Mapping[str, object]], panels: Sequence[Sequence[str]],
) -> dict[str, object]:
    """Cover all 630 bases once globally and repeat small four-rotation anchors."""
    grouped = _group_bases(rows)
    if len(rows) != 4 * G2_LEARNING_BASES:
        raise ValueError("G2 v2 requires 2,520 counterbalanced learning rows")
    if len(panels) != G2_PANEL_COUNT:
        raise ValueError("wrong panel count")
    global_ids = _one_balanced_variant_per_base(grouped)
    if len(global_ids) != G2_STEPS * G2_BATCH:
        raise AssertionError("global schedule dimensions do not cover all bases")
    global_schedule = [
        global_ids[start:start + G2_BATCH]
        for start in range(0, len(global_ids), G2_BATCH)
    ]
    panel_schedules: dict[str, dict[str, object]] = {}
    for panel_index, panel_values in enumerate(panels):
        panel = list(panel_values)
        if len(panel) != G2_ANCHOR_BASES_PER_PANEL or not set(panel) <= set(grouped):
            raise ValueError("invalid G2 v2 panel")
        anchor_schedule: list[list[str]] = []
        cycle = 0
        while len(anchor_schedule) < G2_STEPS:
            ordered_bases = list(panel)
            random.Random(G2_SEED + 1000 * panel_index + cycle).shuffle(ordered_bases)
            for start in range(0, len(ordered_bases), G2_ANCHOR_BASES_PER_STEP):
                base_pair = ordered_bases[start:start + G2_ANCHOR_BASES_PER_STEP]
                batch = [
                    str(row["id"])
                    for base_id in base_pair
                    for row in sorted(
                        grouped[base_id], key=lambda value: int(value["label_rotation"])
                    )
                ]
                if len(batch) != G2_ANCHOR_ROWS_PER_STEP:
                    raise AssertionError("invalid anchor batch")
                anchor_schedule.append(batch)
                if len(anchor_schedule) == G2_STEPS:
                    break
            cycle += 1
        panel_schedules[str(panel_index)] = {
            "base_ids": panel,
            "anchor_schedule": anchor_schedule,
        }
    return {
        "global": global_schedule,
        "global_unique_bases": G2_LEARNING_BASES,
        "global_variants_per_base": 1,
        "panels": panel_schedules,
    }


def expected_g2_arm_names() -> list[str]:
    names = ["baseline", "raw_immediate", "oracle_delayed", "transition_sanity"]
    for panel_index in range(G2_PANEL_COUNT):
        names.extend([
            f"panel_{panel_index:02d}_anchor_sdpo",
            f"panel_{panel_index:02d}_anchor_sft",
            f"panel_{panel_index:02d}_augmented",
        ])
    return names


def g2_development_routing_power_audit(*, trials: int = 10_000) -> dict[str, object]:
    if trials < 1000:
        raise ValueError("too few routing-power trials")
    scenarios = (
        ("null", 0.00, 0.35),
        ("signal", 0.08, 0.35),
        ("noisy_signal", 0.08, 0.50),
    )
    rates: dict[str, float] = {}
    for index, (name, mean_gain, standard_deviation) in enumerate(scenarios):
        rng = np.random.default_rng(G2_SEED + 10_000 + index)
        samples = rng.normal(mean_gain, standard_deviation, size=(trials, 96))
        rates[name] = float(np.mean(samples.mean(axis=1) >= DEV_MIN_NLL_GAIN))
    gates = {
        "null_routing_rate_at_most_point25": rates["null"] <= .25,
        "signal_routing_power_at_least_point90": rates["signal"] >= .90,
        "noisy_signal_routing_power_at_least_point75": rates["noisy_signal"] >= .75,
    }
    return {
        "decision": (
            "ENDO_PAHF_G2_V2_DEV_ROUTING_RULE_POWER_QUALIFIED"
            if all(gates.values())
            else "ENDO_PAHF_G2_V2_DEV_ROUTING_RULE_POWER_NOT_QUALIFIED"
        ),
        "design": {
            "base_clusters": 96,
            "trials": trials,
            "minimum_mean_nll_gain": DEV_MIN_NLL_GAIN,
            "confidence_interval_required": False,
            "scenarios": [
                {"name": name, "mean_gain": mean, "standard_deviation": sd}
                for name, mean, sd in scenarios
            ],
        },
        "routing_rates": rates,
        "gates": gates,
        "scope": "Development routing only; final confirmation uses clustered inference.",
        "paper_green_light": False,
    }


def _average_panel_predictions(
    panel_rows: Sequence[Sequence[Mapping[str, object]]],
) -> list[dict[str, object]]:
    if len(panel_rows) != G2_PANEL_COUNT:
        raise ValueError("all four panels are required")
    by_panel = [{str(row["id"]): row for row in rows} for rows in panel_rows]
    ids = set(by_panel[0])
    if any(set(panel) != ids or len(panel) != len(ids) for panel in by_panel):
        raise ValueError("panel prediction grids differ")
    result = []
    for row_id in sorted(ids):
        probabilities = np.asarray([
            np.exp(np.asarray(panel[row_id]["normalized_choice_log_probabilities"], dtype=np.float64))
            for panel in by_panel
        ])
        averaged = probabilities.mean(axis=0)
        averaged /= averaged.sum()
        masses = [float(panel[row_id]["full_vocabulary_choice_mass"]) for panel in by_panel]
        result.append({
            "id": row_id,
            "normalized_choice_log_probabilities": np.log(averaged).tolist(),
            "full_vocabulary_choice_mass": float(np.mean(masses)),
        })
    return result


def _transition_difference(
    raw: Sequence[Mapping[str, object]], sanity: Sequence[Mapping[str, object]],
) -> float:
    raw_by_id = {str(row["id"]): row for row in raw}
    sanity_by_id = {str(row["id"]): row for row in sanity}
    if set(raw_by_id) != set(sanity_by_id):
        raise ValueError("transition sanity grid mismatch")
    return max(
        abs(
            math.exp(float(raw_by_id[row_id]["normalized_choice_log_probabilities"][index]))
            - math.exp(float(sanity_by_id[row_id]["normalized_choice_log_probabilities"][index]))
        )
        for row_id in raw_by_id for index in range(4)
    )


def summarize_g2_stage(
    source_rows: Sequence[Mapping[str, object]],
    arm_predictions: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    stage: str,
    transition_adapter_max_abs_difference: float,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, object]:
    if stage not in {"development", "confirmation"}:
        raise ValueError("invalid G2 stage")
    if set(arm_predictions) != set(expected_g2_arm_names()):
        raise ValueError("G2 v2 arm grid differs from frozen design")
    scored = {
        name: score_prediction_rows(source_rows, rows)
        for name, rows in arm_predictions.items()
    }
    ensembles = {}
    for method in ("anchor_sdpo", "anchor_sft", "augmented"):
        predictions = _average_panel_predictions([
            arm_predictions[f"panel_{panel:02d}_{method}"]
            for panel in range(G2_PANEL_COUNT)
        ])
        ensembles[method] = score_prediction_rows(source_rows, predictions)
    minimum = DEV_MIN_NLL_GAIN if stage == "development" else CONFIRM_MIN_NLL_GAIN
    require_ci = stage == "confirmation"
    comparisons = {
        "raw_new_vs_baseline": paired_target_summary(
            scored["baseline"], scored["raw_immediate"], target="new",
            minimum_nll_gain=minimum, require_positive_ci=require_ci,
            samples=bootstrap_samples, seed=G2_SEED + 1,
        ),
        "oracle_old_vs_baseline": paired_target_summary(
            scored["baseline"], scored["oracle_delayed"], target="old",
            minimum_nll_gain=minimum, require_positive_ci=require_ci,
            samples=bootstrap_samples, seed=G2_SEED + 2,
        ),
        "oracle_old_vs_raw": paired_target_summary(
            scored["raw_immediate"], scored["oracle_delayed"], target="old",
            minimum_nll_gain=minimum, require_positive_ci=require_ci,
            samples=bootstrap_samples, seed=G2_SEED + 3,
        ),
        "raw_new_vs_oracle": paired_target_summary(
            scored["oracle_delayed"], scored["raw_immediate"], target="new",
            minimum_nll_gain=minimum, require_positive_ci=require_ci,
            samples=bootstrap_samples, seed=G2_SEED + 4,
        ),
        "augmented_old_vs_anchor_sdpo": paired_target_summary(
            ensembles["anchor_sdpo"], ensembles["augmented"], target="old",
            minimum_nll_gain=minimum, require_positive_ci=require_ci,
            samples=bootstrap_samples, seed=G2_SEED + 5,
        ),
        "augmented_old_vs_anchor_sft": paired_target_summary(
            ensembles["anchor_sft"], ensembles["augmented"], target="old",
            minimum_nll_gain=minimum, require_positive_ci=require_ci,
            samples=bootstrap_samples, seed=G2_SEED + 6,
        ),
        "augmented_old_vs_raw": paired_target_summary(
            scored["raw_immediate"], ensembles["augmented"], target="old",
            minimum_nll_gain=minimum, require_positive_ci=require_ci,
            samples=bootstrap_samples, seed=G2_SEED + 7,
        ),
    }
    diagnostics = {
        name: arm_diagnostics(rows)
        for name, rows in {
            "baseline": scored["baseline"],
            "raw_immediate": scored["raw_immediate"],
            "oracle_delayed": scored["oracle_delayed"],
            **{f"ensemble_{name}": values for name, values in ensembles.items()},
        }.items()
    }
    transition_max_difference = _transition_difference(
        scored["raw_immediate"], scored["transition_sanity"]
    )
    if (
        not math.isfinite(transition_adapter_max_abs_difference)
        or transition_adapter_max_abs_difference < 0
    ):
        raise ValueError("invalid transition adapter difference")
    controls = {
        "all_primary_comparisons_qualified": all(
            value["qualified"] for value in comparisons.values()
        ),
        "transition_augmented_matches_raw": transition_max_difference <= 1e-6,
        "transition_adapter_matches_raw": transition_adapter_max_abs_difference <= 1e-6,
        "all_reported_arms_preserve_choice_mass": all(
            float(value["minimum_label_mean_choice_mass"]) >= .05
            for value in diagnostics.values()
        ),
        "panel_ensembles_not_position_dominated": all(
            float(diagnostics[f"ensemble_{name}"]["old_accuracy_label_gap"]) <= .20
            for name in ("anchor_sdpo", "anchor_sft", "augmented")
        ),
    }
    decision = (
        "ENDO_PAHF_G2_V2_DEV_QUALIFIED"
        if stage == "development" and all(controls.values())
        else "ENDO_PAHF_G2_V2_DEV_NOT_QUALIFIED"
        if stage == "development"
        else "ENDO_PAHF_G2_V2_CONFIRMATION_QUALIFIED"
        if all(controls.values())
        else "ENDO_PAHF_G2_V2_CONFIRMATION_NOT_QUALIFIED"
    )
    return {
        "decision": decision,
        "stage": stage,
        "controls": controls,
        "comparisons": comparisons,
        "diagnostics": diagnostics,
        "transition_sanity_max_probability_difference": transition_max_difference,
        "transition_adapter_max_abs_difference": transition_adapter_max_abs_difference,
        "primary_estimand": (
            "old persistent-target normalized choice NLL, averaged across rotations "
            "within base task and then paired across base tasks"
        ),
        "panel_aggregation": "arithmetic mean of four panel probability vectors",
        "confirmation_opened": stage == "confirmation",
        "paper_green_light": False,
    }
