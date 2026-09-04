"""Frozen design and endpoint rules for the EndoPAHF neural transfer gate."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math
import random
from typing import Mapping, Sequence

import numpy as np

from interaction_sprint.hindsight_pahf_cluster_stats import (
    ACCURACY_NONINFERIORITY,
    BOOTSTRAP_SAMPLES,
    bootstrap_mean_interval,
)
from interaction_sprint.hindsight_pahf_interface import OPTION_LETTERS


G2_SEED = 2026090431
G2_STEPS = 32
G2_BATCH = 16
G2_PANEL_COUNT = 8
G2_ANCHOR_BASES_PER_PANEL = 8
G2_ANCHOR_BASES_PER_STEP = 2
G2_ANCHOR_ROWS_PER_STEP = 8
G2_LEARNING_RATE = 1e-4
DEV_MIN_NLL_GAIN = 0.03
CONFIRM_MIN_NLL_GAIN = 0.05
ROTATIONS = {0, 1, 2, 3}


def _hash_rank(value: str, salt: str) -> str:
    return hashlib.sha256(f"{value}|{salt}".encode("utf-8")).hexdigest()


def _group_bases(rows: Sequence[Mapping[str, object]]) -> dict[str, list[Mapping[str, object]]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    ids: set[str] = set()
    for row in rows:
        row_id = str(row["id"])
        if row_id in ids:
            raise ValueError("duplicate row id")
        ids.add(row_id)
        grouped[str(row["base_id"])].append(row)
    if not grouped:
        raise ValueError("no base clusters")
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
    """Choose eight disjoint outcome-blind panels of eight base tasks."""
    grouped = _group_bases(rows)
    needed = G2_PANEL_COUNT * G2_ANCHOR_BASES_PER_PANEL
    if len(grouped) < needed:
        raise ValueError("too few base clusters for disjoint panels")
    ordered = sorted(grouped, key=lambda base_id: _hash_rank(base_id, "endo-pahf-g2-panels-v1"))
    panels = [
        ordered[start:start + G2_ANCHOR_BASES_PER_PANEL]
        for start in range(0, needed, G2_ANCHOR_BASES_PER_PANEL)
    ]
    flat = [base_id for panel in panels for base_id in panel]
    if len(panels) != G2_PANEL_COUNT or len(flat) != len(set(flat)):
        raise AssertionError("invalid G2 panels")
    return panels


def build_g2_schedules(
    rows: Sequence[Mapping[str, object]], panels: Sequence[Sequence[str]],
) -> dict[str, object]:
    """Create a one-epoch global schedule and balanced repeated anchor schedules."""
    grouped = _group_bases(rows)
    if len(rows) != G2_STEPS * G2_BATCH:
        raise ValueError("G2 global schedule requires exactly 512 variants")
    if len(panels) != G2_PANEL_COUNT:
        raise ValueError("wrong panel count")
    global_ids = [str(row["id"]) for row in rows]
    random.Random(G2_SEED).shuffle(global_ids)
    global_schedule = [
        global_ids[start:start + G2_BATCH]
        for start in range(0, len(global_ids), G2_BATCH)
    ]
    panel_schedules: dict[str, dict[str, object]] = {}
    for panel_index, panel_values in enumerate(panels):
        panel = list(panel_values)
        if len(panel) != G2_ANCHOR_BASES_PER_PANEL or not set(panel) <= set(grouped):
            raise ValueError("invalid G2 panel")
        anchor_schedule: list[list[str]] = []
        for cycle in range(G2_STEPS // (G2_ANCHOR_BASES_PER_PANEL // G2_ANCHOR_BASES_PER_STEP)):
            ordered_bases = list(panel)
            random.Random(G2_SEED + 1000 * panel_index + cycle).shuffle(ordered_bases)
            for start in range(0, len(ordered_bases), G2_ANCHOR_BASES_PER_STEP):
                base_pair = ordered_bases[start:start + G2_ANCHOR_BASES_PER_STEP]
                batch = [
                    str(row["id"])
                    for base_id in base_pair
                    for row in sorted(grouped[base_id], key=lambda value: int(value["label_rotation"]))
                ]
                if len(batch) != G2_ANCHOR_ROWS_PER_STEP:
                    raise AssertionError("invalid anchor batch")
                anchor_schedule.append(batch)
        if len(anchor_schedule) != G2_STEPS:
            raise AssertionError("invalid anchor schedule length")
        panel_schedules[str(panel_index)] = {
            "base_ids": panel,
            "anchor_schedule": anchor_schedule,
        }
    return {"global": global_schedule, "panels": panel_schedules}


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
    """Audit the permissive 96-base DEV routing rule, not final inference."""
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
            "ENDO_PAHF_G2_DEV_ROUTING_RULE_POWER_QUALIFIED"
            if all(gates.values()) else "ENDO_PAHF_G2_DEV_ROUTING_RULE_POWER_NOT_QUALIFIED"
        ),
        "design": {
            "base_clusters": 96,
            "trials": trials,
            "minimum_mean_nll_gain": DEV_MIN_NLL_GAIN,
            "confidence_interval_required": False,
            "scenarios": [
                {"name": name, "mean_gain": mean_gain, "standard_deviation": standard_deviation}
                for name, mean_gain, standard_deviation in scenarios
            ],
        },
        "routing_rates": rates,
        "gates": gates,
        "scope": "Development routing only; final confirmation uses clustered inference.",
        "paper_green_light": False,
    }


def score_prediction_rows(
    source_rows: Sequence[Mapping[str, object]],
    predictions: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Validate four-way predictions and attach old/new target metrics."""
    _group_bases(source_rows)
    source_by_id = {str(row["id"]): row for row in source_rows}
    prediction_by_id = {str(row["id"]): row for row in predictions}
    if len(prediction_by_id) != len(predictions) or set(prediction_by_id) != set(source_by_id):
        raise ValueError("prediction grid does not match source rows")
    scored: list[dict[str, object]] = []
    for row_id in sorted(source_by_id):
        source = source_by_id[row_id]
        prediction = prediction_by_id[row_id]
        log_values = [float(value) for value in prediction["normalized_choice_log_probabilities"]]  # type: ignore[index]
        mass = float(prediction["full_vocabulary_choice_mass"])
        if (
            len(log_values) != 4
            or any(not math.isfinite(value) or value > 1e-7 for value in log_values)
            or abs(sum(math.exp(value) for value in log_values) - 1.0) > 1e-5
            or not math.isfinite(mass) or not 0 <= mass <= 1 + 1e-6
        ):
            raise ValueError(f"invalid prediction: {row_id}")
        predicted = max(range(4), key=log_values.__getitem__)
        old_index = OPTION_LETTERS.index(str(source["old_target"]))
        new_index = OPTION_LETTERS.index(str(source["new_target"]))
        scored.append({
            "id": row_id,
            "base_id": str(source["base_id"]),
            "rotation": int(source["label_rotation"]),
            "old_target": str(source["old_target"]),
            "new_target": str(source["new_target"]),
            "normalized_choice_log_probabilities": log_values,
            "full_vocabulary_choice_mass": mass,
            "old_target_log_loss": -log_values[old_index],
            "new_target_log_loss": -log_values[new_index],
            "old_target_correct": predicted == old_index,
            "new_target_correct": predicted == new_index,
        })
    return scored


def average_panel_predictions(
    panel_rows: Sequence[Sequence[Mapping[str, object]]],
) -> list[dict[str, object]]:
    """Average panel probabilities prospectively; never select a best panel."""
    if len(panel_rows) != G2_PANEL_COUNT:
        raise ValueError("all eight panels are required")
    by_panel = [{str(row["id"]): row for row in rows} for rows in panel_rows]
    ids = set(by_panel[0])
    if any(set(panel) != ids or len(panel) != len(ids) for panel in by_panel):
        raise ValueError("panel prediction grids differ")
    result: list[dict[str, object]] = []
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


def _base_metrics(rows: Sequence[Mapping[str, object]], target: str) -> dict[str, dict[str, float]]:
    if target not in {"old", "new"}:
        raise ValueError("target must be old or new")
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["base_id"])].append(row)
    result: dict[str, dict[str, float]] = {}
    for base_id, variants in grouped.items():
        if {int(row["rotation"]) for row in variants} != ROTATIONS or len(variants) != 4:
            raise ValueError("scored base lacks rotations")
        result[base_id] = {
            "nll": float(np.mean([float(row[f"{target}_target_log_loss"]) for row in variants])),
            "accuracy": float(np.mean([float(bool(row[f"{target}_target_correct"])) for row in variants])),
        }
    return result


def paired_target_summary(
    baseline: Sequence[Mapping[str, object]],
    candidate: Sequence[Mapping[str, object]],
    *,
    target: str,
    minimum_nll_gain: float,
    require_positive_ci: bool,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = G2_SEED,
) -> dict[str, object]:
    baseline_base = _base_metrics(baseline, target)
    candidate_base = _base_metrics(candidate, target)
    if set(baseline_base) != set(candidate_base):
        raise ValueError("paired base grids differ")
    ids = sorted(baseline_base)
    nll_gain = [baseline_base[key]["nll"] - candidate_base[key]["nll"] for key in ids]
    accuracy_gain = [
        candidate_base[key]["accuracy"] - baseline_base[key]["accuracy"] for key in ids
    ]
    nll_ci = bootstrap_mean_interval(nll_gain, samples=samples, seed=seed)
    accuracy_ci = bootstrap_mean_interval(accuracy_gain, samples=samples, seed=seed + 1)
    aggregate = {
        "base_clusters": len(ids),
        "mean_nll_gain": float(np.mean(nll_gain)),
        "nll_gain_ci95": list(nll_ci),
        "mean_accuracy_gain": float(np.mean(accuracy_gain)),
        "accuracy_gain_ci95": list(accuracy_ci),
        "minimum_nll_gain": minimum_nll_gain,
        "require_positive_ci": require_positive_ci,
        "accuracy_noninferiority_margin": ACCURACY_NONINFERIORITY,
    }
    gates = {
        "mean_nll_gain_reaches_minimum": aggregate["mean_nll_gain"] >= minimum_nll_gain,
        "nll_gain_ci_requirement": (not require_positive_ci) or nll_ci[0] > 0,
        "mean_accuracy_noninferior": aggregate["mean_accuracy_gain"] >= ACCURACY_NONINFERIORITY,
    }
    return {"qualified": all(gates.values()), "gates": gates, "aggregate": aggregate}


def arm_diagnostics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("empty scored rows")
    per_label = {}
    for label in OPTION_LETTERS:
        selected = [row for row in rows if row["old_target"] == label]
        if not selected:
            raise ValueError("missing old-target label")
        per_label[label] = {
            "n": len(selected),
            "old_accuracy": float(np.mean([bool(row["old_target_correct"]) for row in selected])),
            "new_accuracy": float(np.mean([bool(row["new_target_correct"]) for row in selected])),
            "mean_choice_mass": float(np.mean([float(row["full_vocabulary_choice_mass"]) for row in selected])),
        }
    return {
        "old_target_nll": float(np.mean([float(row["old_target_log_loss"]) for row in rows])),
        "new_target_nll": float(np.mean([float(row["new_target_log_loss"]) for row in rows])),
        "old_target_accuracy": float(np.mean([bool(row["old_target_correct"]) for row in rows])),
        "new_target_accuracy": float(np.mean([bool(row["new_target_correct"]) for row in rows])),
        "minimum_label_mean_choice_mass": min(value["mean_choice_mass"] for value in per_label.values()),
        "old_accuracy_label_gap": max(value["old_accuracy"] for value in per_label.values())
        - min(value["old_accuracy"] for value in per_label.values()),
        "per_old_target_label": per_label,
    }


def _transition_difference(raw: Sequence[Mapping[str, object]], sanity: Sequence[Mapping[str, object]]) -> float:
    raw_by_id = {str(row["id"]): row for row in raw}
    sanity_by_id = {str(row["id"]): row for row in sanity}
    if set(raw_by_id) != set(sanity_by_id):
        raise ValueError("transition sanity grid mismatch")
    return max(
        abs(math.exp(float(raw_by_id[row_id]["normalized_choice_log_probabilities"][index]))
            - math.exp(float(sanity_by_id[row_id]["normalized_choice_log_probabilities"][index])))
        for row_id in raw_by_id for index in range(4)
    )


def summarize_g2_stage(
    source_rows: Sequence[Mapping[str, object]],
    arm_predictions: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    stage: str,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, object]:
    """Apply the fixed DEV or confirmation rule to complete arm predictions."""
    if stage not in {"development", "confirmation"}:
        raise ValueError("invalid G2 stage")
    if set(arm_predictions) != set(expected_g2_arm_names()):
        raise ValueError("G2 arm grid differs from frozen design")
    scored = {
        name: score_prediction_rows(source_rows, rows)
        for name, rows in arm_predictions.items()
    }
    ensembles = {}
    for method in ("anchor_sdpo", "anchor_sft", "augmented"):
        predictions = average_panel_predictions([
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
    controls = {
        "all_primary_comparisons_qualified": all(
            value["qualified"] for value in comparisons.values()
        ),
        "transition_augmented_matches_raw": transition_max_difference <= 1e-6,
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
        "ENDO_PAHF_G2_DEV_QUALIFIED" if stage == "development" and all(controls.values())
        else "ENDO_PAHF_G2_DEV_NOT_QUALIFIED" if stage == "development"
        else "ENDO_PAHF_G2_CONFIRMATION_QUALIFIED" if all(controls.values())
        else "ENDO_PAHF_G2_CONFIRMATION_NOT_QUALIFIED"
    )
    return {
        "decision": decision,
        "stage": stage,
        "controls": controls,
        "comparisons": comparisons,
        "diagnostics": diagnostics,
        "transition_sanity_max_probability_difference": transition_max_difference,
        "primary_estimand": (
            "old persistent-target normalized choice NLL, averaged across rotations "
            "within base task and then paired across base tasks"
        ),
        "panel_aggregation": "arithmetic mean of eight panel probability vectors",
        "confirmation_opened": stage == "confirmation",
        "paper_green_light": False,
    }
