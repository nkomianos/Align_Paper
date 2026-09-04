"""Pure design and decision rules for the Hindsight neural policy-learning G1."""
from __future__ import annotations

import hashlib
import math
import random
from statistics import mean, median


POLICY_SEED = 2026090413
POLICY_STEPS = 32
POLICY_BATCH = 16
POLICY_PANEL_COUNT = 8
POLICY_ANCHORS_PER_PANEL = 8
POLICY_ANCHORS_PER_ACTION = 4
POLICY_LEARNING_RATE = 1e-4


def _hash_rank(row_id: str, salt: str) -> str:
    return hashlib.sha256(f"{row_id}|{salt}".encode()).hexdigest()


def build_disjoint_policy_panels(rows: list[dict[str, object]]) -> list[list[str]]:
    """Return eight disjoint, outcome-blind panels balanced on logged action.

    Selection can use row identity and randomized logged action, but never the
    latent state or any immediate/delayed outcome.
    """
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("row ids must be unique")
    ordered: dict[int, list[str]] = {}
    needed = POLICY_PANEL_COUNT * POLICY_ANCHORS_PER_ACTION
    for action in (0, 1):
        candidates = [str(row["id"]) for row in rows if int(row["logged_action"]) == action]
        candidates.sort(key=lambda row_id: _hash_rank(row_id, "policy-panel-disjoint-v1"))
        if len(candidates) < needed:
            raise ValueError("too few records for disjoint action-balanced panels")
        ordered[action] = candidates
    panels: list[list[str]] = []
    for panel_index in range(POLICY_PANEL_COUNT):
        start = panel_index * POLICY_ANCHORS_PER_ACTION
        stop = start + POLICY_ANCHORS_PER_ACTION
        panel = ordered[0][start:stop] + ordered[1][start:stop]
        panels.append(panel)
    flat = [row_id for panel in panels for row_id in panel]
    if (
        any(len(panel) != POLICY_ANCHORS_PER_PANEL for panel in panels)
        or len(flat) != len(set(flat))
    ):
        raise AssertionError("invalid disjoint policy panels")
    return panels


def _cycled_action_batches(
    row_ids: list[str], *, per_step: int, steps: int, salt: str,
) -> list[list[str]]:
    if not row_ids or per_step < 1 or steps < 1:
        raise ValueError("invalid schedule inputs")
    rng = random.Random(int(hashlib.sha256(salt.encode()).hexdigest()[:16], 16))
    stream: list[str] = []
    while len(stream) < per_step * steps:
        cycle = list(row_ids)
        rng.shuffle(cycle)
        stream.extend(cycle)
    return [stream[start:start + per_step] for start in range(0, per_step * steps, per_step)]


def build_policy_schedules(
    rows: list[dict[str, object]], panels: list[list[str]],
) -> dict[str, object]:
    """Build fixed global and panel schedules shared by all paired learners."""
    by_id = {str(row["id"]): row for row in rows}
    if len(by_id) != len(rows) or len(panels) != POLICY_PANEL_COUNT:
        raise ValueError("rows or panels do not match the frozen design")
    known = set(by_id)
    if any(len(panel) != POLICY_ANCHORS_PER_PANEL or not set(panel) <= known for panel in panels):
        raise ValueError("invalid panel membership")

    global_by_action = {
        action: [str(row["id"]) for row in rows if int(row["logged_action"]) == action]
        for action in (0, 1)
    }
    global_parts = {
        action: _cycled_action_batches(
            global_by_action[action], per_step=POLICY_BATCH // 2, steps=POLICY_STEPS,
            salt=f"global-action-{action}-{POLICY_SEED}",
        )
        for action in (0, 1)
    }
    global_schedule: list[list[str]] = []
    for step in range(POLICY_STEPS):
        batch = global_parts[0][step] + global_parts[1][step]
        random.Random(POLICY_SEED + step).shuffle(batch)
        global_schedule.append(batch)

    panel_schedules: dict[str, dict[str, object]] = {}
    for panel_index, panel in enumerate(panels):
        panel_set = set(panel)
        if any(
            sum(int(by_id[row_id]["logged_action"]) == action for row_id in panel)
            != POLICY_ANCHORS_PER_ACTION
            for action in (0, 1)
        ):
            raise ValueError("panels must be balanced on logged action")
        ordinary_by_action = {
            action: [
                str(row["id"]) for row in rows
                if int(row["logged_action"]) == action and str(row["id"]) not in panel_set
            ]
            for action in (0, 1)
        }
        ordinary_parts = {
            action: _cycled_action_batches(
                ordinary_by_action[action], per_step=POLICY_ANCHORS_PER_ACTION,
                steps=POLICY_STEPS,
                salt=f"panel-{panel_index}-ordinary-action-{action}-{POLICY_SEED}",
            )
            for action in (0, 1)
        }
        batches: list[list[str]] = []
        for step in range(POLICY_STEPS):
            batch = list(panel) + ordinary_parts[0][step] + ordinary_parts[1][step]
            random.Random(POLICY_SEED + 1000 * (panel_index + 1) + step).shuffle(batch)
            if len(batch) != POLICY_BATCH or len(set(batch)) != POLICY_BATCH:
                raise AssertionError("invalid panel training batch")
            batches.append(batch)
        panel_schedules[str(panel_index)] = {"anchor_ids": list(panel), "batches": batches}
    return {"global": global_schedule, "panels": panel_schedules}


def summarize_evaluation_rows(rows: list[dict[str, object]]) -> dict[str, float]:
    """Reduce swap-paired A/B evaluations to semantic, mass, and position metrics."""
    if not rows or len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("evaluation rows must be nonempty with unique ids")
    probabilities: list[float] = []
    by_swap: dict[int, list[float]] = {0: [], 1: []}
    masses: list[float] = []
    for row in rows:
        swap = int(row["swap"])
        values = [float(value) for value in row["semantic_probabilities"]]  # type: ignore[arg-type]
        mass = float(row["ab_mass"])
        if (
            swap not in (0, 1) or len(values) != 2
            or any(not math.isfinite(value) or value < 0 or value > 1 for value in values)
            or abs(sum(values) - 1.) > 1e-5
            or not math.isfinite(mass) or mass < 0 or mass > 1 + 1e-5
        ):
            raise ValueError("invalid evaluation row")
        probabilities.append(values[1])
        by_swap[swap].append(values[1])
        masses.append(mass)
    if not by_swap[0] or not by_swap[1]:
        raise ValueError("both option orders are required")
    swap_means = {swap: mean(values) for swap, values in by_swap.items()}
    return {
        "semantic1_probability_mean": mean(probabilities),
        "swap0_semantic1_probability_mean": swap_means[0],
        "swap1_semantic1_probability_mean": swap_means[1],
        "semantic_position_gap": abs(swap_means[0] - swap_means[1]),
        "min_ab_mass": min(masses),
    }


def expected_policy_arm_names() -> list[str]:
    names = ["baseline", "raw_immediate", "oracle_delayed"]
    for panel_index in range(POLICY_PANEL_COUNT):
        names.extend([
            f"panel_{panel_index:02d}_anchor_sdpo",
            f"panel_{panel_index:02d}_anchor_sft",
            f"panel_{panel_index:02d}_augmented",
        ])
    return names


def summarize_policy_endpoints(metrics: dict[str, dict[str, float]]) -> dict[str, object]:
    """Apply the prospectively frozen G1 endpoint decision rule."""
    expected = expected_policy_arm_names()
    if set(metrics) != set(expected):
        raise ValueError("policy arm keys do not match the frozen design")
    required = {
        "semantic1_probability_mean", "swap0_semantic1_probability_mean",
        "swap1_semantic1_probability_mean", "semantic_position_gap", "min_ab_mass",
    }
    for name, values in metrics.items():
        if set(values) != required or any(not math.isfinite(float(value)) for value in values.values()):
            raise ValueError(f"invalid endpoint metrics: {name}")
        if not all(0 <= float(values[key]) <= 1 + 1e-5 for key in required):
            raise ValueError(f"endpoint metric outside probability range: {name}")

    probability = {name: float(values["semantic1_probability_mean"]) for name, values in metrics.items()}
    baseline = probability["baseline"]
    raw = probability["raw_immediate"]
    oracle = probability["oracle_delayed"]
    panel_rows: list[dict[str, object]] = []
    for panel_index in range(POLICY_PANEL_COUNT):
        prefix = f"panel_{panel_index:02d}"
        sdpo_name = f"{prefix}_anchor_sdpo"
        sft_name = f"{prefix}_anchor_sft"
        augmented_name = f"{prefix}_augmented"
        comparator_name = sdpo_name if probability[sdpo_name] >= probability[sft_name] else sft_name
        comparator = probability[comparator_name]
        augmented = probability[augmented_name]
        panel_rows.append({
            "panel": panel_index,
            "best_anchor_baseline": comparator_name.rsplit("_", 2)[-2] + "_" + comparator_name.rsplit("_", 1)[-1],
            "best_anchor_probability": comparator,
            "augmented_probability": augmented,
            "augmented_gain": augmented - comparator,
            "best_anchor_oracle_distance": abs(comparator - oracle),
            "augmented_oracle_distance": abs(augmented - oracle),
            "augmented_wins": augmented > comparator,
        })

    gains = [float(row["augmented_gain"]) for row in panel_rows]
    baseline_distances = [float(row["best_anchor_oracle_distance"]) for row in panel_rows]
    augmented_distances = [float(row["augmented_oracle_distance"]) for row in panel_rows]
    min_mass_floor = max(.10, .50 * float(metrics["baseline"]["min_ab_mass"]))
    position_gap_ceiling = max(.10, float(metrics["baseline"]["semantic_position_gap"]) + .02)
    aggregate = {
        "baseline_probability": baseline,
        "raw_probability": raw,
        "oracle_probability": oracle,
        "raw_shift": raw - baseline,
        "oracle_shift": oracle - baseline,
        "oracle_raw_gap": oracle - raw,
        "augmented_wins": sum(bool(row["augmented_wins"]) for row in panel_rows),
        "median_augmented_gain": median(gains),
        "mean_augmented_gain": mean(gains),
        "median_best_anchor_oracle_distance": median(baseline_distances),
        "median_augmented_oracle_distance": median(augmented_distances),
        "mean_best_anchor_oracle_distance": mean(baseline_distances),
        "mean_augmented_oracle_distance": mean(augmented_distances),
        "min_endpoint_ab_mass": min(float(values["min_ab_mass"]) for values in metrics.values()),
        "max_endpoint_position_gap": max(float(values["semantic_position_gap"]) for values in metrics.values()),
        "min_mass_floor": min_mass_floor,
        "position_gap_ceiling": position_gap_ceiling,
    }
    gates = {
        "oracle_increases_semantic1_by_point10": aggregate["oracle_shift"] >= .10,
        "raw_decreases_semantic1_by_point10": aggregate["raw_shift"] <= -.10,
        "oracle_raw_gap_at_least_point25": aggregate["oracle_raw_gap"] >= .25,
        "augmented_wins_at_least_six_panels": aggregate["augmented_wins"] >= 6,
        "median_augmented_gain_at_least_point05": aggregate["median_augmented_gain"] >= .05,
        "median_oracle_distance_reduced_twenty_percent": (
            aggregate["median_augmented_oracle_distance"]
            <= .80 * aggregate["median_best_anchor_oracle_distance"]
        ),
        "mean_oracle_distance_reduced_twenty_percent": (
            aggregate["mean_augmented_oracle_distance"]
            <= .80 * aggregate["mean_best_anchor_oracle_distance"]
        ),
        "answer_mass_preserved": aggregate["min_endpoint_ab_mass"] >= min_mass_floor,
        "position_control_preserved": aggregate["max_endpoint_position_gap"] <= position_gap_ceiling,
    }
    return {
        "decision": "NEURAL_POLICY_G1_QUALIFIED" if all(gates.values()) else "NEURAL_POLICY_G1_NOT_QUALIFIED",
        "gates": gates,
        "aggregate": aggregate,
        "panels": panel_rows,
        "scope": (
            "Synthetic first-token Qwen3.5-9B policy-learning gate after gradient G0. "
            "It tests whether paired sparse delayed anchors improve actual SDPO policy endpoints "
            "over equal-anchor SDPO and SFT, not human prevalence or paper viability."
        ),
    }
