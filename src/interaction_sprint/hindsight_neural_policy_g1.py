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
    """Build the fixed population schedule and paired anchor sets.

    Augmented training must not insert the repeatedly measured anchors into the
    population batch.  It uses ``global`` for the population expectation and a
    panel's separate ``anchor_ids`` for the paired correction.
    """
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
        if any(
            sum(int(by_id[row_id]["logged_action"]) == action for row_id in panel)
            != POLICY_ANCHORS_PER_ACTION
            for action in (0, 1)
        ):
            raise ValueError("panels must be balanced on logged action")
        panel_schedules[str(panel_index)] = {"anchor_ids": list(panel)}
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


def _validate_endpoint_metrics(metrics: dict[str, dict[str, float]], expected: set[str]) -> None:
    if set(metrics) != expected:
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


def summarize_policy_controls(metrics: dict[str, dict[str, float]]) -> dict[str, object]:
    """Qualify raw/oracle policy acquisition before sparse-panel training."""
    expected = {"baseline", "raw_immediate", "oracle_delayed"}
    _validate_endpoint_metrics(metrics, expected)
    probability = {name: float(values["semantic1_probability_mean"]) for name, values in metrics.items()}
    baseline = probability["baseline"]
    raw = probability["raw_immediate"]
    oracle = probability["oracle_delayed"]
    min_mass_floor = max(.10, .50 * float(metrics["baseline"]["min_ab_mass"]))
    position_gap_ceiling = max(.10, float(metrics["baseline"]["semantic_position_gap"]) + .02)
    aggregate = {
        "baseline_probability": baseline,
        "raw_probability": raw,
        "oracle_probability": oracle,
        "raw_shift": raw - baseline,
        "oracle_shift": oracle - baseline,
        "oracle_raw_gap": oracle - raw,
        "min_control_ab_mass": min(float(values["min_ab_mass"]) for values in metrics.values()),
        "max_control_position_gap": max(float(values["semantic_position_gap"]) for values in metrics.values()),
        "min_mass_floor": min_mass_floor,
        "position_gap_ceiling": position_gap_ceiling,
    }
    gates = {
        "oracle_increases_semantic1_by_point10": aggregate["oracle_shift"] >= .10,
        "raw_decreases_semantic1_by_point10": aggregate["raw_shift"] <= -.10,
        "oracle_raw_gap_at_least_point25": aggregate["oracle_raw_gap"] >= .25,
        "control_answer_mass_preserved": aggregate["min_control_ab_mass"] >= min_mass_floor,
        "control_position_preserved": aggregate["max_control_position_gap"] <= position_gap_ceiling,
    }
    return {
        "decision": "POLICY_ACQUISITION_QUALIFIED" if all(gates.values()) else "STOP_POLICY_ACQUISITION_UNQUALIFIED",
        "gates": gates,
        "aggregate": aggregate,
    }


def summarize_policy_endpoints(metrics: dict[str, dict[str, float]]) -> dict[str, object]:
    """Apply the prospectively frozen G1 endpoint decision rule."""
    expected = expected_policy_arm_names()
    _validate_endpoint_metrics(metrics, set(expected))

    probability = {name: float(values["semantic1_probability_mean"]) for name, values in metrics.items()}
    oracle = probability["oracle_delayed"]
    controls = summarize_policy_controls({name: metrics[name] for name in expected[:3]})
    panel_rows: list[dict[str, object]] = []
    for panel_index in range(POLICY_PANEL_COUNT):
        prefix = f"panel_{panel_index:02d}"
        sdpo_name = f"{prefix}_anchor_sdpo"
        sft_name = f"{prefix}_anchor_sft"
        augmented_name = f"{prefix}_augmented"
        sdpo_distance = abs(probability[sdpo_name] - oracle)
        sft_distance = abs(probability[sft_name] - oracle)
        augmented = probability[augmented_name]
        augmented_distance = abs(augmented - oracle)
        panel_rows.append({
            "panel": panel_index,
            "anchor_sdpo_probability": probability[sdpo_name],
            "anchor_sft_probability": probability[sft_name],
            "augmented_probability": augmented,
            "anchor_sdpo_oracle_distance": sdpo_distance,
            "anchor_sft_oracle_distance": sft_distance,
            "augmented_oracle_distance": augmented_distance,
            "sdpo_oracle_distance_gain": sdpo_distance - augmented_distance,
            "sft_oracle_distance_gain": sft_distance - augmented_distance,
            "augmented_strictly_closer_than_sdpo": sdpo_distance - augmented_distance > .01,
            "augmented_noninferior_to_sdpo": augmented_distance <= sdpo_distance + .01,
        })

    sdpo_gains = [float(row["sdpo_oracle_distance_gain"]) for row in panel_rows]
    sft_gains = [float(row["sft_oracle_distance_gain"]) for row in panel_rows]
    sdpo_distances = [float(row["anchor_sdpo_oracle_distance"]) for row in panel_rows]
    sft_distances = [float(row["anchor_sft_oracle_distance"]) for row in panel_rows]
    augmented_distances = [float(row["augmented_oracle_distance"]) for row in panel_rows]
    min_mass_floor = max(.10, .50 * float(metrics["baseline"]["min_ab_mass"]))
    position_gap_ceiling = max(.10, float(metrics["baseline"]["semantic_position_gap"]) + .02)
    aggregate = {
        **controls["aggregate"],
        "augmented_strictly_closer_than_sdpo_panels": sum(
            bool(row["augmented_strictly_closer_than_sdpo"]) for row in panel_rows
        ),
        "augmented_noninferior_to_sdpo_panels": sum(
            bool(row["augmented_noninferior_to_sdpo"]) for row in panel_rows
        ),
        "mean_sdpo_oracle_distance_gain": mean(sdpo_gains),
        "mean_sft_oracle_distance_gain": mean(sft_gains),
        "median_anchor_sdpo_oracle_distance": median(sdpo_distances),
        "median_anchor_sft_oracle_distance": median(sft_distances),
        "median_augmented_oracle_distance": median(augmented_distances),
        "mean_anchor_sdpo_oracle_distance": mean(sdpo_distances),
        "mean_anchor_sft_oracle_distance": mean(sft_distances),
        "mean_augmented_oracle_distance": mean(augmented_distances),
        "rms_anchor_sdpo_oracle_distance": math.sqrt(mean([value * value for value in sdpo_distances])),
        "rms_anchor_sft_oracle_distance": math.sqrt(mean([value * value for value in sft_distances])),
        "rms_augmented_oracle_distance": math.sqrt(mean([value * value for value in augmented_distances])),
        "max_anchor_sdpo_oracle_distance": max(sdpo_distances),
        "max_anchor_sft_oracle_distance": max(sft_distances),
        "max_augmented_oracle_distance": max(augmented_distances),
        "min_endpoint_ab_mass": min(float(values["min_ab_mass"]) for values in metrics.values()),
        "max_endpoint_position_gap": max(float(values["semantic_position_gap"]) for values in metrics.values()),
        "min_mass_floor": min_mass_floor,
        "position_gap_ceiling": position_gap_ceiling,
    }
    gates = {
        **controls["gates"],
        "mean_oracle_distance_reduced_twenty_percent_vs_sdpo": (
            aggregate["mean_augmented_oracle_distance"]
            <= .80 * aggregate["mean_anchor_sdpo_oracle_distance"]
        ),
        "rms_oracle_distance_reduced_twenty_percent_vs_sdpo": (
            aggregate["rms_augmented_oracle_distance"]
            <= .80 * aggregate["rms_anchor_sdpo_oracle_distance"]
        ),
        "mean_oracle_distance_reduced_twenty_percent_vs_sft": (
            aggregate["mean_augmented_oracle_distance"]
            <= .80 * aggregate["mean_anchor_sft_oracle_distance"]
        ),
        "rms_oracle_distance_reduced_twenty_percent_vs_sft": (
            aggregate["rms_augmented_oracle_distance"]
            <= .80 * aggregate["rms_anchor_sft_oracle_distance"]
        ),
        "mean_sdpo_oracle_distance_gain_at_least_point015": (
            aggregate["mean_sdpo_oracle_distance_gain"] >= .015
        ),
        "mean_sft_oracle_distance_gain_at_least_point015": (
            aggregate["mean_sft_oracle_distance_gain"] >= .015
        ),
        "augmented_strictly_closer_than_sdpo_at_least_three_panels": (
            aggregate["augmented_strictly_closer_than_sdpo_panels"] >= 3
        ),
        "augmented_noninferior_to_sdpo_at_least_six_panels": (
            aggregate["augmented_noninferior_to_sdpo_panels"] >= 6
        ),
        "median_oracle_distance_not_worse_than_sdpo": (
            aggregate["median_augmented_oracle_distance"]
            <= aggregate["median_anchor_sdpo_oracle_distance"] + 1e-12
        ),
        "maximum_oracle_distance_not_worse_than_sdpo": (
            aggregate["max_augmented_oracle_distance"]
            <= aggregate["max_anchor_sdpo_oracle_distance"] + 1e-12
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
            "It tests whether paired sparse delayed anchors move actual SDPO policy endpoints "
            "closer to a full delayed-feedback oracle than equal-anchor SDPO and SFT, not human "
            "prevalence or paper viability."
        ),
    }
