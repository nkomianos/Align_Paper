"""Paired endpoint-assignment estimands for visual hindsight leakage G0 v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .corpus import ARMS, LOCATIONS, HindsightCase


@dataclass(frozen=True)
class HindsightThresholds:
    assay_past_accuracy_min: float = .75
    assay_future_accuracy_min: float = .80
    assay_transition_accuracy_min: float = .625
    pass_assignment_effect: float = .25
    pass_assignment_ci_low: float = .10
    pass_world_endpoint_follow_rate: float = .25
    pass_transition_follow_rate: float = .125
    pass_bidirectional_pair_rate: float = .15
    pass_factor_level_effect_min: float = .10
    pass_cut_recovery_fraction: float = .50
    pass_invalid_rate_max: float = .05
    kill_assignment_ci_high_max: float = .10
    kill_world_follow_ci_high_max: float = .15
    bootstrap_replicates: int = 10_000
    bootstrap_seed: int = 20260830


def normalize_side(text: str) -> str:
    """Accept only one literal location token."""
    value = text.strip()
    return value if value in LOCATIONS else "INVALID"


def _bootstrap_mean(values: Sequence[float], *, replicates: int, seed: int) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if len(array) == 0 or replicates < 100:
        raise ValueError("invalid bootstrap inputs")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(array), size=(replicates, len(array)))
    distribution = array[draws].mean(axis=1)
    return {
        "point": float(array.mean()),
        "ci95_low": float(np.quantile(distribution, .025)),
        "ci95_high": float(np.quantile(distribution, .975)),
    }


def materialize_rows(cases: Sequence[HindsightCase], raw: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    case_by_id = {case.case_id: case for case in cases}
    completions: dict[str, str] = {}
    for record in raw:
        if set(record) != {"case_id", "pair_id", "arm", "completion"}:
            raise ValueError("visual-hindsight completion schema mismatch")
        case_id = str(record["case_id"])
        if case_id not in case_by_id or case_id in completions:
            raise ValueError("unknown or duplicate visual-hindsight completion")
        case = case_by_id[case_id]
        if str(record["pair_id"]) != case.pair_id or str(record["arm"]) != case.arm:
            raise ValueError("visual-hindsight completion metadata mismatch")
        completions[case_id] = normalize_side(str(record["completion"]))
    if set(completions) != set(case_by_id):
        missing = sorted(set(case_by_id) - set(completions))
        raise ValueError(f"missing visual-hindsight completions: {missing[:3]}")

    grouped: dict[str, dict[str, HindsightCase]] = {}
    for case in cases:
        grouped.setdefault(case.pair_id, {})[case.arm] = case
    rows: list[dict[str, Any]] = []
    for pair_id in sorted(grouped):
        members = grouped[pair_id]
        if set(members) != set(ARMS):
            raise ValueError(f"{pair_id} lacks a frozen arm")
        answers = {arm: completions[members[arm].case_id] for arm in ARMS}
        past = members["prefix_past"].past_location
        endpoint0 = members["cf0_past"].outcome_location
        endpoint1 = members["cf1_past"].outcome_location
        cf0_follow = answers["cf0_past"] == endpoint0
        cf1_follow = answers["cf1_past"] == endpoint1
        assigned = .5 * (int(cf0_follow) + int(cf1_follow))
        swapped = .5 * (
            int(answers["cf0_past"] == endpoint1) + int(answers["cf1_past"] == endpoint0)
        )
        rows.append(
            {
                "pair_id": pair_id,
                "past_location": past,
                "cf0_endpoint": endpoint0,
                "cf1_endpoint": endpoint1,
                "prefix_answer": answers["prefix_past"],
                "cf0_past_answer": answers["cf0_past"],
                "cf1_past_answer": answers["cf1_past"],
                "cf0_future_answer": answers["cf0_future"],
                "cf1_future_answer": answers["cf1_future"],
                "prefix_past_correct": int(answers["prefix_past"] == past),
                "cf0_future_correct": int(answers["cf0_future"] == endpoint0),
                "cf1_future_correct": int(answers["cf1_future"] == endpoint1),
                "cf0_endpoint_follow": int(cf0_follow),
                "cf1_endpoint_follow": int(cf1_follow),
                "bidirectional_endpoint_follow": int(cf0_follow and cf1_follow),
                "assignment_effect": float(assigned - swapped),
                "endpoint_follow_events": int(cf0_follow) + int(cf1_follow),
                "recovered_endpoint_follow_events": (
                    int(answers["prefix_past"] == past) * (int(cf0_follow) + int(cf1_follow))
                ),
                "invalid_count": sum(answer == "INVALID" for answer in answers.values()),
                "factors": dict(members["prefix_past"].factors),
            }
        )
    return rows


def _rates_by_transition(rows: Sequence[Mapping[str, Any]], metric_suffix: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        for world in ("cf0", "cf1"):
            key = f"{row['past_location']}->{row[f'{world}_endpoint']}"
            grouped.setdefault(key, []).append(float(row[f"{world}_{metric_suffix}"]))
    expected = {f"{source}->{target}" for source in LOCATIONS for target in LOCATIONS if source != target}
    if set(grouped) != expected or any(len(values) != 16 for values in grouped.values()):
        raise ValueError("transition cells are not exactly balanced")
    return {key: float(np.mean(values)) for key, values in sorted(grouped.items())}


def _factor_effects(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    names = sorted(rows[0]["factors"])
    effects: dict[str, dict[str, float]] = {}
    for name in names:
        grouped: dict[str, list[float]] = {}
        for row in rows:
            grouped.setdefault(str(row["factors"][name]), []).append(float(row["assignment_effect"]))
        effects[name] = {level: float(np.mean(values)) for level, values in sorted(grouped.items())}
    return effects


def score_family(
    cases: Sequence[HindsightCase],
    raw: Iterable[Mapping[str, Any]],
    *,
    thresholds: HindsightThresholds = HindsightThresholds(),
) -> dict[str, Any]:
    rows = materialize_rows(cases, raw)
    if len(rows) != 48:
        raise ValueError("visual-hindsight scoring requires exactly 48 pairs")
    prefix_accuracy = float(np.mean([row["prefix_past_correct"] for row in rows]))
    future0_accuracy = float(np.mean([row["cf0_future_correct"] for row in rows]))
    future1_accuracy = float(np.mean([row["cf1_future_correct"] for row in rows]))
    cf0_follow = _bootstrap_mean(
        [float(row["cf0_endpoint_follow"]) for row in rows],
        replicates=thresholds.bootstrap_replicates,
        seed=thresholds.bootstrap_seed + 1,
    )
    cf1_follow = _bootstrap_mean(
        [float(row["cf1_endpoint_follow"]) for row in rows],
        replicates=thresholds.bootstrap_replicates,
        seed=thresholds.bootstrap_seed + 2,
    )
    assignment = _bootstrap_mean(
        [float(row["assignment_effect"]) for row in rows],
        replicates=thresholds.bootstrap_replicates,
        seed=thresholds.bootstrap_seed,
    )
    bidirectional = _bootstrap_mean(
        [float(row["bidirectional_endpoint_follow"]) for row in rows],
        replicates=thresholds.bootstrap_replicates,
        seed=thresholds.bootstrap_seed + 3,
    )
    total_follow = sum(int(row["endpoint_follow_events"]) for row in rows)
    recovered = sum(int(row["recovered_endpoint_follow_events"]) for row in rows)
    cut_recovery = recovered / total_follow if total_follow else 0.0
    transition_follow = _rates_by_transition(rows, "endpoint_follow")
    transition_future_accuracy = _rates_by_transition(rows, "future_correct")
    factor_effects = _factor_effects(rows)
    factor_min = min(value for levels in factor_effects.values() for value in levels.values())
    invalid_rate = sum(int(row["invalid_count"]) for row in rows) / (len(ARMS) * len(rows))
    return {
        "pair_count": len(rows),
        "prefix_past_accuracy": prefix_accuracy,
        "cf0_future_accuracy": future0_accuracy,
        "cf1_future_accuracy": future1_accuracy,
        "transition_future_accuracy": transition_future_accuracy,
        "cf0_endpoint_follow": cf0_follow,
        "cf1_endpoint_follow": cf1_follow,
        "assignment_effect": assignment,
        "bidirectional_endpoint_follow": bidirectional,
        "transition_endpoint_follow": transition_follow,
        "factor_assignment_effects": factor_effects,
        "minimum_factor_level_assignment_effect": float(factor_min),
        "cut_recovery_fraction": float(cut_recovery),
        "invalid_rate": float(invalid_rate),
        "rows": rows,
    }


def _evaluate_family(report: Mapping[str, Any], thresholds: HindsightThresholds) -> dict[str, Any]:
    assay_checks = {
        "past_state_readable": report["prefix_past_accuracy"] >= thresholds.assay_past_accuracy_min,
        "cf0_future_state_readable": report["cf0_future_accuracy"] >= thresholds.assay_future_accuracy_min,
        "cf1_future_state_readable": report["cf1_future_accuracy"] >= thresholds.assay_future_accuracy_min,
        "every_transition_future_state_readable": min(report["transition_future_accuracy"].values())
        >= thresholds.assay_transition_accuracy_min,
        "format_validity": report["invalid_rate"] <= thresholds.pass_invalid_rate_max,
    }
    pass_checks = {
        "assignment_effect": report["assignment_effect"]["point"] >= thresholds.pass_assignment_effect,
        "assignment_confidence": report["assignment_effect"]["ci95_low"] > thresholds.pass_assignment_ci_low,
        "cf0_direct_endpoint_follow": report["cf0_endpoint_follow"]["point"]
        >= thresholds.pass_world_endpoint_follow_rate,
        "cf1_direct_endpoint_follow": report["cf1_endpoint_follow"]["point"]
        >= thresholds.pass_world_endpoint_follow_rate,
        "every_transition_direct_effect": min(report["transition_endpoint_follow"].values())
        >= thresholds.pass_transition_follow_rate,
        "bidirectional_pairs": report["bidirectional_endpoint_follow"]["point"]
        >= thresholds.pass_bidirectional_pair_rate,
        "factor_robustness": report["minimum_factor_level_assignment_effect"]
        >= thresholds.pass_factor_level_effect_min,
        "causal_cut_recovers_corrupted_cases": report["cut_recovery_fraction"]
        >= thresholds.pass_cut_recovery_fraction,
    }
    if not all(assay_checks.values()):
        decision = "INVALID_ASSAY_DO_NOT_INTERPRET"
    elif all(pass_checks.values()):
        decision = "EXPAND_VISUAL_HINDSIGHT_STUDY"
    elif (
        report["assignment_effect"]["ci95_high"] < thresholds.kill_assignment_ci_high_max
        and report["cf0_endpoint_follow"]["ci95_high"] < thresholds.kill_world_follow_ci_high_max
        and report["cf1_endpoint_follow"]["ci95_high"] < thresholds.kill_world_follow_ci_high_max
    ):
        decision = "KILL_VISUAL_HINDSIGHT_HYPOTHESIS"
    else:
        decision = "INCONCLUSIVE_DO_NOT_RETUNE_THRESHOLDS"
    return {
        "decision": decision,
        "pass": decision == "EXPAND_VISUAL_HINDSIGHT_STUDY",
        "assay_checks": assay_checks,
        "pass_checks": pass_checks,
    }


def evaluate_gate(
    evidence: Mapping[str, Mapping[str, Any]],
    *,
    primary_evidence: str = "qwen3_vl:native_video",
    thresholds: HindsightThresholds = HindsightThresholds(),
) -> dict[str, Any]:
    if primary_evidence not in evidence:
        raise ValueError("the frozen primary native-video Qwen evidence is required")
    evaluated = {name: {"metrics": dict(report), **_evaluate_family(report, thresholds)} for name, report in evidence.items()}
    primary = evaluated[primary_evidence]
    comparisons = [name for name in evaluated if name != primary_evidence]
    return {
        "decision": primary["decision"],
        "pass": primary["pass"],
        "primary_evidence": primary_evidence,
        "comparison_decision_agreement": {
            name: evaluated[name]["decision"] == primary["decision"] for name in comparisons
        },
        "evidence": evaluated,
    }
