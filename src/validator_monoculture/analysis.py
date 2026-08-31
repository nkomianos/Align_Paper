"""Locked, CWE-clustered analysis for validator-monoculture G0.

The sole primary endpoint uses the specification-only verifier arm and a fixed
number of proposed test slots. Invalid, duplicate, and malformed tests consume
slots and count as failures. This intention-to-test estimand avoids conditioning
on verifier validity. A capped valid-test analysis, patch-aware arm, and
DEV-selected mixed portfolio are prespecified secondary controls.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


SPEC_ONLY = "spec_only"
PATCH_AWARE = "patch_aware"
PROMPT_MODES = (SPEC_ONLY, PATCH_AWARE)


@dataclass(frozen=True)
class GateThresholds:
    proposal_test_budget: int = 12
    valid_test_budget: int = 4
    minimum_test_patches: int = 30
    minimum_patches_per_generator: int = 10
    minimum_test_cwes: int = 4
    minimum_generator_direction: float = 0.05
    effect_to_expand: float = 0.10
    effect_to_kill: float = 0.05
    minimum_valid_budget_reach: float = 0.80
    maximum_valid_budget_reach_gap: float = 0.10
    minimum_planted_control_detection_rate: float = 0.20
    minimum_planted_control_cell_detection_rate: float = 0.20
    minimum_planted_control_cwe_coverage: float = 1.0
    bootstrap_replicates: int = 10_000
    bootstrap_seed: int = 20260830
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        if self.proposal_test_budget < 2:
            raise ValueError("proposal_test_budget must be at least two")
        if not 1 <= self.valid_test_budget <= self.proposal_test_budget:
            raise ValueError("valid_test_budget must be within the proposal budget")
        if min(self.minimum_test_patches, self.minimum_patches_per_generator, self.minimum_test_cwes) < 1:
            raise ValueError("minimum sample counts must be positive")
        if not 0 <= self.effect_to_kill <= self.effect_to_expand <= 1:
            raise ValueError("effect thresholds must satisfy 0 <= kill <= expand <= 1")
        if not 0 <= self.minimum_generator_direction <= self.effect_to_expand:
            raise ValueError("minimum_generator_direction is invalid")
        if not 0 <= self.minimum_valid_budget_reach <= 1:
            raise ValueError("minimum_valid_budget_reach is invalid")
        if not 0 <= self.maximum_valid_budget_reach_gap <= 1:
            raise ValueError("maximum_valid_budget_reach_gap is invalid")
        if not 0 <= self.minimum_planted_control_detection_rate <= 1:
            raise ValueError("minimum_planted_control_detection_rate is invalid")
        if not 0 <= self.minimum_planted_control_cell_detection_rate <= 1:
            raise ValueError("minimum_planted_control_cell_detection_rate is invalid")
        if not 0 <= self.minimum_planted_control_cwe_coverage <= 1:
            raise ValueError("minimum_planted_control_cwe_coverage is invalid")
        if self.bootstrap_replicates < 100:
            raise ValueError("bootstrap_replicates must be at least 100")
        if not 0.5 < self.confidence_level < 1:
            raise ValueError("confidence_level must lie in (0.5, 1)")


def _stable_order(ids: Iterable[str]) -> list[str]:
    return sorted(
        {str(item) for item in ids},
        key=lambda item: (hashlib.sha256(item.encode("utf-8")).digest(), item),
    )


def _normalize_rows(rows: Sequence[Mapping[str, Any]], families: Sequence[str], proposal_budget: int) -> list[dict[str, Any]]:
    if len(set(families)) != 2:
        raise ValueError("the frozen G0 requires exactly two distinct model families")
    family_set = set(families)
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    task_meta: dict[str, tuple[str, str]] = {}
    cwe_split: dict[str, str] = {}
    for raw in rows:
        required = {
            "task_id", "split", "cwe", "patch_id", "patch_family",
            "verifier_family", "prompt_mode", "proposal_test_ids",
            "valid_test_ids", "kill_test_ids", "indeterminate_execution_count",
        }
        missing = required.difference(raw)
        if missing:
            raise ValueError(f"evaluation row is missing {sorted(missing)}")
        patch_family, verifier = str(raw["patch_family"]), str(raw["verifier_family"])
        mode = str(raw["prompt_mode"])
        if patch_family not in family_set or verifier not in family_set:
            raise ValueError("evaluation row contains an unfrozen model family")
        if mode not in PROMPT_MODES:
            raise ValueError("prompt_mode must be spec_only or patch_aware")
        split_raw = str(raw["split"])
        split = {"development": "DEV", "locked_test": "TEST"}.get(split_raw, split_raw)
        if split not in {"DEV", "TEST"}:
            raise ValueError("split must be DEV or TEST")
        task_id, cwe, patch_id = str(raw["task_id"]), str(raw["cwe"]), str(raw["patch_id"])
        old_task = task_meta.setdefault(task_id, (cwe, split))
        if old_task != (cwe, split):
            raise ValueError(f"task metadata changes across rows: {task_id}")
        old_split = cwe_split.setdefault(cwe, split)
        if old_split != split:
            raise ValueError(f"CWE family appears in both DEV and TEST: {cwe}")
        key = (patch_id, verifier, mode)
        if key in seen:
            raise ValueError(f"duplicate patch/verifier/mode row: {key}")
        seen.add(key)
        proposals = [str(item) for item in raw["proposal_test_ids"]]
        if len(proposals) != proposal_budget or len(set(proposals)) != proposal_budget:
            raise ValueError(f"row must contain exactly {proposal_budget} unique proposal slots")
        valid = _stable_order(str(item) for item in raw["valid_test_ids"])
        killed = {str(item) for item in raw["kill_test_ids"]}
        if not set(valid).issubset(proposals) or not killed.issubset(valid):
            raise ValueError(f"valid/kill IDs violate proposal containment for {key}")
        indeterminate = raw["indeterminate_execution_count"]
        if (
            isinstance(indeterminate, bool)
            or not isinstance(indeterminate, int)
            or not 0 <= indeterminate <= proposal_budget
        ):
            raise ValueError(f"indeterminate execution count is invalid for {key}")
        normalized.append({
            "task_id": task_id, "split": split, "cwe": cwe, "patch_id": patch_id,
            "patch_family": patch_family, "verifier_family": verifier,
            "prompt_mode": mode, "proposal_test_ids": proposals,
            "valid_test_ids": valid, "kill_test_ids": killed,
            "indeterminate_execution_count": indeterminate,
        })
    return normalized


def _paired(rows: Sequence[Mapping[str, Any]], families: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
    patch_modes: dict[str, set[str]] = {}
    for row in rows:
        grouped.setdefault((row["patch_id"], row["prompt_mode"]), {})[row["verifier_family"]] = row
        patch_modes.setdefault(row["patch_id"], set()).add(row["prompt_mode"])
    if any(modes != set(PROMPT_MODES) for modes in patch_modes.values()):
        raise ValueError("every patch must have both frozen verifier prompt modes")
    paired: list[dict[str, Any]] = []
    for (patch_id, mode), cells in grouped.items():
        if set(cells) != set(families):
            raise ValueError(f"patch/mode lacks an exact crossed verifier pair: {(patch_id, mode)}")
        first = cells[families[0]]
        for family in families[1:]:
            if any(cells[family][field] != first[field] for field in ("task_id", "split", "cwe", "patch_family", "prompt_mode")):
                raise ValueError(f"crossed rows disagree on patch metadata: {patch_id}")
        paired.append({
            "patch_id": patch_id, "task_id": first["task_id"], "split": first["split"],
            "cwe": first["cwe"], "patch_family": first["patch_family"],
            "prompt_mode": mode, "cells": cells,
        })
    return paired


def _proposal_detected(cell: Mapping[str, Any], count: int) -> float:
    return float(bool(set(cell["proposal_test_ids"][:count]).intersection(cell["kill_test_ids"])))


def _valid_detected(cell: Mapping[str, Any], count: int) -> float:
    if len(cell["valid_test_ids"]) < count:
        return 0.0
    return float(bool(set(cell["valid_test_ids"][:count]).intersection(cell["kill_test_ids"])))


Detector = Callable[[Mapping[str, Any], int], float]


def _macro_rate(patches: Sequence[Mapping[str, Any]], *, patch_family: str, verifier_family: str, budget: int, detector: Detector) -> float:
    """Patch -> task -> CWE macro-average, preventing prolific tasks dominating."""
    task_values: dict[tuple[str, str], list[float]] = {}
    for patch in patches:
        if patch["patch_family"] == patch_family:
            task_values.setdefault((patch["cwe"], patch["task_id"]), []).append(detector(patch["cells"][verifier_family], budget))
    cwe_values: dict[str, list[float]] = {}
    for (cwe, _task), values in task_values.items():
        cwe_values.setdefault(cwe, []).append(float(np.mean(values)))
    if not cwe_values:
        return float("nan")
    return float(np.mean([np.mean(values) for values in cwe_values.values()]))


def _cell_rates(patches: Sequence[Mapping[str, Any]], families: Sequence[str], budget: int, detector: Detector) -> dict[str, dict[str, float]]:
    return {
        patch_family: {
            verifier: _macro_rate(patches, patch_family=patch_family, verifier_family=verifier, budget=budget, detector=detector)
            for verifier in families
        }
        for patch_family in families
    }


def _crossed_effect(rates: Mapping[str, Mapping[str, float]], families: Sequence[str]) -> tuple[float, dict[str, float]]:
    a, b = families
    directions = {a: float(rates[a][b] - rates[a][a]), b: float(rates[b][a] - rates[b][b])}
    return float(0.5 * (directions[a] + directions[b])), directions


def _subset_cwes(patches: Sequence[Mapping[str, Any]], cwes: Sequence[str]) -> list[Mapping[str, Any]]:
    wanted = set(cwes)
    return [patch for patch in patches if patch["cwe"] in wanted]


def _cwe_bootstrap(test: Sequence[Mapping[str, Any]], families: Sequence[str], budget: int, detector: Detector, thresholds: GateThresholds) -> dict[str, float]:
    cwes = sorted({str(row["cwe"]) for row in test})
    if len(cwes) < 2:
        return {"lower": float("nan"), "upper": float("nan")}
    per_cwe: list[float] = []
    for cwe in cwes:
        rates = _cell_rates(_subset_cwes(test, [cwe]), families, budget, detector)
        if any(np.isnan(value) for cell in rates.values() for value in cell.values()):
            return {"lower": float("nan"), "upper": float("nan")}
        per_cwe.append(_crossed_effect(rates, families)[0])
    rng = np.random.default_rng(thresholds.bootstrap_seed)
    samples = rng.choice(np.asarray(per_cwe, dtype=float), size=(thresholds.bootstrap_replicates, len(per_cwe)), replace=True).mean(axis=1)
    alpha = (1 - thresholds.confidence_level) / 2
    return {"lower": float(np.quantile(samples, alpha)), "upper": float(np.quantile(samples, 1 - alpha))}


def _leave_one_cwe_out(test: Sequence[Mapping[str, Any]], families: Sequence[str], budget: int, detector: Detector) -> dict[str, float]:
    cwes = sorted({str(row["cwe"]) for row in test})
    result: dict[str, float] = {}
    for held_out in cwes:
        rates = _cell_rates([row for row in test if row["cwe"] != held_out], families, budget, detector)
        result[held_out] = _crossed_effect(rates, families)[0]
    return result


def _valid_budget_reach(patches: Sequence[Mapping[str, Any]], families: Sequence[str], count: int) -> dict[str, float]:
    return {
        verifier: float(np.mean([len(patch["cells"][verifier]["valid_test_ids"]) >= count for patch in patches])) if patches else float("nan")
        for verifier in families
    }


def _valid_budget_reach_by_cwe(
    patches: Sequence[Mapping[str, Any]], families: Sequence[str], count: int
) -> dict[str, dict[str, float]]:
    """Task-macro reach within each CWE, avoiding patch-volume weighting."""

    result: dict[str, dict[str, float]] = {}
    for cwe in sorted({str(patch["cwe"]) for patch in patches}):
        result[cwe] = {}
        for family in families:
            task_values: dict[str, list[float]] = {}
            for patch in patches:
                if patch["cwe"] != cwe:
                    continue
                task_values.setdefault(str(patch["task_id"]), []).append(
                    float(len(patch["cells"][family]["valid_test_ids"]) >= count)
                )
            result[cwe][family] = (
                float(np.mean([np.mean(values) for values in task_values.values()]))
                if task_values
                else float("nan")
            )
    return result


def _common_task_support(
    patches: Sequence[Mapping[str, Any]], families: Sequence[str]
) -> tuple[list[Mapping[str, Any]], set[str]]:
    """Keep tasks that produced at least one eligible patch from every family.

    Without this restriction the two generator directions can be averaged over
    different task mixtures, which can manufacture a crossed interaction even
    when verification errors are not lineage-specific.
    """

    presence: dict[str, set[str]] = {}
    for patch in patches:
        presence.setdefault(str(patch["task_id"]), set()).add(str(patch["patch_family"]))
    common = {task_id for task_id, observed in presence.items() if observed == set(families)}
    return [patch for patch in patches if str(patch["task_id"]) in common], common


def _mixed_detected(patch: Mapping[str, Any], families: Sequence[str], allocation_first: int, budget: int) -> float:
    counts = (allocation_first, budget - allocation_first)
    return float(any(
        _proposal_detected(patch["cells"][family], count)
        for family, count in zip(families, counts, strict=True)
    ))


def _mixed_macro_rate(patches: Sequence[Mapping[str, Any]], families: Sequence[str], patch_family: str, allocation_first: int, budget: int) -> float:
    task_values: dict[tuple[str, str], list[float]] = {}
    for patch in patches:
        if patch["patch_family"] == patch_family:
            task_values.setdefault((patch["cwe"], patch["task_id"]), []).append(_mixed_detected(patch, families, allocation_first, budget))
    cwe_values: dict[str, list[float]] = {}
    for (cwe, _task), values in task_values.items():
        cwe_values.setdefault(cwe, []).append(float(np.mean(values)))
    return float(np.mean([np.mean(values) for values in cwe_values.values()])) if cwe_values else float("nan")


def _portfolio(dev: Sequence[Mapping[str, Any]], test: Sequence[Mapping[str, Any]], families: Sequence[str], budget: int) -> dict[str, Any]:
    allocations: dict[str, int] = {}
    for patch_family in families:
        scores = {count: _mixed_macro_rate(dev, families, patch_family, count, budget) for count in range(1, budget)}
        best = max(scores.values())
        candidates = [count for count, score in scores.items() if np.isclose(score, best)]
        allocations[patch_family] = min(candidates, key=lambda n: (abs(n - budget / 2), n))
    mixed_rates: dict[str, float] = {}
    gains: dict[str, float] = {}
    homogeneous: dict[str, dict[str, float]] = {}
    for patch_family in families:
        homogeneous[patch_family] = {
            verifier: _macro_rate(test, patch_family=patch_family, verifier_family=verifier, budget=budget, detector=_proposal_detected)
            for verifier in families
        }
        mixed_rates[patch_family] = _mixed_macro_rate(test, families, patch_family, allocations[patch_family], budget)
        gains[patch_family] = mixed_rates[patch_family] - max(homogeneous[patch_family].values())
    return {
        "family0_allocations": allocations,
        "test_mixed_detection_rates": mixed_rates,
        "test_homogeneous_detection_rates": homogeneous,
        "gain_over_best_homogeneous": gains,
        "mean_gain_over_best_homogeneous": float(np.mean(list(gains.values()))),
    }


def evaluate_gate(rows: Sequence[Mapping[str, Any]], *, families: Sequence[str] = ("qwen3_5", "gemma4"), thresholds: GateThresholds | None = None) -> dict[str, Any]:
    thresholds = thresholds or GateThresholds()
    families = tuple(families)
    normalized = _normalize_rows(rows, families, thresholds.proposal_test_budget)
    paired = _paired(normalized, families)
    primary = [row for row in paired if row["prompt_mode"] == SPEC_ONLY]
    patch_aware = [row for row in paired if row["prompt_mode"] == PATCH_AWARE]
    dev_raw = [row for row in primary if row["split"] == "DEV"]
    test_raw = [row for row in primary if row["split"] == "TEST"]
    dev, dev_common_tasks = _common_task_support(dev_raw, families)
    test, test_common_tasks = _common_task_support(test_raw, families)
    test_aware = [
        row for row in patch_aware
        if row["split"] == "TEST" and row["task_id"] in test_common_tasks
    ]
    counts = {
        "DEV_raw_patches": len(dev_raw), "TEST_raw_patches": len(test_raw),
        "DEV_patches": len(dev), "TEST_patches": len(test),
        "TEST_by_patch_family": {family: sum(row["patch_family"] == family for row in test) for family in families},
        "TEST_tasks": len({row["task_id"] for row in test}),
        "DEV_common_tasks": sorted(dev_common_tasks),
        "TEST_common_tasks": sorted(test_common_tasks),
        "TEST_cwes": sorted({row["cwe"] for row in test}),
    }
    enough = (
        counts["TEST_patches"] >= thresholds.minimum_test_patches
        and len(counts["TEST_cwes"]) >= thresholds.minimum_test_cwes
        and all(counts["TEST_by_patch_family"][family] >= thresholds.minimum_patches_per_generator for family in families)
    )
    proposal_rates = _cell_rates(test, families, thresholds.proposal_test_budget, _proposal_detected)
    proposal_effect, proposal_directions = _crossed_effect(proposal_rates, families)
    proposal_ci = _cwe_bootstrap(test, families, thresholds.proposal_test_budget, _proposal_detected, thresholds)
    loo = _leave_one_cwe_out(test, families, thresholds.proposal_test_budget, _proposal_detected) if len(counts["TEST_cwes"]) > 1 else {}
    valid_rates = _cell_rates(test, families, thresholds.valid_test_budget, _valid_detected)
    valid_effect, valid_directions = _crossed_effect(valid_rates, families)
    reach = _valid_budget_reach(test, families, thresholds.valid_test_budget)
    reach_gap = abs(reach[families[0]] - reach[families[1]])
    reach_by_cwe = _valid_budget_reach_by_cwe(
        test, families, thresholds.valid_test_budget
    )
    per_cwe_reach_ok = bool(reach_by_cwe) and all(
        all(value >= thresholds.minimum_valid_budget_reach for value in cells.values())
        and abs(cells[families[0]] - cells[families[1]])
        <= thresholds.maximum_valid_budget_reach_gap
        for cells in reach_by_cwe.values()
    )
    indeterminate_execution_count = sum(
        int(patch["cells"][family]["indeterminate_execution_count"])
        for patch in test
        for family in families
    )
    aware_rates = _cell_rates(test_aware, families, thresholds.proposal_test_budget, _proposal_detected)
    aware_effect, aware_directions = _crossed_effect(aware_rates, families)
    portfolio = _portfolio(dev, test, families, thresholds.proposal_test_budget) if dev and test else {}
    checks = {
        "enough_plausible_security_patches": enough,
        "primary_effect_at_least_10pp": proposal_effect >= thresholds.effect_to_expand,
        "cwe_cluster_lcb_above_zero": proposal_ci["lower"] > 0,
        "both_patch_generators_at_least_5pp": all(proposal_directions[family] >= thresholds.minimum_generator_direction for family in families),
        "all_leave_one_cwe_out_positive": bool(loo) and all(value > 0 for value in loo.values()),
        "valid_matched_direction_positive": valid_effect > 0 and all(valid_directions[family] > 0 for family in families),
        "valid_budget_reach_adequate_and_balanced": all(value >= thresholds.minimum_valid_budget_reach for value in reach.values()) and reach_gap <= thresholds.maximum_valid_budget_reach_gap and per_cwe_reach_ok,
        "no_indeterminate_executions": indeterminate_execution_count == 0,
    }
    kill_checks = {
        "enough_plausible_security_patches": enough,
        "primary_effect_below_5pp": proposal_effect < thresholds.effect_to_kill,
        "cwe_heuristic_upper_below_expand_threshold": proposal_ci["upper"] < thresholds.effect_to_expand,
        "valid_budget_reach_adequate_and_balanced": checks["valid_budget_reach_adequate_and_balanced"],
        "no_indeterminate_executions": checks["no_indeterminate_executions"],
    }
    if all(checks.values()):
        decision = "EXPAND_VALIDATOR_MONOCULTURE"
    elif indeterminate_execution_count:
        decision = "INCONCLUSIVE_EXECUTION_ANOMALIES"
    elif not enough:
        decision = "INCONCLUSIVE_INSUFFICIENT_SECURITY_PATCHES"
    elif all(kill_checks.values()):
        decision = "KILL_VALIDATOR_MONOCULTURE"
    else:
        decision = "INCONCLUSIVE_BORDERLINE_OR_CONFOUNDED"
    return {
        "kind": "validator_monoculture_g0_report", "decision": decision,
        "pass": decision == "EXPAND_VALIDATOR_MONOCULTURE",
        "families": list(families),
        "primary_endpoint": "specification_only_fixed_proposal_budget_crossed_effect",
        "thresholds": asdict(thresholds), "eligible_counts": counts,
        "primary_fixed_proposal": {
            "cell_detection_rates": proposal_rates, "crossed_effect": proposal_effect,
            "directions": proposal_directions, "cwe_cluster_interval": proposal_ci,
            "leave_one_cwe_out": loo,
        },
        "secondary_capped_valid_budget": {
            "cell_detection_rates": valid_rates, "crossed_effect": valid_effect,
            "directions": valid_directions, "budget_reach_rates": reach,
            "reach_rate_gap": reach_gap, "budget_reach_by_cwe": reach_by_cwe,
        },
        "execution_anomalies": {
            "test_indeterminate_execution_count": indeterminate_execution_count,
        },
        "secondary_patch_aware_canonicalized": {
            "cell_detection_rates": aware_rates, "crossed_effect": aware_effect,
            "directions": aware_directions,
        },
        "secondary_dev_selected_mixed_portfolio": portfolio,
        "checks": checks,
        "kill_checks": kill_checks,
        "reasons": [name for name, passed in checks.items() if not passed],
    }
