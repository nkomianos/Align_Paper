"""Frozen paired-bootstrap decision rule for the effect-only feasibility gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class FeasibilityOutcomes:
    """Matched binary outcomes, keyed by held-out fork or task identity."""

    effect_ranking: Mapping[str, bool]
    delta_ranking: Mapping[str, bool]
    effect_perturbed_ranking: Mapping[str, bool]
    delta_perturbed_ranking: Mapping[str, bool]
    effect_execution: Mapping[str, bool]
    strongest_baseline_execution: Mapping[str, bool]


def _as_matched_array(left: Mapping[str, bool], right: Mapping[str, bool], *, label: str) -> tuple[np.ndarray, np.ndarray]:
    if set(left) != set(right) or not left:
        raise ValueError(f"{label} requires non-empty, exactly matched item ids")
    ids = sorted(left)
    return (
        np.asarray([bool(left[item]) for item in ids], dtype=np.float64),
        np.asarray([bool(right[item]) for item in ids], dtype=np.float64),
    )


def paired_bootstrap_delta(
    left: Mapping[str, bool],
    right: Mapping[str, bool],
    *,
    label: str,
    seed: int = 202708,
    replicates: int = 20_000,
) -> dict[str, float]:
    """Return a deterministic percentile CI for a paired mean-proportion gap."""

    if replicates < 1_000:
        raise ValueError("at least 1,000 bootstrap replicates are required")
    left_values, right_values = _as_matched_array(left, right, label=label)
    differences = left_values - right_values
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, differences.size, size=(replicates, differences.size))
    bootstrap = differences[draws].mean(axis=1)
    return {
        "point": float(differences.mean()),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "count": float(differences.size),
    }


def assess_feasibility_gate(outcomes: FeasibilityOutcomes) -> dict[str, object]:
    """Apply the immutable continuation rule in the candidate protocol.

    The rank and execution comparisons are paired.  For the nuisance diagnostic,
    the statistic is ``delta perturbation loss - effect perturbation loss``;
    positive values mean centering preserved more ranking accuracy.
    """

    ranking = paired_bootstrap_delta(
        outcomes.effect_ranking, outcomes.delta_ranking, label="ranking", seed=202708,
    )
    effect_clean, effect_noisy = _as_matched_array(
        outcomes.effect_ranking, outcomes.effect_perturbed_ranking, label="effect perturbation",
    )
    delta_clean, delta_noisy = _as_matched_array(
        outcomes.delta_ranking, outcomes.delta_perturbed_ranking, label="delta perturbation",
    )
    if effect_clean.size != delta_clean.size:
        raise ValueError("effect and delta perturbation comparisons require the same item count")
    robustness_left = {str(index): bool(delta_clean[index] and not delta_noisy[index]) for index in range(delta_clean.size)}
    robustness_right = {str(index): bool(effect_clean[index] and not effect_noisy[index]) for index in range(effect_clean.size)}
    # The binary construction above is only a diagnostic for loss events; the
    # point and CI below use the signed, paired accuracy-loss difference.
    _as_matched_array(robustness_left, robustness_right, label="robustness")
    loss_difference = (delta_clean - delta_noisy) - (effect_clean - effect_noisy)
    generator = np.random.default_rng(202709)
    draws = generator.integers(0, loss_difference.size, size=(20_000, loss_difference.size))
    robustness = {
        "point": float(loss_difference.mean()),
        "ci95_low": float(np.quantile(loss_difference[draws].mean(axis=1), 0.025)),
        "ci95_high": float(np.quantile(loss_difference[draws].mean(axis=1), 0.975)),
        "count": float(loss_difference.size),
    }
    execution = paired_bootstrap_delta(
        outcomes.effect_execution, outcomes.strongest_baseline_execution, label="execution", seed=202710,
    )
    non_degeneracy = execution["point"] >= -0.01
    checks = {
        "ranking_effect_size": ranking["point"] >= 0.05,
        "ranking_ci_excludes_zero": ranking["ci95_low"] > 0.0,
        "common_mode_effect_size": robustness["point"] >= 0.05,
        "common_mode_ci_excludes_zero": robustness["ci95_low"] > 0.0,
        "execution_effect_size": execution["point"] >= 0.03,
        "execution_ci_excludes_zero": execution["ci95_low"] > 0.0,
        "non_degeneracy": non_degeneracy,
    }
    return {
        "ranking": ranking,
        "common_mode_robustness": robustness,
        "execution": execution,
        "checks": checks,
        "pass": all(checks.values()),
        "decision": "EXPAND_TO_INDEPENDENT_TOOL_ENVIRONMENT" if all(checks.values()) else "STOP_EFFECT_ONLY_TOOL_WORLD_LINE",
    }
