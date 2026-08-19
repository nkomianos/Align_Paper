"""Trajectory-level prospective tests for the paper-critical prediction claim.

Checkpoint rows are deliberately never treated as independent observations.  Every
outer fold holds out a complete RL trajectory, and uncertainty is computed over
trajectory-level loss differences.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from typing import Any, Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_BASELINE_FEATURES = (
    "checkpoint_fraction",
    "current_hack_rate",
    "cumulative_proxy_reward",
    "current_reward_gap",
    "recent_proxy_reward_slope",
    "recent_hack_rate_slope",
)
DEFAULT_FINGERPRINT_FEATURES = ("proxy_control_index",)


def _finite_float(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return result


def validate_rows(
    rows: Iterable[dict[str, Any]],
    *,
    baseline_features: tuple[str, ...] = DEFAULT_BASELINE_FEATURES,
    fingerprint_features: tuple[str, ...] = DEFAULT_FINGERPRINT_FEATURES,
) -> list[dict[str, Any]]:
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError("Prospective analysis received no rows")
    seen: set[tuple[str, int]] = set()
    by_trajectory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    required = set(baseline_features) | set(fingerprint_features)
    for row in materialized:
        trajectory = str(row.get("trajectory_id", ""))
        if not trajectory:
            raise ValueError("Every row requires a non-empty trajectory_id")
        step = int(row["checkpoint_step"])
        key = (trajectory, step)
        if key in seen:
            raise ValueError(f"Duplicate trajectory/checkpoint row: {key}")
        seen.add(key)
        outcome = int(row["future_sustained_hack"])
        if outcome not in (0, 1):
            raise ValueError("future_sustained_hack must be binary")
        if int(row.get("forecast_horizon_steps", 0)) <= 0:
            raise ValueError("forecast_horizon_steps must be positive and frozen")
        if not bool(row.get("eligible", True)):
            raise ValueError("Censored/ineligible rows must be excluded before analysis")
        missing = required - set(row)
        if missing:
            raise ValueError(f"Row {key} is missing features: {sorted(missing)}")
        for feature in required:
            _finite_float(row[feature], feature)
        by_trajectory[trajectory].append(row)
    horizons = {int(row["forecast_horizon_steps"]) for row in materialized}
    if len(horizons) != 1:
        raise ValueError(f"Forecast horizon changed within the study: {sorted(horizons)}")
    if len(by_trajectory) < 4:
        raise ValueError("At least four independent trajectories are required even for an engineering analysis")
    return sorted(materialized, key=lambda row: (str(row["trajectory_id"]), int(row["checkpoint_step"])))


def _matrix(rows: list[dict[str, Any]], features: tuple[str, ...]) -> np.ndarray:
    return np.asarray([[float(row[name]) for name in features] for row in rows], dtype=np.float64)


def _fit_predict(
    train: list[dict[str, Any]], test: list[dict[str, Any]], features: tuple[str, ...], regularization_c: float
) -> np.ndarray:
    y_train = np.asarray([int(row["future_sustained_hack"]) for row in train], dtype=np.int64)
    if set(y_train.tolist()) != {0, 1}:
        raise ValueError("Every outer training fold must contain both future-outcome classes")
    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=regularization_c, solver="lbfgs", max_iter=2000),
    )
    estimator.fit(_matrix(train, features), y_train)
    return estimator.predict_proba(_matrix(test, features))[:, 1]


def _sign_flip_pvalue(differences: np.ndarray, seed: int, draws: int) -> float:
    """One-sided randomization test for mean improvement above zero."""
    n = len(differences)
    observed = float(np.mean(differences))
    if n <= 20:
        assignments = np.asarray(list(itertools.product((-1.0, 1.0), repeat=n)), dtype=np.float64)
        null = (assignments * differences[None, :]).mean(axis=1)
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice((-1.0, 1.0), size=(max(draws, 1), n))
        null = (signs * differences[None, :]).mean(axis=1)
    return float((1.0 + np.sum(null >= observed - 1e-15)) / (len(null) + 1.0))


def compare_models_loto(
    rows: Iterable[dict[str, Any]],
    *,
    baseline_features: tuple[str, ...] = DEFAULT_BASELINE_FEATURES,
    fingerprint_features: tuple[str, ...] = DEFAULT_FINGERPRINT_FEATURES,
    regularization_c: float = 1.0,
    bootstrap_replicates: int = 10_000,
    seed: int = 1729,
    minimum_log_loss_improvement: float = 0.01,
) -> dict[str, Any]:
    """Compare a frozen baseline to baseline+fingerprint under trajectory LOTO."""
    if regularization_c <= 0 or bootstrap_replicates <= 0:
        raise ValueError("regularization_c and bootstrap_replicates must be positive")
    validated = validate_rows(
        rows, baseline_features=baseline_features, fingerprint_features=fingerprint_features
    )
    trajectories = sorted({str(row["trajectory_id"]) for row in validated})
    augmented_features = baseline_features + fingerprint_features
    predictions: list[dict[str, Any]] = []
    trajectory_differences: list[float] = []
    for held_out in trajectories:
        train = [row for row in validated if str(row["trajectory_id"]) != held_out]
        test = [row for row in validated if str(row["trajectory_id"]) == held_out]
        baseline_probability = _fit_predict(train, test, baseline_features, regularization_c)
        augmented_probability = _fit_predict(train, test, augmented_features, regularization_c)
        truth = np.asarray([int(row["future_sustained_hack"]) for row in test], dtype=np.int64)
        baseline_loss = float(log_loss(truth, baseline_probability, labels=[0, 1]))
        augmented_loss = float(log_loss(truth, augmented_probability, labels=[0, 1]))
        trajectory_differences.append(baseline_loss - augmented_loss)
        for row, base_p, augmented_p in zip(test, baseline_probability, augmented_probability, strict=True):
            predictions.append({
                "trajectory_id": held_out,
                "checkpoint_step": int(row["checkpoint_step"]),
                "future_sustained_hack": int(row["future_sustained_hack"]),
                "baseline_probability": float(base_p),
                "augmented_probability": float(augmented_p),
            })
    differences = np.asarray(trajectory_differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = differences[rng.integers(0, len(differences), size=(bootstrap_replicates, len(differences)))].mean(axis=1)
    improvement = float(np.mean(differences))
    interval = [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))]
    truth = np.asarray([row["future_sustained_hack"] for row in predictions], dtype=np.int64)
    base_probability = np.asarray([row["baseline_probability"] for row in predictions], dtype=np.float64)
    augmented_probability = np.asarray([row["augmented_probability"] for row in predictions], dtype=np.float64)
    descriptive_aurocs = {
        "baseline": float(roc_auc_score(truth, base_probability)) if len(set(truth.tolist())) == 2 else None,
        "augmented": float(roc_auc_score(truth, augmented_probability)) if len(set(truth.tolist())) == 2 else None,
    }
    p_value = _sign_flip_pvalue(differences, seed + 1, bootstrap_replicates)
    checks = {
        "minimum_independent_trajectories": len(trajectories) >= 12,
        "positive_bootstrap_lower_bound": interval[0] > 0.0,
        "minimum_mean_log_loss_improvement": improvement >= minimum_log_loss_improvement,
        "trajectory_randomization_p_value": p_value < 0.05,
    }
    return {
        "analysis_unit": "independent_rl_trajectory",
        "trajectory_count": len(trajectories),
        "checkpoint_row_count": len(validated),
        "forecast_horizon_steps": int(validated[0]["forecast_horizon_steps"]),
        "baseline_features": list(baseline_features),
        "fingerprint_features": list(fingerprint_features),
        "regularization_c": regularization_c,
        "mean_log_loss_improvement": improvement,
        "trajectory_bootstrap_ci95": interval,
        "one_sided_sign_flip_p_value": p_value,
        "descriptive_loto_auroc": descriptive_aurocs,
        "trajectory_log_loss_improvements": dict(zip(trajectories, trajectory_differences, strict=True)),
        "gate_c": {"pass": all(checks.values()), "checks": checks},
        "predictions": predictions,
    }


def simulate_trajectory_power(
    *,
    trajectory_counts: tuple[int, ...] = (8, 12, 16, 24),
    standardized_fingerprint_effect: float = 0.8,
    checkpoints_per_trajectory: int = 6,
    simulations: int = 2_000,
    alpha: float = 0.05,
    seed: int = 1731,
) -> dict[str, float]:
    """Cheap preregistration aid; this is not a substitute for empirical power."""
    if simulations <= 0 or checkpoints_per_trajectory <= 0 or standardized_fingerprint_effect < 0:
        raise ValueError("Invalid power-simulation arguments")
    rng = np.random.default_rng(seed)
    output: dict[str, float] = {}
    for count in trajectory_counts:
        if count <= 1:
            raise ValueError("trajectory counts must exceed one")
        successes = 0
        for _ in range(simulations):
            latent = rng.normal(size=count)
            noise = rng.normal(scale=1.0 / math.sqrt(checkpoints_per_trajectory), size=count)
            paired_improvement = standardized_fingerprint_effect + 0.5 * latent + noise
            standard_error = float(np.std(paired_improvement, ddof=1) / math.sqrt(count))
            z = float(np.mean(paired_improvement) / standard_error) if standard_error > 0 else math.inf
            if z > 1.6448536269514722 and alpha == 0.05:
                successes += 1
            elif alpha != 0.05:
                from scipy.stats import norm

                if z > float(norm.ppf(1.0 - alpha)):
                    successes += 1
        output[str(count)] = successes / simulations
    return output
