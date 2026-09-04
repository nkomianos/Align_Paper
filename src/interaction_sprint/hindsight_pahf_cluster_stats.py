"""Cluster-aware statistics for four-rotation EndoPAHF endpoints."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from typing import Mapping, Sequence

import numpy as np


ROTATIONS = {0, 1, 2, 3}
PRIMARY_MIN_NLL_GAIN = 0.05
ACCURACY_NONINFERIORITY = -0.02
BOOTSTRAP_SEED = 2026090421
BOOTSTRAP_SAMPLES = 10_000


def _cluster(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    for row in rows:
        key = (str(row["base_id"]), int(row["rotation"]))
        if key in seen:
            raise ValueError("duplicate base/rotation row")
        seen.add(key)
        grouped[key[0]].append(row)
    result: dict[str, dict[str, float]] = {}
    for base_id, values in grouped.items():
        if {int(row["rotation"]) for row in values} != ROTATIONS or len(values) != 4:
            raise ValueError(f"base does not have four rotations: {base_id}")
        nlls = [float(row["old_target_log_loss"]) for row in values]
        accuracies = [float(bool(row["old_target_correct"])) for row in values]
        if any(not math.isfinite(value) or value < 0 for value in nlls):
            raise ValueError("invalid log loss")
        result[base_id] = {
            "old_target_log_loss": sum(nlls) / 4,
            "old_target_accuracy": sum(accuracies) / 4,
        }
    if not result:
        raise ValueError("no clusters")
    return result


def bootstrap_mean_interval(
    values: Sequence[float], *, samples: int = BOOTSTRAP_SAMPLES, seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or samples < 100:
        raise ValueError("invalid bootstrap inputs")
    if not np.isfinite(array).all():
        raise ValueError("nonfinite bootstrap input")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    chunk = 1000
    for start in range(0, samples, chunk):
        stop = min(start + chunk, samples)
        indices = rng.integers(0, len(array), size=(stop - start, len(array)))
        means[start:stop] = array[indices].mean(axis=1)
    lower, upper = np.quantile(means, (0.025, 0.975))
    return float(lower), float(upper)


def paired_cluster_summary(
    anchor_rows: Sequence[Mapping[str, object]],
    augmented_rows: Sequence[Mapping[str, object]],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    anchor = _cluster(anchor_rows)
    augmented = _cluster(augmented_rows)
    if set(anchor) != set(augmented):
        raise ValueError("paired arms have different base clusters")
    base_ids = sorted(anchor)
    nll_gains = [
        anchor[base_id]["old_target_log_loss"]
        - augmented[base_id]["old_target_log_loss"]
        for base_id in base_ids
    ]
    accuracy_gains = [
        augmented[base_id]["old_target_accuracy"]
        - anchor[base_id]["old_target_accuracy"]
        for base_id in base_ids
    ]
    nll_ci = bootstrap_mean_interval(nll_gains, samples=samples, seed=seed)
    accuracy_ci = bootstrap_mean_interval(accuracy_gains, samples=samples, seed=seed + 1)
    aggregate = {
        "base_clusters": len(base_ids),
        "rotation_rows_per_arm": 4 * len(base_ids),
        "mean_old_target_nll_gain": float(np.mean(nll_gains)),
        "old_target_nll_gain_ci95": list(nll_ci),
        "mean_old_target_accuracy_gain": float(np.mean(accuracy_gains)),
        "old_target_accuracy_gain_ci95": list(accuracy_ci),
        "primary_min_nll_gain": PRIMARY_MIN_NLL_GAIN,
        "accuracy_noninferiority_margin": ACCURACY_NONINFERIORITY,
    }
    gates = {
        "mean_nll_gain_at_least_point05": aggregate["mean_old_target_nll_gain"] >= PRIMARY_MIN_NLL_GAIN,
        "nll_gain_cluster_bootstrap_lower_bound_positive": nll_ci[0] > 0,
        "mean_accuracy_gain_noninferior": aggregate["mean_old_target_accuracy_gain"] >= ACCURACY_NONINFERIORITY,
    }
    return {
        "decision": (
            "ENDO_PAHF_CLUSTERED_COMPARISON_QUALIFIED"
            if all(gates.values()) else "ENDO_PAHF_CLUSTERED_COMPARISON_NOT_QUALIFIED"
        ),
        "gates": gates,
        "aggregate": aggregate,
        "base_ids_sha256": hashlib.sha256("\n".join(base_ids).encode()).hexdigest(),
        "paper_green_light": False,
    }


def power_audit(
    *, n: int = 256, trials: int = 400, bootstrap_samples: int = 1000,
) -> dict[str, object]:
    if n < 2 or trials < 100 or bootstrap_samples < 100:
        raise ValueError("invalid power-audit design")
    scenarios = (
        ("null", 0.00, 0.35),
        ("signal", 0.08, 0.35),
        ("noisy_signal", 0.08, 0.50),
    )
    rates: dict[str, float] = {}
    for scenario_index, (name, mean_gain, standard_deviation) in enumerate(scenarios):
        rng = np.random.default_rng(BOOTSTRAP_SEED + 100 * scenario_index)
        qualified = 0
        for trial in range(trials):
            values = rng.normal(mean_gain, standard_deviation, size=n)
            lower, _ = bootstrap_mean_interval(
                values,
                samples=bootstrap_samples,
                seed=BOOTSTRAP_SEED + 10_000 * scenario_index + trial,
            )
            if float(values.mean()) >= PRIMARY_MIN_NLL_GAIN and lower > 0:
                qualified += 1
        rates[name] = qualified / trials
    gates = {
        "null_false_positive_rate_at_most_point06": rates["null"] <= 0.06,
        "signal_power_at_least_point85": rates["signal"] >= 0.85,
        "noisy_signal_power_at_least_point60": rates["noisy_signal"] >= 0.60,
    }
    return {
        "decision": (
            "ENDO_PAHF_CLUSTER_RULE_POWER_QUALIFIED"
            if all(gates.values()) else "ENDO_PAHF_CLUSTER_RULE_POWER_NOT_QUALIFIED"
        ),
        "design": {
            "base_clusters": n,
            "trials": trials,
            "bootstrap_samples_per_trial": bootstrap_samples,
            "scenarios": [
                {"name": name, "mean_gain": mean_gain, "standard_deviation": standard_deviation}
                for name, mean_gain, standard_deviation in scenarios
            ],
            "minimum_mean_nll_gain": PRIMARY_MIN_NLL_GAIN,
        },
        "qualification_rates": rates,
        "gates": gates,
        "paper_green_light": False,
    }
