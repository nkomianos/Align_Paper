"""Prospective operating-characteristic audit for the neural gradient G0."""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch

from interaction_sprint.hindsight_neural_anchor import build_records
from interaction_sprint.hindsight_neural_gradient import build_anchor_panels, summarize_gradients


SEED = 2026090411
REPLICATES = 500
DIMENSION = 64
RAW_ORACLE_COSINE = -0.60
NOISE_SCALES = (0.25, 0.5, 1.0, 2.0)
RESIDUAL_NOISE_RATIOS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def _quantized_mean(values: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(values.mean(axis=0)).to(torch.bfloat16)


def _panel_indices() -> list[list[int]]:
    rows, _, _ = build_records()
    panels = build_anchor_panels(rows)
    index = {str(row["id"]): position for position, row in enumerate(rows)}
    return [[index[row_id] for row_id in panel] for panel in panels]


def run_power_audit(
    *,
    seed: int = SEED,
    replicates: int = REPLICATES,
    dimension: int = DIMENSION,
) -> dict[str, object]:
    """Estimate frozen-gate pass rates under a transparent gradient model.

    Per-example immediate gradients contain isotropic noise ``X`` and delayed
    gradients contain ``X + R``.  The augmented estimator cancels ``X`` on the
    sparse panel, leaving only residual noise ``R``.  Full-population means are
    centered exactly, so the audit varies estimator noise rather than whether
    the constructed oracle and raw gradients satisfy the conflict premise.
    """
    if replicates <= 0 or dimension < 2:
        raise ValueError("replicates must be positive and dimension at least two")
    rng = np.random.default_rng(seed)
    panel_indices = _panel_indices()
    n = len(build_records()[0])
    delayed_mean = np.zeros(dimension, dtype=np.float32)
    delayed_mean[0] = 1.0
    immediate_mean = np.zeros(dimension, dtype=np.float32)
    immediate_mean[0] = RAW_ORACLE_COSINE
    immediate_mean[1] = float(np.sqrt(1.0 - RAW_ORACLE_COSINE**2))

    cells: list[dict[str, object]] = []
    for noise_scale in NOISE_SCALES:
        coordinate_sd = noise_scale / np.sqrt(dimension)
        for residual_ratio in RESIDUAL_NOISE_RATIOS:
            decisions = 0
            gate_counts: Counter[str] = Counter()
            median_error_ratios: list[float] = []
            median_cosine_improvements: list[float] = []
            wins: list[int] = []
            for _ in range(replicates):
                immediate_noise = rng.normal(0.0, coordinate_sd, size=(n, dimension)).astype(np.float32)
                residual_noise = rng.normal(
                    0.0,
                    coordinate_sd * residual_ratio,
                    size=(n, dimension),
                ).astype(np.float32)
                immediate_noise -= immediate_noise.mean(axis=0, keepdims=True)
                residual_noise -= residual_noise.mean(axis=0, keepdims=True)
                immediate = immediate_mean + immediate_noise
                delayed = delayed_mean + immediate_noise + residual_noise
                vectors: dict[str, torch.Tensor] = {
                    "oracle_delayed": _quantized_mean(delayed),
                    "raw_immediate": _quantized_mean(immediate),
                }
                for panel_index, selected in enumerate(panel_indices):
                    vectors[f"panel_{panel_index:02d}_delayed"] = _quantized_mean(delayed[selected])
                    vectors[f"panel_{panel_index:02d}_immediate"] = _quantized_mean(immediate[selected])
                summary = summarize_gradients(vectors)
                decisions += summary["decision"] == "NEURAL_GRADIENT_G0_QUALIFIED"
                for name, passed in summary["gates"].items():
                    gate_counts[name] += bool(passed)
                aggregate = summary["aggregate"]
                median_error_ratios.append(
                    aggregate["median_augmented_relative_error"]
                    / aggregate["median_anchor_relative_error"]
                )
                median_cosine_improvements.append(
                    aggregate["median_augmented_cosine"] - aggregate["median_anchor_cosine"]
                )
                wins.append(int(aggregate["augmented_error_wins"]))
            theoretical_rmse_ratio = residual_ratio / np.sqrt(1.0 + residual_ratio**2)
            cells.append({
                "noise_scale": noise_scale,
                "residual_noise_ratio": residual_ratio,
                "theoretical_rmse_ratio": float(theoretical_rmse_ratio),
                "qualified_rate": decisions / replicates,
                "gate_pass_rates": {
                    name: count / replicates for name, count in sorted(gate_counts.items())
                },
                "median_observed_error_ratio": float(np.median(median_error_ratios)),
                "median_cosine_improvement": float(np.median(median_cosine_improvements)),
                "median_panel_wins": float(np.median(wins)),
            })

    return {
        "design": {
            "seed": seed,
            "replicates_per_cell": replicates,
            "dimension": dimension,
            "population_size": n,
            "panel_count": len(panel_indices),
            "panel_size": len(panel_indices[0]),
            "raw_oracle_cosine": RAW_ORACLE_COSINE,
            "noise_scales": list(NOISE_SCALES),
            "residual_noise_ratios": list(RESIDUAL_NOISE_RATIOS),
            "storage_dtype": "bfloat16 before gate metrics",
        },
        "panel_overlap": [
            [len(set(left) & set(right)) for right in panel_indices]
            for left in panel_indices
        ],
        "cells": cells,
        "scope": (
            "Parametric operating-characteristic audit of the already-frozen neural gradient gates. "
            "It is not neural evidence and cannot qualify the Hindsight hypothesis."
        ),
    }
