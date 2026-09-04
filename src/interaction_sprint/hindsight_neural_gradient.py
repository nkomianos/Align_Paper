"""Pure panel construction and metrics for the neural anchor-gradient G0."""
from __future__ import annotations

import hashlib

import numpy as np
import torch


PANEL_COUNT = 8
ANCHORS_PER_PANEL = 8
ANCHORS_PER_ACTION = 4


def build_anchor_panels(rows: list[dict[str, object]]) -> list[list[str]]:
    """Outcome-blind hash panels, balanced only on randomized logged action."""
    if len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("row ids must be unique")
    panels = []
    for panel_index in range(PANEL_COUNT):
        panel: list[str] = []
        for action in (0, 1):
            candidates = [row for row in rows if row["logged_action"] == action]
            candidates.sort(key=lambda row: hashlib.sha256(
                f"{row['id']}|gradient-panel-{panel_index}".encode()
            ).hexdigest())
            if len(candidates) < ANCHORS_PER_ACTION:
                raise ValueError("too few records for action-balanced panels")
            panel.extend(str(row["id"]) for row in candidates[:ANCHORS_PER_ACTION])
        panels.append(panel)
    if any(len(panel) != ANCHORS_PER_PANEL or len(set(panel)) != ANCHORS_PER_PANEL for panel in panels):
        raise AssertionError("invalid anchor panel")
    return panels


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise ValueError("gradient vectors must have matching rank-one shape")
    denominator = float(torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right))
    if denominator == 0:
        return 0.
    return float(torch.dot(left, right) / denominator)


def relative_error(estimate: torch.Tensor, target: torch.Tensor) -> float:
    if estimate.ndim != 1 or target.ndim != 1 or estimate.shape != target.shape:
        raise ValueError("gradient vectors must have matching rank-one shape")
    denominator = float(torch.linalg.vector_norm(target))
    if denominator == 0:
        return float("inf")
    return float(torch.linalg.vector_norm(estimate - target) / denominator)


def summarize_gradients(vectors: dict[str, torch.Tensor]) -> dict[str, object]:
    required = {"oracle_delayed", "raw_immediate"}
    required |= {f"panel_{index:02d}_delayed" for index in range(PANEL_COUNT)}
    required |= {f"panel_{index:02d}_immediate" for index in range(PANEL_COUNT)}
    if set(vectors) != required:
        raise ValueError("gradient vector keys do not match the frozen design")
    float_vectors = {name: value.float() for name, value in vectors.items()}
    shapes = {tuple(value.shape) for value in float_vectors.values()}
    if len(shapes) != 1:
        raise ValueError("gradient vector shapes differ or are empty")
    common_shape = next(iter(shapes))
    if len(common_shape) != 1 or common_shape[0] == 0:
        raise ValueError("gradient vector shapes differ or are empty")
    if any(value.ndim != 1 or not bool(torch.isfinite(value).all()) for value in float_vectors.values()):
        raise ValueError("gradients must be finite vectors")

    oracle = float_vectors["oracle_delayed"]
    raw = float_vectors["raw_immediate"]
    oracle_norm = float(torch.linalg.vector_norm(oracle))
    panel_rows = []
    anchor_vectors = []
    augmented_vectors = []
    for index in range(PANEL_COUNT):
        delayed = float_vectors[f"panel_{index:02d}_delayed"]
        immediate = float_vectors[f"panel_{index:02d}_immediate"]
        augmented = raw + delayed - immediate
        anchor_vectors.append(delayed)
        augmented_vectors.append(augmented)
        anchor_error = relative_error(delayed, oracle)
        augmented_error = relative_error(augmented, oracle)
        panel_rows.append({
            "panel": index,
            "anchor_relative_error": anchor_error,
            "augmented_relative_error": augmented_error,
            "anchor_cosine": cosine(delayed, oracle),
            "augmented_cosine": cosine(augmented, oracle),
            "augmented_wins_error": augmented_error < anchor_error,
        })

    anchor_errors = [row["anchor_relative_error"] for row in panel_rows]
    augmented_errors = [row["augmented_relative_error"] for row in panel_rows]
    anchor_cosines = [row["anchor_cosine"] for row in panel_rows]
    augmented_cosines = [row["augmented_cosine"] for row in panel_rows]
    mean_anchor = torch.stack(anchor_vectors).mean(0)
    mean_augmented = torch.stack(augmented_vectors).mean(0)
    aggregate = {
        "vector_dimension": oracle.numel(),
        "oracle_gradient_norm": oracle_norm,
        "raw_gradient_norm": float(torch.linalg.vector_norm(raw)),
        "raw_oracle_cosine": cosine(raw, oracle),
        "median_anchor_relative_error": float(np.median(anchor_errors)),
        "median_augmented_relative_error": float(np.median(augmented_errors)),
        "median_anchor_cosine": float(np.median(anchor_cosines)),
        "median_augmented_cosine": float(np.median(augmented_cosines)),
        "augmented_error_wins": sum(bool(row["augmented_wins_error"]) for row in panel_rows),
        "mean_anchor_relative_error": relative_error(mean_anchor, oracle),
        "mean_augmented_relative_error": relative_error(mean_augmented, oracle),
    }
    gates = {
        "oracle_gradient_is_nonzero": oracle_norm >= 1e-6,
        "raw_immediate_conflicts_with_delayed_oracle": aggregate["raw_oracle_cosine"] <= -.25,
        "augmented_median_error_reduction": (
            aggregate["median_augmented_relative_error"]
            <= .80 * aggregate["median_anchor_relative_error"]
        ),
        "augmented_wins_at_least_six_panels": aggregate["augmented_error_wins"] >= 6,
        "augmented_mean_error_reduction": (
            aggregate["mean_augmented_relative_error"]
            <= .80 * aggregate["mean_anchor_relative_error"]
        ),
        "augmented_median_cosine_improves": (
            aggregate["median_augmented_cosine"] >= aggregate["median_anchor_cosine"] + .05
        ),
    }
    return {
        "decision": "NEURAL_GRADIENT_G0_QUALIFIED" if all(gates.values()) else "NEURAL_GRADIENT_G0_NOT_QUALIFIED",
        "gates": gates,
        "aggregate": aggregate,
        "panels": panel_rows,
        "scope": (
            "Initial-adapter neural gradient-estimator gate. It tests full-vocabulary first-token SDPO "
            "gradient fidelity under eight outcome-blind sparse panels, not policy learning or paper viability."
        ),
    }
