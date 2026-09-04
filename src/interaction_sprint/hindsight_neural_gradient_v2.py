"""Nested-budget metrics for the prospectively corrected neural gradient G0."""
from __future__ import annotations

import hashlib

import torch

from interaction_sprint.hindsight_neural_gradient import PANEL_COUNT, summarize_gradients


ANCHOR_BUDGETS = (4, 8)


def build_nested_anchor_panels(rows: list[dict[str, object]]) -> dict[int, list[list[str]]]:
    """Build nested outcome-blind panels balanced only on logged action."""
    if len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("row ids must be unique")
    result = {budget: [] for budget in ANCHOR_BUDGETS}
    for panel_index in range(PANEL_COUNT):
        ordered_by_action: dict[int, list[dict[str, object]]] = {}
        for action in (0, 1):
            candidates = [row for row in rows if row["logged_action"] == action]
            candidates.sort(key=lambda row: hashlib.sha256(
                f"{row['id']}|gradient-panel-{panel_index}".encode()
            ).hexdigest())
            if len(candidates) < max(ANCHOR_BUDGETS) // 2:
                raise ValueError("too few records for action-balanced panels")
            ordered_by_action[action] = candidates
        for budget in ANCHOR_BUDGETS:
            per_action = budget // 2
            panel = [
                str(row["id"])
                for action in (0, 1)
                for row in ordered_by_action[action][:per_action]
            ]
            if len(panel) != budget or len(set(panel)) != budget:
                raise AssertionError("invalid nested panel")
            result[budget].append(panel)
    for panel_index in range(PANEL_COUNT):
        if not set(result[4][panel_index]).issubset(result[8][panel_index]):
            raise AssertionError("four-anchor panel is not nested in eight-anchor panel")
    return result


def _single_budget_vectors(
    vectors: dict[str, torch.Tensor], budget: int,
) -> dict[str, torch.Tensor]:
    selected = {
        "oracle_delayed": vectors["oracle_delayed"],
        "raw_immediate": vectors["raw_immediate"],
    }
    for panel_index in range(PANEL_COUNT):
        for field in ("delayed", "immediate"):
            selected[f"panel_{panel_index:02d}_{field}"] = vectors[
                f"budget_{budget:02d}_panel_{panel_index:02d}_{field}"
            ]
    return selected


def summarize_nested_gradients(vectors: dict[str, torch.Tensor]) -> dict[str, object]:
    required = {"oracle_delayed", "raw_immediate"}
    required |= {
        f"budget_{budget:02d}_panel_{panel_index:02d}_{field}"
        for budget in ANCHOR_BUDGETS
        for panel_index in range(PANEL_COUNT)
        for field in ("delayed", "immediate")
    }
    if set(vectors) != required:
        raise ValueError("gradient vector keys do not match the nested design")
    summaries = {
        budget: summarize_gradients(_single_budget_vectors(vectors, budget))
        for budget in ANCHOR_BUDGETS
    }
    first = summaries[ANCHOR_BUDGETS[0]]
    gates: dict[str, bool] = {
        "oracle_gradient_is_nonzero": bool(first["gates"]["oracle_gradient_is_nonzero"]),
        "raw_immediate_conflicts_with_delayed_oracle": bool(
            first["gates"]["raw_immediate_conflicts_with_delayed_oracle"]
        ),
    }
    for budget, summary in summaries.items():
        gates[f"budget_{budget:02d}_augmented_median_error_reduction"] = bool(
            summary["gates"]["augmented_median_error_reduction"]
        )
        gates[f"budget_{budget:02d}_augmented_wins_at_least_six_panels"] = bool(
            summary["gates"]["augmented_wins_at_least_six_panels"]
        )
        gates[f"budget_{budget:02d}_augmented_mean_error_reduction"] = bool(
            summary["gates"]["augmented_mean_error_reduction"]
        )
    return {
        "decision": (
            "NEURAL_GRADIENT_G0_V2_QUALIFIED"
            if all(gates.values()) else "NEURAL_GRADIENT_G0_V2_NOT_QUALIFIED"
        ),
        "gates": gates,
        "budgets": {
            f"{budget:02d}": {
                "aggregate": summary["aggregate"],
                "panels": summary["panels"],
                "cosine_diagnostic_pass": summary["gates"]["augmented_median_cosine_improves"],
            }
            for budget, summary in summaries.items()
        },
        "scope": (
            "Initial-adapter nested-budget neural gradient-estimator gate. Fixed absolute cosine "
            "gain is reported diagnostically after a prospective ceiling-effect audit; policy "
            "learning and paper viability remain untested."
        ),
    }
