"""Matched expression/transition audit with sparse delayed anchors.

This is a finite-state method feasibility check.  It deliberately does not
claim to implement neural SDPO.  The raw learner uses immediate user agreement;
the augmented learner uses the same immediate logs plus a two-phase-sampling
correction estimated from the exact same delayed anchors available to the
anchor-only comparator.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class DelayedAnchorConfig:
    seed: int = 20260904
    repeats: int = 2000
    immediate_per_action: int = 512
    anchors_per_action: tuple[int, ...] = (16, 64)
    p_grid: tuple[float, ...] = (.20, .35, .45, .55, .65, .80)
    copying_grid: tuple[float, ...] = (0., .25, .50, .75, 1.)
    decision_margin: float = .05

    def validate(self) -> None:
        if self.repeats <= 0 or self.immediate_per_action <= 0:
            raise ValueError("sample counts must be positive")
        if not self.anchors_per_action:
            raise ValueError("at least one anchor budget is required")
        if any(k <= 0 or k > self.immediate_per_action for k in self.anchors_per_action):
            raise ValueError("anchor budgets must lie in [1, immediate_per_action]")
        if not self.p_grid or not self.copying_grid:
            raise ValueError("parameter grids cannot be empty")
        if any(not 0 <= value <= 1 for value in (*self.p_grid, *self.copying_grid)):
            raise ValueError("probabilities must lie in [0, 1]")
        if not 0 <= self.decision_margin <= 1:
            raise ValueError("decision_margin must lie in [0, 1]")


def _true_values(p: float, copying: tuple[float, float], mechanism: str) -> np.ndarray:
    if mechanism == "expression":
        return np.asarray([1 - p, p], dtype=np.float64)
    if mechanism == "transition":
        c0, c1 = copying
        return np.asarray([(1 - p) + p * c0, p + (1 - p) * c1], dtype=np.float64)
    raise ValueError("unknown mechanism")


def _policy_from_values(values: np.ndarray) -> np.ndarray:
    """Return probability of action one; exact ties deploy a 50/50 mixture."""
    delta = values[:, 1] - values[:, 0]
    return np.where(delta > 0, 1., np.where(delta < 0, 0., .5))


def _summarize(values: np.ndarray, true_values: np.ndarray) -> dict[str, float]:
    policy_one = _policy_from_values(values)
    deployed = (1 - policy_one) * true_values[0] + policy_one * true_values[1]
    optimum = float(np.max(true_values))
    best = int(true_values[1] > true_values[0])
    correct = np.where(policy_one == .5, .5, policy_one == best)
    return {
        "mean_value": float(np.mean(deployed)),
        "mean_regret": float(np.mean(optimum - deployed)),
        "correct_rate": float(np.mean(correct)),
        "action_one_rate": float(np.mean(policy_one)),
    }


def simulate_cell(
    *,
    p: float,
    copying: tuple[float, float],
    anchors_per_action: int,
    repeats: int,
    immediate_per_action: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    """Simulate a coupled pair whose immediate logs are exactly identical."""
    if not 0 <= p <= 1 or any(not 0 <= c <= 1 for c in copying):
        raise ValueError("probabilities must lie in [0, 1]")
    if not 0 < anchors_per_action <= immediate_per_action or repeats <= 0:
        raise ValueError("invalid sample counts")

    raw = np.empty((repeats, 2), dtype=np.float64)
    anchor_expression = np.empty_like(raw)
    anchor_transition = np.empty_like(raw)
    augmented_expression = np.empty_like(raw)
    augmented_transition = np.empty_like(raw)

    for action in (0, 1):
        z = rng.binomial(1, p, size=(repeats, anchors_per_action))
        copy = rng.random(size=z.shape) < copying[action]
        immediate_agree = (z == action) | ((z != action) & copy)
        baseline_agree = z == action
        immediate_probability = (
            ((1 - p) if action == 0 else p)
            + ((p if action == 0 else 1 - p) * copying[action])
        )
        remaining = rng.binomial(
            immediate_per_action - anchors_per_action,
            immediate_probability,
            size=repeats,
        )
        mean_immediate = (np.sum(immediate_agree, axis=1) + remaining) / immediate_per_action
        mean_expression_anchor = np.mean(baseline_agree, axis=1)
        mean_transition_anchor = np.mean(immediate_agree, axis=1)

        raw[:, action] = mean_immediate
        anchor_expression[:, action] = mean_expression_anchor
        anchor_transition[:, action] = mean_transition_anchor
        # Difference estimator for two-phase samples:
        # E[B] = E[O] + E[B-O].  Both methods use the identical B sample.
        residual = baseline_agree.astype(np.float64) - immediate_agree.astype(np.float64)
        augmented_expression[:, action] = np.clip(
            mean_immediate + np.mean(residual, axis=1), 0., 1.
        )
        augmented_transition[:, action] = np.clip(
            mean_immediate, 0., 1.
        )

    true_expression = _true_values(p, copying, "expression")
    true_transition = _true_values(p, copying, "transition")
    # The two observational worlds receive distinct arrays populated from the
    # same coupled immediate outcomes.  Keeping both makes an accidental future
    # divergence detectable rather than defining the mismatch as a constant.
    raw_expression = raw.copy()
    raw_transition = raw.copy()
    raw_expression_policy = _policy_from_values(raw_expression)
    raw_transition_policy = _policy_from_values(raw_transition)
    return {
        "p": p,
        "copying": list(copying),
        "anchors_per_action": anchors_per_action,
        "true_values": {
            "expression": true_expression.tolist(),
            "transition": true_transition.tolist(),
        },
        "true_margin": {
            "expression": float(abs(true_expression[1] - true_expression[0])),
            "transition": float(abs(true_transition[1] - true_transition[0])),
        },
        "best_action": {
            "expression": int(true_expression[1] > true_expression[0]),
            "transition": int(true_transition[1] > true_transition[0]),
        },
        "methods": {
            "expression": {
                "raw_immediate": _summarize(raw_expression, true_expression),
                "anchor_only": _summarize(anchor_expression, true_expression),
                "augmented": _summarize(augmented_expression, true_expression),
            },
            "transition": {
                "raw_immediate": _summarize(raw_transition, true_transition),
                "anchor_only": _summarize(anchor_transition, true_transition),
                "augmented": _summarize(augmented_transition, true_transition),
            },
        },
        "coupling": {
            # The same raw matrix is used in both worlds by construction.  Keep
            # this explicit so the verifier tests the matched-log invariant.
            "raw_policy_mismatch_rate": float(np.mean(raw_expression_policy != raw_transition_policy)),
            "truthful_augmented_raw_mismatch_rate": (
                float(np.mean(_policy_from_values(augmented_expression) != raw_expression_policy))
                if copying == (0., 0.) else None
            ),
        },
    }


def _mean_metric(
    cells: Iterable[dict[str, object]], mechanism: str, method: str, metric: str
) -> float:
    values = [cell["methods"][mechanism][method][metric] for cell in cells]  # type: ignore[index]
    return float(np.mean(values)) if values else float("nan")


def run_audit(config: DelayedAnchorConfig = DelayedAnchorConfig()) -> dict[str, object]:
    config.validate()
    rng = np.random.default_rng(config.seed)
    rows: list[dict[str, object]] = []
    for anchors in config.anchors_per_action:
        for p in config.p_grid:
            for c0 in config.copying_grid:
                for c1 in config.copying_grid:
                    rows.append(simulate_cell(
                        p=p,
                        copying=(c0, c1),
                        anchors_per_action=anchors,
                        repeats=config.repeats,
                        immediate_per_action=config.immediate_per_action,
                        rng=rng,
                    ))

    aggregate: dict[str, object] = {}
    for anchors in config.anchors_per_action:
        budget_cells = [row for row in rows if row["anchors_per_action"] == anchors]
        expression_informative = [
            row for row in budget_cells
            if row["true_margin"]["expression"] >= config.decision_margin  # type: ignore[index]
        ]
        transition_informative = [
            row for row in budget_cells
            if row["true_margin"]["transition"] >= config.decision_margin  # type: ignore[index]
        ]
        reversals = [
            row for row in budget_cells
            if row["best_action"]["expression"] != row["best_action"]["transition"]  # type: ignore[index]
            and row["true_margin"]["expression"] >= config.decision_margin  # type: ignore[index]
            and row["true_margin"]["transition"] >= config.decision_margin  # type: ignore[index]
        ]
        truthful = [row for row in budget_cells if row["copying"] == [0., 0.]]
        aggregate[str(anchors)] = {
            "cell_counts": {
                "all": len(budget_cells),
                "expression_informative": len(expression_informative),
                "transition_informative": len(transition_informative),
                "ranking_reversals": len(reversals),
                "truthful_controls": len(truthful),
            },
            "expression_all_informative_mean_regret": {
                method: _mean_metric(expression_informative, "expression", method, "mean_regret")
                for method in ("raw_immediate", "anchor_only", "augmented")
            },
            "transition_all_informative_mean_regret": {
                method: _mean_metric(transition_informative, "transition", method, "mean_regret")
                for method in ("raw_immediate", "anchor_only", "augmented")
            },
            "expression_reversal_mean_regret": {
                method: _mean_metric(reversals, "expression", method, "mean_regret")
                for method in ("raw_immediate", "anchor_only", "augmented")
            },
            "max_raw_policy_mismatch_rate": max(
                row["coupling"]["raw_policy_mismatch_rate"] for row in budget_cells  # type: ignore[index]
            ),
            "max_truthful_augmented_raw_mismatch_rate": max(
                row["coupling"]["truthful_augmented_raw_mismatch_rate"] for row in truthful  # type: ignore[index]
            ),
        }

    small = aggregate[str(min(config.anchors_per_action))]  # type: ignore[assignment]
    large = aggregate[str(max(config.anchors_per_action))]  # type: ignore[assignment]
    small_expression = small["expression_all_informative_mean_regret"]  # type: ignore[index]
    small_transition = small["transition_all_informative_mean_regret"]  # type: ignore[index]
    large_reversal = large["expression_reversal_mean_regret"]  # type: ignore[index]
    gates = {
        "matched_immediate_policies_exact": all(
            summary["max_raw_policy_mismatch_rate"] == 0 for summary in aggregate.values()  # type: ignore[index]
        ),
        "problem_is_nontrivial": (
            small["expression_reversal_mean_regret"]["raw_immediate"] >= .03  # type: ignore[index]
        ),
        "large_anchor_corrects_expression_reversals": (
            large_reversal["augmented"] <= .60 * large_reversal["raw_immediate"]
        ),
        "small_anchor_beats_equal_anchor_baseline": (
            small_expression["augmented"] <= .80 * small_expression["anchor_only"]
        ),
        "transition_preserves_immediate_information": (
            small_transition["augmented"] <= .50 * small_transition["anchor_only"]
        ),
        "truthful_correction_positive_control_exact": all(
            summary["max_truthful_augmented_raw_mismatch_rate"] == 0  # type: ignore[index]
            for summary in aggregate.values()
        ),
    }
    return {
        "decision": "DEV_METHOD_SIGNAL_QUALIFIED" if all(gates.values()) else "DEV_METHOD_SIGNAL_NOT_QUALIFIED",
        "config": asdict(config),
        "estimand": "probability deployed action matches delayed latent preference under randomized action",
        "methods": {
            "raw_immediate": "mean immediate-report agreement",
            "anchor_only": "mean delayed-anchor agreement using k anchors per action",
            "augmented": "mean immediate agreement plus mean anchored delayed-minus-immediate residual",
        },
        "gates": gates,
        "aggregate": aggregate,
        "rows": rows,
        "scope": (
            "Finite-state developmental method gate. It tests identification and sample reuse, not neural SDPO, "
            "human prevalence, welfare, or ICLR viability. Equal-anchor means anchor-only and augmented use the "
            "same delayed measurements; augmented additionally uses abundant immediate logs."
        ),
    }
