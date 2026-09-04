"""Selection and contamination stress test for delayed-anchor correction.

The target is the expression-world initial preference.  Delayed anchors may be
contaminated toward the immediate report, and their observation probability may
depend on that report.  Known-propensity Horvitz--Thompson estimators identify
the contaminated anchor target under report-conditional missingness; they do
not recover an uncontaminated target after the anchor itself has changed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class AnchorRobustnessConfig:
    seed: int = 2026090419
    repeats: int = 1000
    observations_per_action: int = 512
    base_anchor_rate: float = .125
    p_grid: tuple[float, ...] = (.55, .65, .75, .85)
    c0_grid: tuple[float, ...] = (.75, 1.)
    c1_grid: tuple[float, ...] = (0., .25)
    contamination_grid: tuple[float, ...] = (0., .10, .25, .50, .75)
    selection_bias_grid: tuple[float, ...] = (0., .25, .50, .75)
    informative_margin: float = .05

    def validate(self) -> None:
        if self.repeats <= 0 or self.observations_per_action <= 0:
            raise ValueError("sample counts must be positive")
        if not 0 < self.base_anchor_rate < 1:
            raise ValueError("base anchor rate must lie in (0,1)")
        grids = (
            self.p_grid, self.c0_grid, self.c1_grid,
            self.contamination_grid, self.selection_bias_grid,
        )
        if any(not grid for grid in grids):
            raise ValueError("probability grids cannot be empty")
        if any(not 0 <= value <= 1 for grid in grids for value in grid):
            raise ValueError("grid probabilities must lie in [0,1]")
        if any(self.base_anchor_rate * (1 + bias) > 1 for bias in self.selection_bias_grid):
            raise ValueError("selection propensity exceeds one")
        if not 0 <= self.informative_margin <= 1:
            raise ValueError("informative margin must lie in [0,1]")


def analytic_values(
    p: float, copying: tuple[float, float], contamination: float,
) -> dict[str, np.ndarray]:
    """Return baseline, immediate, and contaminated-anchor agreement values."""
    if not 0 <= p <= 1 or any(not 0 <= value <= 1 for value in (*copying, contamination)):
        raise ValueError("probabilities must lie in [0,1]")
    baseline = np.asarray([1 - p, p], dtype=np.float64)
    immediate = baseline + (1 - baseline) * np.asarray(copying, dtype=np.float64)
    anchor = (1 - contamination) * baseline + contamination * immediate
    return {"baseline": baseline, "immediate": immediate, "anchor": anchor}


def contamination_flip_point(p: float, copying: tuple[float, float]) -> float | None:
    """First contamination level where the anchor target ceases to favor action 1."""
    values0 = analytic_values(p, copying, 0.)
    baseline_delta = float(values0["baseline"][1] - values0["baseline"][0])
    immediate_delta = float(values0["immediate"][1] - values0["immediate"][0])
    if baseline_delta <= 0:
        raise ValueError("this expression-world audit requires p > .5")
    if immediate_delta >= 0:
        return None
    root = baseline_delta / (baseline_delta - immediate_delta)
    return float(root) if 0 <= root <= 1 else None


def category_probabilities(
    baseline_agreement: float, copying: float, contamination: float,
) -> np.ndarray:
    """Probabilities in order (O0,B0), (O0,B1), (O1,B0), (O1,B1)."""
    if any(not 0 <= value <= 1 for value in (baseline_agreement, copying, contamination)):
        raise ValueError("probabilities must lie in [0,1]")
    mismatch = 1 - baseline_agreement
    probabilities = np.asarray([
        mismatch * (1 - copying),
        0.,
        mismatch * copying * (1 - contamination),
        baseline_agreement + mismatch * copying * contamination,
    ], dtype=np.float64)
    if not np.isclose(probabilities.sum(), 1.) or np.any(probabilities < 0):
        raise AssertionError("invalid category law")
    return probabilities


def _policies(estimates: np.ndarray) -> np.ndarray:
    delta = estimates[:, 1] - estimates[:, 0]
    return np.where(delta > 0, 1., np.where(delta < 0, 0., .5))


def _method_summary(
    estimates: np.ndarray, baseline: np.ndarray, anchor: np.ndarray,
) -> dict[str, object]:
    policy_one = _policies(estimates)
    deployed = (1 - policy_one) * baseline[0] + policy_one * baseline[1]
    optimum = float(baseline.max())
    correct = np.where(policy_one == .5, .5, policy_one == int(baseline[1] > baseline[0]))
    bias = estimates.mean(axis=0) - anchor
    return {
        "mean_true_regret": float(np.mean(optimum - deployed)),
        "true_correct_rate": float(np.mean(correct)),
        "action_one_rate": float(np.mean(policy_one)),
        "anchor_target_bias_by_action": bias.tolist(),
        "anchor_target_mean_absolute_bias": float(np.mean(np.abs(bias))),
        "anchor_target_rmse": float(np.sqrt(np.mean((estimates - anchor[None, :]) ** 2))),
        "out_of_unit_rate": float(np.mean((estimates < 0) | (estimates > 1))),
    }


def simulate_cell(
    *,
    p: float,
    copying: tuple[float, float],
    contamination: float,
    selection_bias: float,
    config: AnchorRobustnessConfig,
    rng: np.random.Generator,
) -> dict[str, object]:
    config.validate()
    values = analytic_values(p, copying, contamination)
    n = config.observations_per_action
    estimates = {
        name: np.empty((config.repeats, 2), dtype=np.float64)
        for name in (
            "raw_immediate", "anchor_unweighted", "augmented_unweighted",
            "anchor_ipw", "augmented_ipw",
        )
    }
    selected_counts = np.empty((config.repeats, 2), dtype=np.int64)
    e0 = config.base_anchor_rate * (1 - selection_bias)
    e1 = config.base_anchor_rate * (1 + selection_bias)
    if e0 <= 0 or e1 <= 0 or e0 > 1 or e1 > 1:
        raise ValueError("selection propensities must lie in (0,1]")
    outcome = np.asarray([0., 0., 1., 1.])
    anchor = np.asarray([0., 1., 0., 1.])
    propensities = np.asarray([e0, e0, e1, e1])

    for action in (0, 1):
        probabilities = category_probabilities(
            float(values["baseline"][action]), copying[action], contamination,
        )
        counts = rng.multinomial(n, probabilities, size=config.repeats)
        selected = rng.binomial(counts, propensities[None, :])
        selected_total = selected.sum(axis=1)
        if np.any(selected_total == 0):
            raise RuntimeError("zero observed anchors; increase the frozen sample budget")
        observed_o = (counts * outcome[None, :]).sum(axis=1) / n
        selected_b = (selected * anchor[None, :]).sum(axis=1)
        selected_residual = (selected * (anchor - outcome)[None, :]).sum(axis=1)
        weighted_b = (selected * (anchor / propensities)[None, :]).sum(axis=1) / n
        weighted_residual = (
            selected * ((anchor - outcome) / propensities)[None, :]
        ).sum(axis=1) / n

        estimates["raw_immediate"][:, action] = observed_o
        estimates["anchor_unweighted"][:, action] = selected_b / selected_total
        estimates["augmented_unweighted"][:, action] = observed_o + selected_residual / selected_total
        estimates["anchor_ipw"][:, action] = weighted_b
        estimates["augmented_ipw"][:, action] = observed_o + weighted_residual
        selected_counts[:, action] = selected_total

    methods = {
        name: _method_summary(matrix, values["baseline"], values["anchor"])
        for name, matrix in estimates.items()
    }
    return {
        "p": p,
        "copying": list(copying),
        "contamination": contamination,
        "selection_bias": selection_bias,
        "selection_propensity": {"o0": e0, "o1": e1},
        "analytic_values": {name: vector.tolist() for name, vector in values.items()},
        "anchor_target_margin": float(abs(values["anchor"][1] - values["anchor"][0])),
        "anchor_target_best_action": int(values["anchor"][1] > values["anchor"][0]),
        "baseline_best_action": 1,
        "contamination_flip_point": contamination_flip_point(p, copying),
        "mean_selected_anchors_per_action": selected_counts.mean(axis=0).tolist(),
        "methods": methods,
    }


def _mean(cells: list[dict[str, object]], method: str, metric: str) -> float:
    return float(np.mean([cell["methods"][method][metric] for cell in cells]))  # type: ignore[index]


def _max_abs_bias(cells: list[dict[str, object]], method: str) -> float:
    return max(
        abs(value)
        for cell in cells
        for value in cell["methods"][method]["anchor_target_bias_by_action"]  # type: ignore[index]
    )


def run_audit(config: AnchorRobustnessConfig = AnchorRobustnessConfig()) -> dict[str, object]:
    config.validate()
    rng = np.random.default_rng(config.seed)
    rows = []
    for p in config.p_grid:
        for c0 in config.c0_grid:
            for c1 in config.c1_grid:
                for contamination in config.contamination_grid:
                    for selection_bias in config.selection_bias_grid:
                        rows.append(simulate_cell(
                            p=p,
                            copying=(c0, c1),
                            contamination=contamination,
                            selection_bias=selection_bias,
                            config=config,
                            rng=rng,
                        ))

    reversal = [
        row for row in rows
        if row["analytic_values"]["immediate"][0] > row["analytic_values"]["immediate"][1]  # type: ignore[index]
        and abs(2 * row["p"] - 1) >= config.informative_margin  # type: ignore[operator]
    ]
    clean_random = [
        row for row in reversal
        if row["contamination"] == 0 and row["selection_bias"] == 0
    ]
    clean_selected = [row for row in reversal if row["contamination"] == 0]
    high_selection = [
        row for row in reversal
        if row["contamination"] == 0 and row["selection_bias"] == max(config.selection_bias_grid)
    ]
    rank_preserving = [
        row for row in reversal
        if row["anchor_target_best_action"] == 1
        and row["anchor_target_margin"] >= config.informative_margin
    ]
    rank_flipped = [
        row for row in reversal
        if row["anchor_target_best_action"] == 0
        and row["anchor_target_margin"] >= config.informative_margin
    ]
    if not all((reversal, clean_random, high_selection, rank_preserving, rank_flipped)):
        raise AssertionError("frozen grid lacks a required diagnostic subset")

    summaries = {
        "cell_counts": {
            "all": len(rows),
            "immediate_ranking_reversals": len(reversal),
            "clean_random": len(clean_random),
            "clean_selected": len(clean_selected),
            "rank_preserving": len(rank_preserving),
            "rank_flipped": len(rank_flipped),
        },
        "clean_random_mean_true_regret": {
            method: _mean(clean_random, method, "mean_true_regret")
            for method in ("raw_immediate", "anchor_ipw", "augmented_ipw")
        },
        "high_selection_mean_absolute_bias": {
            method: _mean(high_selection, method, "anchor_target_mean_absolute_bias")
            for method in ("anchor_unweighted", "augmented_unweighted", "anchor_ipw", "augmented_ipw")
        },
        "all_selection_max_absolute_bias": {
            method: _max_abs_bias(clean_selected, method)
            for method in ("anchor_unweighted", "augmented_unweighted", "anchor_ipw", "augmented_ipw")
        },
        "rank_preserving_augmented_ipw_true_correct_rate": _mean(
            rank_preserving, "augmented_ipw", "true_correct_rate"
        ),
        "rank_flipped_augmented_ipw_true_correct_rate": _mean(
            rank_flipped, "augmented_ipw", "true_correct_rate"
        ),
    }
    clean_regret = summaries["clean_random_mean_true_regret"]
    high_bias = summaries["high_selection_mean_absolute_bias"]
    max_bias = summaries["all_selection_max_absolute_bias"]
    gates = {
        "clean_augmented_beats_equal_anchor_ipw": (
            clean_regret["augmented_ipw"] <= .90 * clean_regret["anchor_ipw"]
        ),
        "post_report_selection_creates_material_naive_bias": (
            max_bias["augmented_unweighted"] >= .02
        ),
        "known_propensity_controls_augmented_bias": (
            max_bias["augmented_ipw"] <= .012
        ),
        "ipw_improves_high_selection_bias": (
            high_bias["augmented_ipw"] <= .35 * high_bias["augmented_unweighted"]
        ),
        "rank_preserving_region_remains_learnable": (
            summaries["rank_preserving_augmented_ipw_true_correct_rate"] >= .80
        ),
        "contaminated_anchor_rank_flip_is_detected": (
            summaries["rank_flipped_augmented_ipw_true_correct_rate"] <= .20
        ),
    }
    return {
        "decision": "DEV_ROBUSTNESS_LAW_QUALIFIED" if all(gates.values()) else "DEV_ROBUSTNESS_LAW_NOT_QUALIFIED",
        "config": asdict(config),
        "estimand": "initial-preference agreement, with a separately reported contaminated-anchor target",
        "identification": (
            "Known positive P(S=1|A,O) identifies the observed delayed-anchor target under "
            "S independent of B conditional on A,O. It cannot undo contamination of B itself."
        ),
        "gates": gates,
        "aggregate": summaries,
        "rows": rows,
        "scope": (
            "Finite-state developmental assumption audit. Passing characterizes selection correction and "
            "a contamination boundary; it is not neural evidence, human prevalence, or a paper green light."
        ),
    }

