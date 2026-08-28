"""Pre-registered, deterministic decision rule for semantic-ancestry RAG G0.

The runner is intentionally separated from this module.  It may use any
open-weight inference implementation, but it must produce one auditable row
per completion with the exact schema below.  The decision rule never reads a
model explanation or an LLM judge score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np


Condition = Literal[
    "baseline",
    "self_ancestor",
    "cross_ancestor",
    "style_only",
    "independent_rewrite",
    "mmr",
    "history_aware",
]


class Conditions:
    BASELINE: Condition = "baseline"
    SELF_ANCESTOR: Condition = "self_ancestor"
    CROSS_ANCESTOR: Condition = "cross_ancestor"
    STYLE_ONLY: Condition = "style_only"
    INDEPENDENT_REWRITE: Condition = "independent_rewrite"
    MMR: Condition = "mmr"
    HISTORY_AWARE: Condition = "history_aware"
    ALL: tuple[Condition, ...] = (
        BASELINE,
        SELF_ANCESTOR,
        CROSS_ANCESTOR,
        STYLE_ONLY,
        INDEPENDENT_REWRITE,
        MMR,
        HISTORY_AWARE,
    )


@dataclass(frozen=True)
class ResultRow:
    """One completed generation, scored by deterministic source/entity rules."""

    question_id: str
    model_family: str
    condition: Condition
    sample_id: int
    collapsed: int
    faithful: float

    def __post_init__(self) -> None:
        if not self.question_id or not self.model_family:
            raise ValueError("question_id and model_family are required")
        if self.condition not in Conditions.ALL:
            raise ValueError(f"unknown condition: {self.condition}")
        if self.sample_id < 0:
            raise ValueError("sample_id must be non-negative")
        if self.collapsed not in (0, 1):
            raise ValueError("collapsed must be exactly 0 or 1")
        if not 0.0 <= self.faithful <= 1.0:
            raise ValueError("faithful must be in [0, 1]")


@dataclass(frozen=True)
class Thresholds:
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 20260828
    ancestry_effect_lower_bound: float = 0.10
    specificity_lower_bound: float = 0.08
    history_vs_mmr_upper_bound: float = -0.08
    faithfulness_lower_bound: float = -0.02

    def __post_init__(self) -> None:
        if self.bootstrap_samples < 1_000:
            raise ValueError("at least 1,000 question-bootstrap samples are required")


@dataclass(frozen=True)
class GateReport:
    by_model: Mapping[str, Mapping[str, object]]
    pass_gate: bool
    decision: str
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _cell_means(rows: Sequence[ResultRow], model_family: str, condition: Condition) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    selected = [row for row in rows if row.model_family == model_family and row.condition == condition]
    grouped: dict[str, list[ResultRow]] = {}
    for row in selected:
        grouped.setdefault(row.question_id, []).append(row)
    ordered_ids = tuple(sorted(grouped))
    if not ordered_ids:
        raise ValueError(f"no rows for {model_family}/{condition}")
    collapsed: list[float] = []
    faithful: list[float] = []
    for question_id in ordered_ids:
        cell = grouped[question_id]
        sample_ids = [row.sample_id for row in cell]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError(f"duplicate sample_id in {model_family}/{condition}/{question_id}")
        collapsed.append(float(np.mean([row.collapsed for row in cell])))
        faithful.append(float(np.mean([row.faithful for row in cell])))
    return np.asarray(collapsed), np.asarray(faithful), ordered_ids


def _paired_difference(
    rows: Sequence[ResultRow], model_family: str, left: Condition, right: Condition, field: Literal["collapsed", "faithful"], thresholds: Thresholds
) -> tuple[float, float, float, int]:
    left_collapse, left_faithful, left_ids = _cell_means(rows, model_family, left)
    right_collapse, right_faithful, right_ids = _cell_means(rows, model_family, right)
    if left_ids != right_ids:
        raise ValueError(f"unpaired question sets for {model_family}: {left} vs {right}")
    values = (left_collapse - right_collapse) if field == "collapsed" else (left_faithful - right_faithful)
    if len(values) < 30:
        raise ValueError(f"at least 30 paired questions required for {model_family}/{left}/{right}")
    rng = np.random.default_rng(thresholds.bootstrap_seed + sum(map(ord, model_family + left + right + field)))
    draws = rng.integers(0, len(values), size=(thresholds.bootstrap_samples, len(values)))
    sampled = np.mean(values[draws], axis=1)
    return float(np.mean(values)), float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975)), len(values)


def evaluate_gate(rows: Iterable[ResultRow], thresholds: Thresholds = Thresholds()) -> GateReport:
    """Evaluate the frozen G0 rule using question-level paired bootstrapping."""

    materialized = tuple(rows)
    if not materialized:
        raise ValueError("G0 requires result rows")
    families = tuple(sorted({row.model_family for row in materialized}))
    if len(families) < 2:
        raise ValueError("G0 requires two independent model families")
    by_model: dict[str, Mapping[str, object]] = {}
    failures: list[str] = []
    for family in families:
        ancestry = _paired_difference(materialized, family, Conditions.CROSS_ANCESTOR, Conditions.BASELINE, "collapsed", thresholds)
        specificity = _paired_difference(materialized, family, Conditions.CROSS_ANCESTOR, Conditions.STYLE_ONLY, "collapsed", thresholds)
        mitigation = _paired_difference(materialized, family, Conditions.HISTORY_AWARE, Conditions.MMR, "collapsed", thresholds)
        fidelity = _paired_difference(materialized, family, Conditions.HISTORY_AWARE, Conditions.MMR, "faithful", thresholds)
        passes = {
            "ancestry": ancestry[1] >= thresholds.ancestry_effect_lower_bound,
            "specificity": specificity[1] >= thresholds.specificity_lower_bound,
            "history_beats_mmr": mitigation[2] <= thresholds.history_vs_mmr_upper_bound,
            "fidelity": fidelity[1] >= thresholds.faithfulness_lower_bound,
        }
        by_model[family] = {
            "question_count": ancestry[3],
            "ancestry_cross_minus_baseline": _summary(ancestry),
            "specificity_cross_minus_style": _summary(specificity),
            "mitigation_history_minus_mmr": _summary(mitigation),
            "faithfulness_history_minus_mmr": _summary(fidelity),
            "passes": passes,
        }
        failures.extend(f"{family}:{name}" for name, passed in passes.items() if not passed)
    return GateReport(
        by_model=by_model,
        pass_gate=not failures,
        decision="PROCEED_TO_OFFLINE_REPRODUCTION" if not failures else "KILL_SEMANTIC_ANCESTRY_CANDIDATE",
        failures=tuple(failures),
    )


def _summary(values: tuple[float, float, float, int]) -> dict[str, float | int]:
    estimate, lower, upper, count = values
    return {"estimate": estimate, "lower_95": lower, "upper_95": upper, "question_count": count}
