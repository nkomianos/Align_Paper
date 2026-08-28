"""Fail-closed role contract for a future semantic-ancestry G0b.

This is deliberately separate from the sealed G0 runner.  It encodes the
causal-role requirements before any new corpus is materialized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class G0BCell:
    """One serving-family × external-pair corpus construction cell."""

    serving_model: str
    ancestor_model: str
    rewriter_model: str
    style_rewriter_model: str
    summary_rewriter_model: str
    shadow_answer_model: str

    def __post_init__(self) -> None:
        if not all(vars(self).values()):
            raise ValueError("all G0b model roles must be named")
        if self.ancestor_model != self.serving_model:
            raise ValueError("G0b ancestor must be the evaluated serving model")
        if self.rewriter_model != self.style_rewriter_model:
            raise ValueError("cross and style-only passages require the same rewriter")
        if self.rewriter_model != self.summary_rewriter_model:
            raise ValueError("independent summary requires the same rewriter")
        if self.rewriter_model == self.serving_model:
            raise ValueError("G0b rewriter must be external to the serving family")
        if self.shadow_answer_model in {self.serving_model, self.rewriter_model}:
            raise ValueError("G0b shadow-answer model must be independent of serving and rewriter roles")


def validate_role_plan(cells: Iterable[G0BCell], *, serving_models: Iterable[str], external_pairs: Iterable[tuple[str, str]]) -> tuple[G0BCell, ...]:
    """Require every serving model to be crossed with every external pair."""

    materialized = tuple(cells)
    serving = tuple(sorted(set(serving_models)))
    pairs = tuple(sorted(set(external_pairs)))
    if len(serving) < 2 or len(pairs) < 2:
        raise ValueError("G0b requires at least two serving models and two external pairs")
    expected = {(target, rewriter, shadow) for target in serving for rewriter, shadow in pairs}
    observed = {(cell.serving_model, cell.rewriter_model, cell.shadow_answer_model) for cell in materialized}
    if observed != expected or len(materialized) != len(expected):
        raise ValueError("G0b must fully cross every serving model with every external pair exactly once")
    external = {value for pair in pairs for value in pair}
    if external.intersection(serving):
        raise ValueError("external G0b pairs may not contain a serving model")
    return materialized
