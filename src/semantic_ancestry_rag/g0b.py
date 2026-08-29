"""Fail-closed role contract and input materialization for semantic-ancestry G0b.

This is deliberately separate from the sealed G0 runner.  It encodes the
causal-role requirements before any new corpus is materialized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .corpus import BaseQuestion
from .retrieval import history_aware_select, mmr_select
from .runner import Question


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


def materialize_question(
    base: BaseQuestion,
    *,
    ancestor_answer: str,
    shadow_answer: str,
    cross_rewrite: str,
    style_rewrite: str,
    independent_summary: str,
) -> Question:
    """Build one role-matched G0b question without exposing role metadata.

    ``cross_rewrite`` and ``style_rewrite`` must be produced by the *same*
    external rewriter under the same rewrite prompt.  The caller records that
    provenance in an immutable role manifest; this function deliberately keeps
    model labels out of serving prompts.
    """

    if not all((ancestor_answer, shadow_answer, cross_rewrite, style_rewrite, independent_summary)):
        raise ValueError("G0b requires non-empty frozen transformations")
    pool = (*base.base_references, cross_rewrite)
    mmr = mmr_select(base.question, pool, limit=len(base.base_references))
    history = history_aware_select(base.question, pool, (ancestor_answer,), limit=len(base.base_references))
    references: Mapping[str, Sequence[str]] = {
        "baseline": base.base_references,
        "self_ancestor": (*base.base_references, ancestor_answer),
        "cross_ancestor": (*base.base_references, cross_rewrite),
        "style_only": (*base.base_references, style_rewrite),
        "independent_summary": (*base.base_references, independent_summary),
        "mmr": tuple(pool[index] for index in mmr.indices),
        "history_aware": tuple(pool[index] for index in history.indices),
    }
    support = {
        condition: tuple(
            entity for entity, aliases in base.entity_aliases.items()
            if any(alias.lower() in "\n".join(passages).lower() for alias in aliases)
        )
        for condition, passages in references.items()
    }
    return Question(
        question_id=base.question_id,
        question=base.question,
        references=references,
        entity_aliases=base.entity_aliases,
        source_supported_entities=support,
    )
