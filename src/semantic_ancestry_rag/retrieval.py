"""Provenance-free, answer-history-aware retrieval selection for G0.

The method uses only text available in a deployed system: a query, candidate
passages, and that system's prior answer bank.  It neither predicts whether
text is AI-written nor relies on a source-provenance field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class Selection:
    indices: tuple[int, ...]
    relevance: tuple[float, ...]
    ancestry_similarity: tuple[float, ...]


def _similarities(query: str, passages: Sequence[str], history: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not query or not passages:
        raise ValueError("query and at least one candidate passage are required")
    corpus = [query, *passages, *history]
    matrix = TfidfVectorizer(ngram_range=(1, 2), norm="l2").fit_transform(corpus)
    query_vector = matrix[0]
    passage_vectors = matrix[1 : 1 + len(passages)]
    relevance = (passage_vectors @ query_vector.T).toarray().ravel()
    if not history:
        return relevance, np.zeros(len(passages)), (passage_vectors @ passage_vectors.T).toarray()
    history_vectors = matrix[1 + len(passages) :]
    ancestry = (passage_vectors @ history_vectors.T).toarray().max(axis=1)
    return relevance, ancestry, (passage_vectors @ passage_vectors.T).toarray()


def mmr_select(query: str, passages: Sequence[str], *, limit: int, diversity_weight: float = 0.25) -> Selection:
    """Strong generic retrieval-diversity baseline without answer history."""

    return _select(query, passages, (), limit=limit, ancestry_weight=0.0, diversity_weight=diversity_weight)


def history_aware_select(
    query: str, passages: Sequence[str], prior_answers: Sequence[str], *, limit: int,
    ancestry_weight: float = 0.20, diversity_weight: float = 0.25,
) -> Selection:
    """Select relevant passages that are not semantic descendants of prior answers."""

    if not prior_answers:
        raise ValueError("history-aware selection requires at least one prior answer")
    return _select(query, passages, prior_answers, limit=limit, ancestry_weight=ancestry_weight, diversity_weight=diversity_weight)


def _select(query: str, passages: Sequence[str], history: Sequence[str], *, limit: int, ancestry_weight: float, diversity_weight: float) -> Selection:
    if not 1 <= limit <= len(passages):
        raise ValueError("limit must be in [1, number of passages]")
    if ancestry_weight < 0.0 or diversity_weight < 0.0 or ancestry_weight + diversity_weight >= 1.0:
        raise ValueError("weights must be non-negative and sum to less than one")
    relevance, ancestry, pairwise = _similarities(query, passages, history)
    selected: list[int] = []
    remaining = set(range(len(passages)))
    while len(selected) < limit:
        def score(index: int) -> tuple[float, float]:
            redundancy = max((float(pairwise[index, previous]) for previous in selected), default=0.0)
            value = float(relevance[index] - ancestry_weight * ancestry[index] - diversity_weight * redundancy)
            return value, -float(index)  # deterministic tie break
        choice = max(remaining, key=score)
        selected.append(choice)
        remaining.remove(choice)
    return Selection(
        indices=tuple(selected),
        relevance=tuple(float(relevance[index]) for index in selected),
        ancestry_similarity=tuple(float(ancestry[index]) for index in selected),
    )
