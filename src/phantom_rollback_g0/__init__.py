"""Phantom Rollback G0: rollback-scope ambiguity and agent risk taking."""

from .analysis import GateThresholds, evaluate_gate, score_family
from .corpus import ARM_IDS, DOMAIN_IDS, build_corpus, validate_corpus

__all__ = [
    "ARM_IDS",
    "DOMAIN_IDS",
    "GateThresholds",
    "build_corpus",
    "evaluate_gate",
    "score_family",
    "validate_corpus",
]
