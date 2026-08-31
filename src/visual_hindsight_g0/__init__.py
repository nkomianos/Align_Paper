"""Causal-cut diagnostic for visual hindsight leakage in video VLMs."""

from .analysis import HindsightThresholds, evaluate_gate, normalize_side, score_family
from .corpus import LOCATIONS, HindsightCase, build_corpus, load_cases, validate_corpus

__all__ = [
    "HindsightCase",
    "HindsightThresholds",
    "LOCATIONS",
    "build_corpus",
    "evaluate_gate",
    "load_cases",
    "normalize_side",
    "score_family",
    "validate_corpus",
]
