"""Pre-registered gate for cross-recipe causal-mechanism transfer."""

from .gate import analyze_gate, build_corpus, load_config
from .external_registry import compile_core_folds, freeze_fold_plan

__all__ = ("analyze_gate", "build_corpus", "compile_core_folds", "freeze_fold_plan", "load_config")
