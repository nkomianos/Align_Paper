"""Deterministic pre-GPU harness for the recency-gated policy-switch gate."""

from .gate import analyze_gate, build_corpus, load_config

__all__ = ["analyze_gate", "build_corpus", "load_config"]
