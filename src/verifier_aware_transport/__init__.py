"""Exact-marginal transport experiments for verifier-selected test-time scaling."""

from .gate import (
    Scenario,
    TransportGateReport,
    decode_categorical,
    lattice_uniforms,
    run_feasibility_gate,
)

__all__ = [
    "Scenario",
    "TransportGateReport",
    "decode_categorical",
    "lattice_uniforms",
    "run_feasibility_gate",
]
