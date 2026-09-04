"""Exact finite model separating user expression from preference transition."""
from __future__ import annotations

from itertools import product
from typing import Mapping


def _check_probability(value: float, name: str) -> float:
    value = float(value)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must lie in [0, 1]")
    return value


def immediate_joint(
    p: float,
    action_one_probability: Mapping[int, float],
    copying: Mapping[int, float],
    *,
    mechanism: str,
) -> dict[tuple[int, int, int], float]:
    """P(Z0,A,O) for static-expression or truthful-transition mechanisms.

    The logging policy may depend on Z0.  In the expression mechanism Z1=Z0
    and the report copies a discordant action with probability c[A].  In the
    transition mechanism Z1 copies the action with that probability and O=Z1.
    """
    p = _check_probability(p, "p")
    if mechanism not in {"expression", "transition"}:
        raise ValueError("unknown mechanism")
    propensity = {z: _check_probability(action_one_probability[z], f"propensity[{z}]") for z in (0, 1)}
    rates = {a: _check_probability(copying[a], f"copying[{a}]") for a in (0, 1)}
    joint: dict[tuple[int, int, int], float] = {}
    for z, a, o in product((0, 1), repeat=3):
        z_prob = p if z else 1 - p
        a_prob = propensity[z] if a else 1 - propensity[z]
        if a == z:
            o_prob = 1.0 if o == z else 0.0
        else:
            # Both mechanisms induce the same immediate report distribution.
            o_prob = rates[a] if o == a else 1 - rates[a]
        joint[(z, a, o)] = z_prob * a_prob * o_prob
    return joint


def extended_joint(
    p: float,
    action_one_probability: Mapping[int, float],
    copying: Mapping[int, float],
    *,
    mechanism: str,
) -> dict[tuple[int, int, int, int], float]:
    """P(Z0,A,O,B) when delayed neutral anchor B truthfully measures Z1."""
    p = _check_probability(p, "p")
    if mechanism not in {"expression", "transition"}:
        raise ValueError("unknown mechanism")
    propensity = {z: _check_probability(action_one_probability[z], f"propensity[{z}]") for z in (0, 1)}
    rates = {a: _check_probability(copying[a], f"copying[{a}]") for a in (0, 1)}
    joint: dict[tuple[int, int, int, int], float] = {}
    for z, a, o, anchor in product((0, 1), repeat=4):
        z_prob = p if z else 1 - p
        a_prob = propensity[z] if a else 1 - propensity[z]
        if a == z:
            branches = ((1.0, z, z),)
        elif mechanism == "expression":
            branches = ((rates[a], a, z), (1 - rates[a], z, z))
        else:
            branches = ((rates[a], a, a), (1 - rates[a], z, z))
        probability = 0.0
        for branch_probability, branch_o, branch_z1 in branches:
            if o == branch_o and anchor == branch_z1:
                probability += branch_probability
        joint[(z, a, o, anchor)] = z_prob * a_prob * probability
    return joint


def total_variation(first: Mapping[tuple, float], second: Mapping[tuple, float]) -> float:
    if set(first) != set(second):
        raise ValueError("support mismatch")
    return 0.5 * sum(abs(float(first[key]) - float(second[key])) for key in first)


def policy_values(p: float, copying: Mapping[int, float], *, mechanism: str) -> dict[int, float]:
    """Utility of a constant action under contemporaneous latent preference."""
    p = _check_probability(p, "p")
    rates = {a: _check_probability(copying[a], f"copying[{a}]") for a in (0, 1)}
    if mechanism == "expression":
        return {0: 1 - p, 1: p}
    if mechanism == "transition":
        return {0: (1 - p) + p * rates[0], 1: p + (1 - p) * rates[1]}
    raise ValueError("unknown mechanism")


def immediate_agreement_values(p: float, copying: Mapping[int, float]) -> dict[int, float]:
    """Probability the immediate report agrees with a constant action."""
    return policy_values(p, copying, mechanism="transition")


def report() -> dict[str, object]:
    p = 0.6
    propensity = {0: 0.35, 1: 0.7}  # deliberately state-dependent, with overlap
    copying = {0: 1.0, 1: 0.0}
    expression = immediate_joint(p, propensity, copying, mechanism="expression")
    transition = immediate_joint(p, propensity, copying, mechanism="transition")
    expression_extended = extended_joint(p, propensity, copying, mechanism="expression")
    transition_extended = extended_joint(p, propensity, copying, mechanism="transition")
    static_values = policy_values(p, copying, mechanism="expression")
    dynamic_values = policy_values(p, copying, mechanism="transition")
    agreement = immediate_agreement_values(p, copying)
    return {
        "status": "EXACT_OBSERVATIONAL_EQUIVALENCE_WITH_POLICY_RANKING_REVERSAL",
        "parameters": {"p_z0_one": p, "logging_action_one_probability": propensity,
                       "copying_by_action": copying},
        "immediate_log_total_variation": total_variation(expression, transition),
        "delayed_anchor_total_variation": total_variation(expression_extended, transition_extended),
        "contemporaneous_latent_preference_value": {
            "expression": static_values,
            "transition": dynamic_values,
        },
        "best_action": {
            "expression": max(static_values, key=static_values.get),
            "transition": max(dynamic_values, key=dynamic_values.get),
            "immediate_agreement_learner": max(agreement, key=agreement.get),
        },
        "immediate_agreement_value": agreement,
        "scope": (
            "Exact finite counterexample. It proves non-identification within the stated binary model; "
            "it is not an empirical claim about users, SDPO training, or a unique welfare objective."
        ),
    }

