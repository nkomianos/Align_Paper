"""Exact non-identification under a PUMA-style transition/emission factorization."""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from typing import Mapping, Sequence


State = tuple[int, int]  # (persistent preference, transient expressed stance)


def _fraction(value: object) -> Fraction:
    if isinstance(value, Fraction):
        return value
    return Fraction(str(value))


def emission(state: State) -> int:
    """Common, action-independent deterministic observation O=stance."""
    return state[1]


def post_action_transition(
    z0: int, action: int, copy_probability: object, mechanism: str,
) -> dict[State, Fraction]:
    """Transition from initial state (z0,z0) after the assistant action."""
    if z0 not in (0, 1) or action not in (0, 1):
        raise ValueError("binary state and action required")
    if mechanism not in {"expression", "transition"}:
        raise ValueError("invalid mechanism")
    copy = _fraction(copy_probability)
    if not 0 <= copy <= 1:
        raise ValueError("copy probability outside [0,1]")
    if action == z0:
        return {(z0, z0): Fraction(1)}
    unchanged = (z0, z0)
    copied = (z0, action) if mechanism == "expression" else (action, action)
    result: defaultdict[State, Fraction] = defaultdict(Fraction)
    result[unchanged] += 1 - copy
    result[copied] += copy
    return dict(result)


def neutral_probe_transition(state: State) -> State:
    """Remove transient expression while retaining persistent preference."""
    persistent, _ = state
    return persistent, persistent


def observed_joint(
    p_z0_one: object,
    propensity_action_one: Sequence[object],
    copy_probability_by_action: Sequence[object],
    *,
    mechanism: str,
    delayed: bool,
) -> dict[tuple[int, ...], Fraction]:
    p = _fraction(p_z0_one)
    propensities = tuple(_fraction(value) for value in propensity_action_one)
    copying = tuple(_fraction(value) for value in copy_probability_by_action)
    if not 0 <= p <= 1 or len(propensities) != 2 or len(copying) != 2:
        raise ValueError("invalid design")
    if any(not 0 <= value <= 1 for value in propensities + copying):
        raise ValueError("probability outside [0,1]")
    result: defaultdict[tuple[int, ...], Fraction] = defaultdict(Fraction)
    for z0 in (0, 1):
        p_z0 = p if z0 else 1 - p
        for action in (0, 1):
            p_action = propensities[z0] if action else 1 - propensities[z0]
            for state, p_state in post_action_transition(
                z0, action, copying[action], mechanism
            ).items():
                immediate = emission(state)
                key: tuple[int, ...] = (z0, action, immediate)
                if delayed:
                    key += (emission(neutral_probe_transition(state)),)
                result[key] += p_z0 * p_action * p_state
    if sum(result.values(), Fraction()) != 1:
        raise AssertionError("joint distribution does not normalize")
    return dict(result)


def total_variation(
    left: Mapping[tuple[int, ...], Fraction],
    right: Mapping[tuple[int, ...], Fraction],
) -> Fraction:
    keys = set(left) | set(right)
    return sum((abs(left.get(key, Fraction()) - right.get(key, Fraction())) for key in keys), Fraction()) / 2


def persistent_policy_values(
    p_z0_one: object, copy_probability_by_action: Sequence[object], *, mechanism: str,
) -> dict[int, Fraction]:
    p = _fraction(p_z0_one)
    copying = tuple(_fraction(value) for value in copy_probability_by_action)
    if mechanism == "expression":
        return {0: 1 - p, 1: p}
    if mechanism == "transition":
        return {0: (1 - p) + p * copying[0], 1: p + (1 - p) * copying[1]}
    raise ValueError("invalid mechanism")


def frozen_result() -> dict[str, object]:
    p = Fraction(3, 5)
    propensity = (Fraction(1, 2), Fraction(1, 2))
    copying = (Fraction(1), Fraction(0))
    immediate_expression = observed_joint(
        p, propensity, copying, mechanism="expression", delayed=False
    )
    immediate_transition = observed_joint(
        p, propensity, copying, mechanism="transition", delayed=False
    )
    delayed_expression = observed_joint(
        p, propensity, copying, mechanism="expression", delayed=True
    )
    delayed_transition = observed_joint(
        p, propensity, copying, mechanism="transition", delayed=True
    )
    expression_values = persistent_policy_values(p, copying, mechanism="expression")
    transition_values = persistent_policy_values(p, copying, mechanism="transition")
    immediate_tv = total_variation(immediate_expression, immediate_transition)
    delayed_tv = total_variation(delayed_expression, delayed_transition)
    return {
        "decision": "PUMA_FACTORIZED_NONIDENTIFICATION_EXACT",
        "state": "(persistent_preference, transient_expressed_stance)",
        "common_action_independent_emission": "O=transient_expressed_stance",
        "neutral_probe": "(P,S)->(P,P)",
        "immediate_log_total_variation_exact": str(immediate_tv),
        "delayed_probe_total_variation_exact": str(delayed_tv),
        "immediate_log_total_variation": float(immediate_tv),
        "delayed_probe_total_variation": float(delayed_tv),
        "persistent_policy_values_exact": {
            "expression": {str(key): str(value) for key, value in expression_values.items()},
            "transition": {str(key): str(value) for key, value in transition_values.items()},
        },
        "best_persistent_action": {
            "expression": max(expression_values, key=expression_values.get),
            "transition": max(transition_values, key=transition_values.get),
        },
        "paper_green_light": False,
    }
