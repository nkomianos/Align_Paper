"""Finite-state expression/transition identification with delayed probes."""
from __future__ import annotations

import numpy as np


def _probability_vector(values: object, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.ndim != 1 or array.size < 2 or not np.isfinite(array).all()
        or np.any(array < 0) or not np.isclose(array.sum(), 1., atol=1e-12)
    ):
        raise ValueError(f"{name} must be a finite probability vector")
    return array


def _row_stochastic(values: object, name: str, rows: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1
        or (rows is not None and array.shape[0] != rows)
        or not np.isfinite(array).all() or np.any(array < 0)
        or not np.allclose(array.sum(axis=1), 1., atol=1e-12)
    ):
        raise ValueError(f"{name} must be a finite row-stochastic matrix")
    return array


def validate_design(
    initial: object, logging: object, action_channels: object, anchor_emission: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    initial_array = _probability_vector(initial, "initial")
    states = initial_array.size
    logging_array = _row_stochastic(logging, "logging", rows=states)
    channels = np.asarray(action_channels, dtype=np.float64)
    if channels.ndim != 3 or channels.shape[1:] != (states, states):
        raise ValueError("action_channels must have shape [actions, states, states]")
    if channels.shape[0] != logging_array.shape[1]:
        raise ValueError("logging/action counts differ")
    channels = np.stack([
        _row_stochastic(channel, f"action_channels[{index}]", rows=states)
        for index, channel in enumerate(channels)
    ])
    emission = _row_stochastic(anchor_emission, "anchor_emission", rows=states)
    return initial_array, logging_array, channels, emission


def immediate_joint(
    initial: object, logging: object, action_channels: object,
) -> np.ndarray:
    """Return P(Z0,A,O); the same channel can mean expression or transition."""
    initial_array = _probability_vector(initial, "initial")
    states = initial_array.size
    logging_array = _row_stochastic(logging, "logging", rows=states)
    channels = np.asarray(action_channels, dtype=np.float64)
    if channels.ndim != 3 or channels.shape != (
        logging_array.shape[1], states, states,
    ):
        raise ValueError("invalid action channel shape")
    channels = np.stack([
        _row_stochastic(channel, f"action_channels[{index}]", rows=states)
        for index, channel in enumerate(channels)
    ])
    return initial_array[:, None, None] * logging_array[:, :, None] * channels.transpose(1, 0, 2)


def extended_joint(
    initial: object,
    logging: object,
    action_channels: object,
    anchor_emission: object,
    *,
    mechanism: str,
) -> np.ndarray:
    """Return P(Z0,A,O,B) for expression or truthful-transition semantics."""
    initial_array, logging_array, channels, emission = validate_design(
        initial, logging, action_channels, anchor_emission,
    )
    states = initial_array.size
    observations = emission.shape[1]
    joint = np.zeros((states, channels.shape[0], states, observations), dtype=np.float64)
    for z0 in range(states):
        for action in range(channels.shape[0]):
            for immediate in range(states):
                base = initial_array[z0] * logging_array[z0, action] * channels[action, z0, immediate]
                if mechanism == "expression":
                    joint[z0, action, immediate, :] = base * emission[z0, :]
                elif mechanism == "transition":
                    joint[z0, action, immediate, :] = base * emission[immediate, :]
                else:
                    raise ValueError("unknown mechanism")
    return joint


def total_variation(first: object, second: object) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("support mismatch")
    return float(.5 * np.abs(left - right).sum())


def delayed_anchor_conditionals(
    action_channels: object, anchor_emission: object, *, mechanism: str,
) -> np.ndarray:
    """Return Q[a,z0,b]=P(B=b|Z0=z0,A=a)."""
    channels = np.asarray(action_channels, dtype=np.float64)
    if channels.ndim != 3 or channels.shape[1] != channels.shape[2]:
        raise ValueError("action_channels must be square state channels")
    states = channels.shape[1]
    channels = np.stack([
        _row_stochastic(channel, f"action_channels[{index}]", rows=states)
        for index, channel in enumerate(channels)
    ])
    emission = _row_stochastic(anchor_emission, "anchor_emission", rows=states)
    if mechanism == "expression":
        return np.repeat(emission[None, :, :], channels.shape[0], axis=0)
    if mechanism == "transition":
        return np.stack([channel @ emission for channel in channels])
    raise ValueError("unknown mechanism")


def recover_transition_channels(conditionals: object, anchor_emission: object) -> np.ndarray:
    """Recover T from Q=T M when the delayed-probe emission has full row rank."""
    emission = np.asarray(anchor_emission, dtype=np.float64)
    if emission.ndim != 2:
        raise ValueError("anchor_emission must be a matrix")
    emission = _row_stochastic(emission, "anchor_emission")
    states = emission.shape[0]
    if np.linalg.matrix_rank(emission, tol=1e-12) != states:
        raise ValueError("anchor emission is not full row rank")
    q = np.asarray(conditionals, dtype=np.float64)
    if q.ndim != 3 or q.shape[1:] != (states, emission.shape[1]):
        raise ValueError("conditional shape mismatch")
    return q @ np.linalg.pinv(emission, rcond=1e-12)


def recovery_stability_bound(
    conditional_error_frobenius: float, anchor_emission: object,
) -> float:
    """Bound ||T_hat-T||_F by ||M^+||_2 ||Q_hat-Q||_F."""
    error = float(conditional_error_frobenius)
    if not np.isfinite(error) or error < 0:
        raise ValueError("conditional error must be finite and nonnegative")
    emission = _row_stochastic(anchor_emission, "anchor_emission")
    if np.linalg.matrix_rank(emission, tol=1e-12) != emission.shape[0]:
        raise ValueError("anchor emission is not full row rank")
    amplification = float(np.linalg.norm(np.linalg.pinv(emission, rcond=1e-12), ord=2))
    return amplification * error


def example_report() -> dict[str, object]:
    initial = np.asarray([.2, .5, .3])
    logging = np.asarray([[.6, .4], [.5, .5], [.4, .6]])
    channels = np.asarray([
        [[.8, .1, .1], [.2, .7, .1], [.1, .2, .7]],
        [[.6, .3, .1], [.1, .6, .3], [.2, .2, .6]],
    ])
    emission = np.asarray([
        [.85, .10, .05],
        [.05, .90, .05],
        [.10, .10, .80],
    ])
    immediate_expression = immediate_joint(initial, logging, channels)
    immediate_transition = immediate_joint(initial, logging, channels)
    extended_expression = extended_joint(
        initial, logging, channels, emission, mechanism="expression",
    )
    extended_transition = extended_joint(
        initial, logging, channels, emission, mechanism="transition",
    )
    transition_q = delayed_anchor_conditionals(channels, emission, mechanism="transition")
    expression_q = delayed_anchor_conditionals(channels, emission, mechanism="expression")
    recovered_transition = recover_transition_channels(transition_q, emission)
    recovered_expression = recover_transition_channels(expression_q, emission)

    rank_deficient_emission = np.asarray([[1., 0.], [0., 1.], [0., 1.]])
    identity = np.eye(3)[None, :, :]
    swap = np.asarray([[[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]]])
    deficient_identity_q = delayed_anchor_conditionals(
        identity, rank_deficient_emission, mechanism="transition",
    )
    deficient_swap_q = delayed_anchor_conditionals(
        swap, rank_deficient_emission, mechanism="transition",
    )
    return {
        "decision": "FINITE_STATE_DELAYED_PROBE_IDENTIFICATION_VERIFIED",
        "states": 3,
        "actions": 2,
        "immediate_log_total_variation": total_variation(
            immediate_expression, immediate_transition,
        ),
        "extended_log_total_variation": total_variation(
            extended_expression, extended_transition,
        ),
        "anchor_emission_rank": int(np.linalg.matrix_rank(emission)),
        "anchor_emission_condition_number": float(np.linalg.cond(emission)),
        "max_transition_recovery_error": float(np.max(np.abs(recovered_transition - channels))),
        "max_expression_identity_recovery_error": float(
            np.max(np.abs(recovered_expression - np.repeat(np.eye(3)[None, :, :], 2, axis=0)))
        ),
        "rank_deficient_witness": {
            "rank": int(np.linalg.matrix_rank(rank_deficient_emission)),
            "transition_distance": float(np.linalg.norm(identity - swap)),
            "conditional_distance": float(np.linalg.norm(deficient_identity_q - deficient_swap_q)),
        },
        "scope": (
            "Exact finite-state extension and linear inverse stability result. The rank criterion is "
            "standard latent-state identification machinery applied to next-turn expression versus "
            "persistent transition; it is not standalone paper novelty or empirical evidence."
        ),
    }
