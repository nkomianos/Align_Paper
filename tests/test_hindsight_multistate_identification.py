import numpy as np
import pytest

from interaction_sprint.hindsight_multistate_identification import (
    delayed_anchor_conditionals,
    example_report,
    extended_joint,
    immediate_joint,
    recover_transition_channels,
    recovery_stability_bound,
    total_variation,
)


def _design():
    initial = np.array([.2, .5, .3])
    logging = np.array([[.6, .4], [.5, .5], [.4, .6]])
    channels = np.array([
        [[.8, .1, .1], [.2, .7, .1], [.1, .2, .7]],
        [[.6, .3, .1], [.1, .6, .3], [.2, .2, .6]],
    ])
    emission = np.array([[.85, .10, .05], [.05, .90, .05], [.10, .10, .80]])
    return initial, logging, channels, emission


def test_immediate_expression_and_transition_are_exactly_equivalent():
    initial, logging, channels, _ = _design()
    expression = immediate_joint(initial, logging, channels)
    transition = immediate_joint(initial, logging, channels)
    assert expression.shape == (3, 2, 3)
    assert expression.sum() == pytest.approx(1.)
    assert total_variation(expression, transition) == 0.


def test_delayed_probe_separates_and_exactly_recovers_full_rank_transition():
    initial, logging, channels, emission = _design()
    expression = extended_joint(initial, logging, channels, emission, mechanism="expression")
    transition = extended_joint(initial, logging, channels, emission, mechanism="transition")
    assert expression.sum() == pytest.approx(1.)
    assert transition.sum() == pytest.approx(1.)
    assert total_variation(expression, transition) > .05
    q = delayed_anchor_conditionals(channels, emission, mechanism="transition")
    assert np.allclose(recover_transition_channels(q, emission), channels, atol=1e-12)


def test_expression_conditionals_recover_identity_transition():
    _, _, channels, emission = _design()
    q = delayed_anchor_conditionals(channels, emission, mechanism="expression")
    recovered = recover_transition_channels(q, emission)
    assert np.allclose(recovered, np.repeat(np.eye(3)[None, :, :], 2, axis=0), atol=1e-12)


def test_recovery_error_obeys_pseudoinverse_stability_bound():
    _, _, channels, emission = _design()
    q = delayed_anchor_conditionals(channels, emission, mechanism="transition")
    perturbation = np.zeros_like(q)
    perturbation[:, :, 0] = .001
    perturbation[:, :, 1] = -.001
    recovered = recover_transition_channels(q + perturbation, emission)
    error = float(np.linalg.norm(recovered - channels))
    bound = recovery_stability_bound(float(np.linalg.norm(perturbation)), emission)
    assert error <= bound + 1e-12


def test_rank_deficient_probe_has_explicit_transition_alias_and_fails_recovery():
    emission = np.array([[1., 0.], [0., 1.], [0., 1.]])
    identity = np.eye(3)[None, :, :]
    swap = np.array([[[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]]])
    first = delayed_anchor_conditionals(identity, emission, mechanism="transition")
    second = delayed_anchor_conditionals(swap, emission, mechanism="transition")
    assert not np.array_equal(identity, swap)
    assert np.array_equal(first, second)
    with pytest.raises(ValueError, match="full row rank"):
        recover_transition_channels(first, emission)
    report = example_report()
    assert report["rank_deficient_witness"]["conditional_distance"] == 0.
