import numpy as np
import pytest
from pathlib import Path

from interaction_sprint.feedback_gradient_audit import gradients, setup, frozen_reverse_kl, trajectory
from interaction_sprint.theory import bayes_sdpo_gradient


@pytest.mark.parametrize("p", [.1, .3, .49, .5, .51, .7, .9])
def test_sampled_matches_previous_theory_full_has_opposite_direction(p):
    out = gradients(p, [[.9, .1], [.1, .9]])
    assert out["sampled_response_ascent"] == pytest.approx(bayes_sdpo_gradient(p, [.1, .9]))
    assert out["full_reverse_kl_ascent"] == pytest.approx(p*(1-p)*.8*(2*p-1)*np.log(9))
    if p != .5:
        assert out["sampled_response_ascent"] * out["full_reverse_kl_ascent"] < 0


def test_independent_feedback_restores_equivalence_even_for_non_bayes_teacher():
    out = gradients(.3, [[.4, .6], [.4, .6]], teacher=[[.8, .2], [.3, .7]])
    assert out["sampling_gap"] == pytest.approx(0)
    assert abs(out["full_reverse_kl_ascent"]) > .01


def test_general_channels_oracle_weight_and_frozen_target_finite_difference():
    rng = np.random.default_rng(90461)
    for _ in range(30):
        p = rng.uniform(.05, .95)
        k = rng.dirichlet([2, 2], size=2)
        q = rng.dirichlet([2, 2], size=2)
        out = gradients(p, k, q)
        assert out["known_channel_weighted_ascent"] == pytest.approx(out["full_reverse_kl_ascent"])
        _, _, _, m, _ = setup(p, k, q)
        theta, eps = np.log(p/(1-p)), 1e-5
        fd = -(frozen_reverse_kl(theta+eps, m, q)-frozen_reverse_kl(theta-eps, m, q))/(2*eps)
        assert fd == pytest.approx(out["full_reverse_kl_ascent"], abs=1e-9)


def test_repeated_updates_balance_versus_amplify():
    k = [[.9, .1], [.1, .9]]
    sampled = trajectory(.1, k, "sampled_response_ascent")[-1]["action_one_probability"]
    full = trajectory(.1, k, "full_reverse_kl_ascent")[-1]["action_one_probability"]
    assert sampled > .49
    assert full < .01


@pytest.mark.parametrize("p,k", [(0, [[.9, .1], [.1, .9]]), (.5, [[1, 0], [.1, .9]]), (.5, [[.9, .9], [.1, .9]])])
def test_support_and_normalization_required(p, k):
    with pytest.raises(ValueError):
        gradients(p, k)


@pytest.mark.skipif(not Path("artifacts/user_interactions_objective_audit_v1/.git").exists(), reason="pinned upstream checkout not present")
def test_released_loss_functions_execute_and_match_analytic_update():
    from interaction_sprint.feedback_gradient_audit import released_loss_audit
    out = released_loss_audit(Path("artifacts/user_interactions_objective_audit_v1"), .1, [[.9, .1], [.1, .9]])
    assert out["ascent"]["_simple_signal_loss"] == pytest.approx(.10917701923450238)
    assert out["ascent"]["_full_distillation_loss"] == pytest.approx(-.12656013565456625)
