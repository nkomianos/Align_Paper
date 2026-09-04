import pytest

from interaction_sprint.hindsight_expression_transition import (
    extended_joint,
    immediate_joint,
    observational_minimax_regret,
    policy_values,
    report,
    total_variation,
)


@pytest.mark.parametrize("p", [0, .2, .5, .9, 1])
@pytest.mark.parametrize("propensity", [{0: 0, 1: 1}, {0: .2, 1: .8}, {0: 1, 1: 0}])
@pytest.mark.parametrize("copying", [{0: 0, 1: 0}, {0: .3, 1: .7}, {0: 1, 1: 1}])
def test_immediate_logs_are_exactly_equivalent(p, propensity, copying):
    expression = immediate_joint(p, propensity, copying, mechanism="expression")
    transition = immediate_joint(p, propensity, copying, mechanism="transition")
    assert sum(expression.values()) == pytest.approx(1)
    assert sum(transition.values()) == pytest.approx(1)
    assert total_variation(expression, transition) == 0


def test_delayed_anchor_breaks_equivalence_and_ranking_reverses():
    p = .6
    propensity = {0: .35, 1: .7}
    copying = {0: 1, 1: 0}
    expression = extended_joint(p, propensity, copying, mechanism="expression")
    transition = extended_joint(p, propensity, copying, mechanism="transition")
    assert total_variation(expression, transition) > 0
    assert policy_values(p, copying, mechanism="expression")[1] > policy_values(p, copying, mechanism="expression")[0]
    assert policy_values(p, copying, mechanism="transition")[0] > policy_values(p, copying, mechanism="transition")[1]
    result = report()
    assert result["best_action"] == {"expression": 1, "transition": 0, "immediate_agreement_learner": 0}


def test_invalid_parameters_fail():
    with pytest.raises(ValueError):
        immediate_joint(1.1, {0: .5, 1: .5}, {0: .5, 1: .5}, mechanism="expression")
    with pytest.raises(ValueError):
        policy_values(.5, {0: -.1, 1: .5}, mechanism="transition")


def test_observational_minimax_regret_is_positive_despite_unlimited_logs():
    bound = observational_minimax_regret(.6, {0: 1., 1: 0.})
    assert bound == pytest.approx({
        "expression_optimal_action": 1,
        "transition_optimal_action": 0,
        "expression_gap": .2,
        "transition_gap": .4,
        "minimax_probability_action_one": 1 / 3,
        "randomized_minimax_regret_lower_bound": 2 / 15,
        "deterministic_minimax_regret_lower_bound": .2,
    })


@pytest.mark.parametrize("p,copying", [
    (.55, {0: 1., 1: 0.}),
    (.7, {0: .9, 1: 0.}),
    (.3, {0: 0., 1: .9}),
])
def test_minimax_mixture_equalizes_the_two_mechanism_regrets(p, copying):
    bound = observational_minimax_regret(p, copying)
    q = bound["minimax_probability_action_one"]
    expression = policy_values(p, copying, mechanism="expression")
    transition = policy_values(p, copying, mechanism="transition")
    expression_regret = max(expression.values()) - ((1 - q) * expression[0] + q * expression[1])
    transition_regret = max(transition.values()) - ((1 - q) * transition[0] + q * transition[1])
    assert expression_regret == pytest.approx(bound["randomized_minimax_regret_lower_bound"])
    assert transition_regret == pytest.approx(bound["randomized_minimax_regret_lower_bound"])


def test_minimax_bound_requires_a_strict_ranking_reversal():
    with pytest.raises(ValueError, match="ranking reversal"):
        observational_minimax_regret(.5, {0: 0., 1: 0.})
