import pytest

from interaction_sprint.hindsight_expression_transition import (
    extended_joint,
    immediate_joint,
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

