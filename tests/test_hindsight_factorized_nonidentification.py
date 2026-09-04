from fractions import Fraction

from interaction_sprint.hindsight_factorized_nonidentification import (
    emission,
    frozen_result,
    neutral_probe_transition,
    observed_joint,
    post_action_transition,
    total_variation,
)


def test_emission_is_common_and_action_independent() -> None:
    assert emission((0, 1)) == 1
    assert emission((1, 0)) == 0
    assert neutral_probe_transition((0, 1)) == (0, 0)
    assert neutral_probe_transition((1, 0)) == (1, 1)


def test_transient_and_persistent_copy_have_same_immediate_observation() -> None:
    expression = post_action_transition(1, 0, Fraction(2, 3), "expression")
    transition = post_action_transition(1, 0, Fraction(2, 3), "transition")
    assert expression == {(1, 1): Fraction(1, 3), (1, 0): Fraction(2, 3)}
    assert transition == {(1, 1): Fraction(1, 3), (0, 0): Fraction(2, 3)}
    assert {
        emission(state): probability for state, probability in expression.items()
    } == {
        emission(state): probability for state, probability in transition.items()
    }


def test_observational_equivalence_with_arbitrary_logging() -> None:
    for p in (Fraction(0), Fraction(1, 5), Fraction(1, 2), Fraction(4, 5), Fraction(1)):
        for propensity in ((Fraction(0), Fraction(1)), (Fraction(1, 3), Fraction(2, 3))):
            for copying in ((Fraction(0), Fraction(0)), (Fraction(2, 5), Fraction(3, 4)), (Fraction(1), Fraction(1))):
                expression = observed_joint(
                    p, propensity, copying, mechanism="expression", delayed=False
                )
                transition = observed_joint(
                    p, propensity, copying, mechanism="transition", delayed=False
                )
                assert expression == transition
                assert total_variation(expression, transition) == 0


def test_delayed_neutral_probe_separates_nontrivial_mechanisms() -> None:
    expression = observed_joint(
        Fraction(3, 5), (Fraction(1, 2),) * 2, (Fraction(1), Fraction(0)),
        mechanism="expression", delayed=True,
    )
    transition = observed_joint(
        Fraction(3, 5), (Fraction(1, 2),) * 2, (Fraction(1), Fraction(0)),
        mechanism="transition", delayed=True,
    )
    assert total_variation(expression, transition) > 0


def test_frozen_result_reverses_persistent_policy_ranking() -> None:
    result = frozen_result()
    assert result["immediate_log_total_variation_exact"] == "0"
    assert result["delayed_probe_total_variation"] > 0
    assert result["best_persistent_action"] == {"expression": 1, "transition": 0}
    assert result["paper_green_light"] is False
