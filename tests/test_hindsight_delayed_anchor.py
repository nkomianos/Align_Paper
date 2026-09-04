import numpy as np
import pytest

from interaction_sprint.hindsight_delayed_anchor import (
    DelayedAnchorConfig,
    _true_values,
    run_audit,
    simulate_cell,
)


def test_true_values_match_expression_and_transition_definitions():
    assert _true_values(.6, (1., 0.), "expression").tolist() == pytest.approx([.4, .6])
    assert _true_values(.6, (1., 0.), "transition").tolist() == pytest.approx([1., .6])


def test_matched_worlds_have_identical_raw_policy_and_truthful_control():
    cell = simulate_cell(
        p=.65, copying=(0., 0.), anchors_per_action=8, repeats=200,
        immediate_per_action=64, rng=np.random.default_rng(7),
    )
    assert cell["coupling"]["raw_policy_mismatch_rate"] == 0
    assert cell["coupling"]["truthful_augmented_raw_mismatch_rate"] == 0
    for action in (0, 1):
        assert cell["true_values"]["expression"][action] == pytest.approx(
            cell["true_values"]["transition"][action]
        )


def test_transition_augmented_uses_immediate_information():
    cell = simulate_cell(
        p=.6, copying=(1., 0.), anchors_per_action=8, repeats=1000,
        immediate_per_action=512, rng=np.random.default_rng(11),
    )
    methods = cell["methods"]["transition"]
    assert methods["augmented"]["mean_regret"] == pytest.approx(
        methods["raw_immediate"]["mean_regret"]
    )
    assert methods["augmented"]["mean_regret"] < methods["anchor_only"]["mean_regret"]


def test_config_rejects_invalid_anchor_budget():
    with pytest.raises(ValueError):
        DelayedAnchorConfig(immediate_per_action=8, anchors_per_action=(9,)).validate()


def test_small_replay_is_deterministic():
    config = DelayedAnchorConfig(
        seed=3, repeats=20, immediate_per_action=16, anchors_per_action=(4,),
        p_grid=(.4, .6), copying_grid=(0., 1.),
    )
    assert run_audit(config) == run_audit(config)

