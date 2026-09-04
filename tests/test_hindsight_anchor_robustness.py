import numpy as np
import pytest

from interaction_sprint.hindsight_anchor_robustness import (
    AnchorRobustnessConfig,
    analytic_values,
    category_probabilities,
    contamination_flip_point,
)


def test_category_law_has_correct_anchor_and_immediate_marginals():
    t, copying, contamination = .25, .8, .3
    probabilities = category_probabilities(t, copying, contamination)
    assert probabilities.sum() == pytest.approx(1.)
    assert probabilities[2:].sum() == pytest.approx(t + (1 - t) * copying)
    assert probabilities[[1, 3]].sum() == pytest.approx(
        (1 - contamination) * t + contamination * (t + (1 - t) * copying)
    )


def test_contamination_endpoints_and_flip_point():
    clean = analytic_values(.75, (1., 0.), 0.)
    contaminated = analytic_values(.75, (1., 0.), 1.)
    assert np.array_equal(clean["anchor"], clean["baseline"])
    assert np.array_equal(contaminated["anchor"], contaminated["immediate"])
    assert contamination_flip_point(.75, (1., 0.)) == pytest.approx(2 / 3)


def test_no_flip_when_immediate_and_baseline_rankings_agree():
    assert contamination_flip_point(.75, (0., 0.)) is None


def test_invalid_configuration_and_probabilities_are_rejected():
    with pytest.raises(ValueError):
        AnchorRobustnessConfig(base_anchor_rate=0.).validate()
    with pytest.raises(ValueError):
        category_probabilities(.5, 1.1, 0.)
    with pytest.raises(ValueError):
        contamination_flip_point(.5, (1., 0.))

