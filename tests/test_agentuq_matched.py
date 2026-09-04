import numpy as np
import pytest

from interaction_sprint.agentuq_matched import compare, keyed, loo_constant, percentiles, selector


def test_percentiles_monotone_and_ties():
    assert np.allclose(percentiles([7, 7, 9]), [1/3, 1/3, 5/6])
    assert np.allclose(percentiles([7, 7, 9]), percentiles([70, 70, 90]))
    with pytest.raises(ValueError):
        percentiles([float("nan")])


def test_duplicate_tasks_rejected():
    with pytest.raises(ValueError):
        keyed([{"id": "a"}, {"id": "a"}], "id")


def test_selection_orientation_ties():
    y = [[1, 0], [0, 1], [1, 0], [1, 1]]
    assert np.array_equal(selector(y, [[0, 1], [1, 0], [0, 0], [0, 0]]), [1, 1, .5, 1])


def test_loo_excludes_own_outcome():
    # Each task's winner is the other's loser, so leave-one-out always loses.
    assert np.array_equal(loo_constant([[1, 0], [0, 1]]), [0, 0])
    assert np.array_equal(loo_constant([[1, 0], [1, 0]]), [1, 1])


def test_matched_signal_and_empty_score():
    left = {"a": {"failure": 0, "s": 0}, "b": {"failure": 1, "s": 2},
            "c": {"failure": 0, "s": None}}
    right = {"a": {"failure": 1, "s": 2}, "b": {"failure": 0, "s": 0},
             "c": {"failure": 1, "s": 1}}
    result = compare(left, right, "s")
    assert result["tasks"] == 2
    assert result["discordant_concordance"] == 1
    assert result["rank_selector_success"] == result["oracle_success"] == 1
    assert result["pooled_rank_auroc"] == 1
    assert result["conditional_bootstrap95_delta"] == [1, 1]


def test_label_invalid():
    with pytest.raises(ValueError):
        selector([[.5, 1]], [[1, 0]])
