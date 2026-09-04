import numpy as np
import pytest

from interaction_sprint.cluster_certificate_audit import (
    cp_lower, exact_cluster_case, cluster_hoeffding_certificate,
)


def test_iid_and_task_certificates_control_false_pass():
    for n in [20, 50, 100]:
        for m in [1, 4, 8, 32]:
            row = exact_cluster_case(n, m, .95, .97)
            assert row["taskwise_false_certificate_probability"] <= .05 + 1e-12
            assert row["iid_episode_false_certificate_probability"] <= .05 + 1e-12


def test_explicit_pseudoreplication_counterexample():
    row = exact_cluster_case(20, 8, .95, .97)
    assert np.isclose(row["pooled_all_survive_lower"], .05**(1/160))
    assert row["pooled_all_survive_lower"] > .97
    assert np.isclose(row["pooled_false_certificate_probability"], .95**20)
    assert row["taskwise_all_survive_lower"] < .97


def test_repeats_do_not_change_cluster_bound():
    y = np.ones((100, 1))
    a = np.ones_like(y)
    once = cluster_hoeffding_certificate(y, a, .9)
    repeated = cluster_hoeffding_certificate(np.repeat(y, 50, axis=1), np.repeat(a, 50, axis=1), .9)
    assert np.isclose(once["lower_contrast"], repeated["lower_contrast"])


def test_ratio_not_average_of_task_conditional_ratios():
    y = np.array([[1, 1, 1, 1], [1, 0, 0, 0]])
    a = np.array([[1, 1, 1, 1], [0, 0, 0, 0]])
    row = cluster_hoeffding_certificate(y, a, .7)
    assert row["observed_recall"] == .8  # not (1 + 0)/2
    assert np.isclose(row["mean_contrast"], (.8-.7)*y.mean())


def test_no_success_cannot_certify():
    row = cluster_hoeffding_certificate(np.zeros((100, 8)), np.ones((100, 8)), .9)
    assert row["observed_recall"] is None and not row["certified"]


def test_invalid_inputs():
    with pytest.raises(ValueError):
        cp_lower(3, 2)
    with pytest.raises(ValueError):
        cluster_hoeffding_certificate(np.ones((2, 1)), np.ones((2, 2)), .9)
