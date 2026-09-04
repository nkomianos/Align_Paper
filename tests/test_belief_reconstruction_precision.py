import numpy as np
import pytest
from scripts.audit_belief_reconstruction_precision import precision


def test_independent_and_fully_shared_limits():
    assert precision([10, 10, 10], 0)['variance_equivalent_independent_n'] == 30
    assert precision([10, 10, 10], 1)['variance_equivalent_independent_n'] == 3
    assert precision([10, 20], 1)['variance_equivalent_independent_n'] == pytest.approx(1.8)


def test_gaussian_variance_matches_formula():
    rng = np.random.default_rng(876)
    sizes = np.array([2, 5, 11, 7])
    rho = .3
    # Cluster means retain individual noise variance divided by cluster size.
    means = rng.normal(size=(100000, 4))*np.sqrt(rho+(1-rho)/sizes)
    average = means @ (sizes/sizes.sum())
    assert np.var(average) == pytest.approx(precision(sizes, rho)['standardized_se']**2, rel=.02)


def test_bad_design_rejected():
    for sizes, rho in (([1], .1), ([1, 0], .1), ([1, np.nan], .1), ([1, 2], 2)):
        with pytest.raises(ValueError):
            precision(sizes, rho)
