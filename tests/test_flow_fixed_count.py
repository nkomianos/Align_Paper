import numpy as np
from interaction_sprint.flow_fixed_count import targets, sample_scores


def test_targets_fixed_and_positive():
    assert targets() == targets() and len(targets()) == 16
    for t in targets():
        assert np.linalg.eigvalsh(t['cov']).min() > 0


def test_endpoint_scores_invariant_under_shared_affine_units():
    rng = np.random.default_rng(15)
    x = rng.normal(size=(64, 2)); y = rng.normal(.1, 1.1, size=(64, 2))
    a = np.array([[2., .3], [.1, .4]]); b = np.array([.4, -2.])
    left, right = sample_scores(x, y), sample_scores(x@a+b, y@a+b)
    for key in left:
        assert np.isclose(left[key], right[key], atol=1e-10)
