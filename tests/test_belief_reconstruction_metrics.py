import numpy as np
import pytest
from interaction_sprint.belief_reconstruction_metrics import paired_metrics,measurement_null


def test_paired_direction_and_zero_shift():
    result=paired_metrics([1,2],[2,3],[1,2],[2,3])
    assert result['paired_pre_mse_change']==1
    assert result['late_post_mse']==0
    assert paired_metrics([1],[2],[1],[1])['prediction_shift_times_observed_change']==0


def test_bad_vectors_fail_explicitly():
    for args in (([],[],[],[]),([1],[2,3],[1],[1]),([np.nan],[1],[1],[1])):
        with pytest.raises(ValueError):paired_metrics(*args)


def test_shared_measurement_counterexample():
    result=measurement_null()
    assert result['oracle_latent_mse']==0
    assert abs(result['shared_measurement_covariance']-.16)<.004
    assert abs(result['error_change_correlation']-1/np.sqrt(2))<.008
