from scripts.audit_coupling_covariance import decompose


def test_marginal_change_does_not_masquerade_as_covariance():
    baseline = dict(variance_a=1, variance_b=1, covariance=0, variance_difference=2)
    candidate = dict(variance_a=.5, variance_b=.5, covariance=0, variance_difference=1)
    result = decompose(candidate, baseline)
    assert result['sample_covariance_contribution'] == 0
    assert result['sample_marginal_variance_contribution'] == 1
    assert result['candidate_variance_over_own_marginal_sum'] == 1


def test_positive_covariance_accounts_for_reduction():
    baseline = dict(variance_a=1, variance_b=1, covariance=0, variance_difference=2)
    candidate = dict(variance_a=1, variance_b=1, covariance=.5, variance_difference=1)
    result = decompose(candidate, baseline)
    assert result['sample_covariance_contribution'] == 1
    assert result['sample_marginal_variance_contribution'] == 0


def test_degenerate_marginals_are_undefined():
    zero = dict(variance_a=0, variance_b=0, covariance=0, variance_difference=0)
    assert decompose(zero, zero)['candidate_variance_over_own_marginal_sum'] is None
