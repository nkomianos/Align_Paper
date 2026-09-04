import numpy as np
from scripts.audit_hindsight_anchor_value import policies,decide,interval


def test_no_vacuous_certificate_and_correct_directions():
    assert np.allclose(decide(np.array([.7,.2,.4,.8]),np.array([.9,.4,.7,.3])),[1,0,.5,.5])


def test_feedback_can_resolve_weak_anchor_and_not_erased_feedback():
    methods,_,_=policies(np.array([.5,.5]),np.array([.8,0.]),np.array([.9,1.]),16,500)
    assert np.allclose(methods['anchor_conservative'],[.5,.5])
    assert np.allclose(methods['combined_conservative'],[1,.5])


def test_exact_binomial_endpoints():
    lo,hi=interval(np.array([0.,1.]),16,.05,'clopper_pearson')
    assert np.allclose(lo,[0,.025**(1/16)])
    assert np.allclose(hi,[1-.025**(1/16),1])
