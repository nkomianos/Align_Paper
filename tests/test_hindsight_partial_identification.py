import numpy as np
import pytest
from interaction_sprint.hindsight_partial_identification import preference_interval,witness,minimum_utility_change


def test_random_true_values_and_attainable_endpoints():
    rng=np.random.default_rng(20260904)
    for _ in range(1000):
        q,b=rng.random(2);e0,e1=rng.random(2)*b
        m0=(1-e0)*q;m1=(1-e1)*q+e1
        lo,hi=preference_interval(m0,m1,b)
        assert lo-1e-12<=q<=hi+1e-12
        for endpoint in (lo,hi):
            u,v=witness(m0,m1,endpoint)
            assert -1e-12<=u<=b+1e-12 and -1e-12<=v<=b+1e-12
            assert np.allclose([(1-u)*endpoint,(1-v)*endpoint+v],[m0,m1])


def test_extreme_bounds_and_incompatible_channel():
    assert preference_interval(.3,.3,0)==(.3,.3)
    assert preference_interval(.1,.9,1)==(.1,.9)
    with pytest.raises(ValueError):preference_interval(.1,.9,.1)
    for q in (0,1):
        assert preference_interval(q,q,0)==(q,q)


def test_filter_retains_clear_update_but_blocks_ambiguous_harm():
    assert minimum_utility_change(.5,.7,(.6,.8))>0
    interval=preference_interval(.4,.88,.8)
    assert minimum_utility_change(.5,.64,interval)<0
    assert minimum_utility_change(.5,.5,interval)==0


def test_misspecified_bound_can_pass_compatibility_and_certify_harm():
    # True q=.49, e0=0, e1=(.61-.49)/(.51)>.2.
    # The observed channel is nevertheless compatible with a DIFFERENT q
    # under an incorrectly asserted .2 influence bound.
    interval=preference_interval(.49,.61,.2)
    assert minimum_utility_change(.5,.55,interval)>0
    assert (.55-.5)*(2*.49-1)<0
