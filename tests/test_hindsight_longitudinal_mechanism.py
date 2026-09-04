import numpy as np
import pytest
from interaction_sprint.hindsight_longitudinal_mechanism import simulate,exact


def draws():
    rng=np.random.default_rng(4)
    return np.array([0,1]*16,dtype=bool),rng.random((8,32)),rng.random((8,32))


def test_stationary_direct_learning_exact_and_coupling_null():
    z,a,c=draws()
    out=[simulate(z,a,c,q=.8,eta=.1,rho=0,regime='stationary',coupling=arm,checkpoints=(8,)) for arm in ('closed','open')]
    assert np.array_equal(out[0][0],out[1][0])
    assert out[0][2][-1]['baseline_fidelity']==pytest.approx(1-.2*.9**8)


def test_no_learning_coupling_null():
    z,a,c=draws()
    out=[simulate(z,a,c,q=.8,eta=0,rho=.5,regime='persistent',coupling=arm) for arm in ('closed','open','frozen')]
    assert all(np.array_equal(o[0],out[0][0]) and np.array_equal(o[1],out[0][1]) for o in out)


def test_first_round_reports_match_but_retained_states_can_differ():
    z,a,c=draws()
    x=simulate(z,a,c,q=.5,eta=.5,rho=1,regime='expression',coupling='closed')
    y=simulate(z,a,c,q=.5,eta=.5,rho=1,regime='persistent',coupling='closed')
    assert np.array_equal(x[3],y[3])
    assert np.array_equal(x[1],z) and not np.array_equal(y[1],z)


def test_symmetric_model_no_excess_expected_baseline_harm():
    for regime in ('expression','persistent'):
        for rho in (0,.2,.5,.8,1):
            for eta in (0,.1,.5,1):
                for q in (.1,.5,.8,1):
                    for t in (1,8,64):
                        closed=exact(q,eta,rho,regime,'closed',t)
                        opened=exact(q,eta,rho,regime,'open',t)
                        assert closed['baseline_fidelity']>=opened['baseline_fidelity']-1e-12
                        assert closed['baseline_fidelity']>=q-1e-12
