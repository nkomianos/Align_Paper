import numpy as np
from interaction_sprint.hindsight_anchor_phase import direction, trajectory


def test_zero_influence_is_anchor_direction():
    for p in [.2,.5,.8]:
        assert np.isclose(direction(p,.7,0,.9),.1*(.7-p))


def test_complement_symmetry():
    assert np.isclose(direction(.3,.7,.6,.9),-direction(.7,.3,.6,.9))


def test_benign_adaptation():
    r=trajectory(.7,0,.5,.49)
    assert abs(r['p_final']-.7)<1e-8
    assert r['final_anchor_utility']>r['initial_anchor_utility']


def test_agreement_cannot_rise_while_welfare_falls_at_fixed_rho():
    # This structural identity prevents the originally sought discordant trend.
    q=.3; rho=.6
    for p in [.1,.3,.7,.9]:
        welfare=p*q+(1-p)*(1-q)
        enumerated=0.
        for z in [0,1]:
            for action in [0,1]:
                prob_z=q if z else 1-q
                prob_a=p if action else 1-p
                enumerated+=prob_z*prob_a*(rho+(1-rho)*(z==action))
        assert np.isclose(enumerated,rho+(1-rho)*welfare)
