"""Algebraic diagnostic of posterior-ratio weighting; no welfare guarantee."""
import numpy as np
from interaction_sprint.theory import bayes_sdpo_gradient


def weighted_direction(p,l):
    pi=np.array([1-p,p]); marginal=pi@l
    posterior=pi[:,None]*l/marginal
    weights=posterior/pi[:,None]  # frozen in the differentiated loss
    score=np.array([-p,1-p])
    return -np.sum(marginal*pi[:,None]*weights*score[:,None]*(np.log(pi[:,None]/posterior)+1))


def test_joint_expectation_recovered():
    rng=np.random.default_rng(904)
    for _ in range(100):
        p=rng.uniform(.02,.98); q=rng.uniform(.02,.98,2)
        l=np.stack([1-q,q],axis=1)
        assert np.isclose(weighted_direction(p,l),bayes_sdpo_gradient(p,q),atol=1e-13)


def test_numerical_stopped_weight_loss():
    p=.3; l=np.array([[.9,.1],[.1,.9]])
    pi=np.array([1-p,p]); m=pi@l; posterior=pi[:,None]*l/m
    weights=posterior/pi[:,None]; theta=np.log(p/(1-p))
    def loss(t):
        v=1/(1+np.exp(-t)); s=np.array([1-v,v])
        return np.sum(m*s[:,None]*weights*np.log(s[:,None]/posterior))
    descent=-(loss(theta+1e-5)-loss(theta-1e-5))/2e-5
    assert np.isclose(descent,weighted_direction(p,l),atol=1e-10)


def test_balance_is_not_preference_optimum():
    # Restoring p=.5 for a symmetric channel need not improve utility if q=.9.
    p=.8; q=.9; d=weighted_direction(p,np.array([[.9,.1],[.1,.9]]))
    assert d<0
    utility_direction=(2*q-1)*p*(1-p)*d
    assert utility_direction<0
