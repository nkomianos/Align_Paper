import numpy as np
from interaction_sprint.bayes_hindsight_objective_audit import gradients


def test_forward_posterior_average_identity():
    rng=np.random.default_rng(903)
    for _ in range(100):
        p=rng.uniform(.01,.99); q=rng.uniform(.01,.99,2)
        r=gradients(p,np.stack([1-q,q],axis=1))
        assert abs(r['forward_kl'])<1e-14


def test_uninformative_no_update():
    r=gradients(.7,[[.3,.7],[.3,.7]])
    assert all(abs(v)<1e-14 for v in r.values())


def test_joint_and_marginal_objectives_differ():
    r=gradients(.1,[[.9,.1],[.1,.9]])
    assert r['reverse_kl']<0 and r['own_action_logratio']>0
    assert not np.isclose(r['reverse_kl'],r['own_action_logratio'])


def test_reverse_independent_finite_difference():
    for p in [.1,.3,.5,.7,.9]:
        l=np.array([[.9,.1],[.1,.9]])
        pi=np.array([1-p,p]); m=pi@l; teacher=pi[:,None]*l/m
        theta=np.log(p/(1-p))
        def loss(t):
            s=1/(1+np.exp(-t)); student=np.array([1-s,s])
            return np.sum(m*np.sum(student[:,None]*np.log(student[:,None]/teacher),axis=0))
        descent=-(loss(theta+1e-5)-loss(theta-1e-5))/2e-5
        exact=p*(1-p)*(2*p-1)*.8*np.log(9)
        assert np.isclose(descent,exact,atol=1e-10)
        assert np.isclose(gradients(p,l)['reverse_kl'],exact,atol=1e-14)
