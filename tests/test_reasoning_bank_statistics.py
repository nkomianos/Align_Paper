import itertools
import numpy as np
from analyze_reasoning_bank import variance


def test_bernoulli_ht_variance_matches_exhaustive_design():
    g=np.array([[1.,-2.],[3.,4.],[-1.,.5]])
    p=np.array([.2,.6,1.])
    truth=g.mean(0);mse=0.;mean=np.zeros(2)
    for bits in itertools.product((0,1),repeat=3):
        b=np.array(bits);prob=np.prod(np.where(b,p,1-p))
        estimate=((b/p)[:,None]*g).mean(0)
        mean+=prob*estimate;mse+=prob*np.sum((estimate-truth)**2)
    np.testing.assert_allclose(mean,truth)
    np.testing.assert_allclose(mse,variance(g,p))


def test_temperature_digit_bias_derivative():
    logits=np.array([.1,-.3,1.2]);temperature=.8;selected=1
    def logprob(bias):
        x=(logits+bias)/temperature
        return x[selected]-np.log(np.exp(x).sum())
    probabilities=np.exp(logits/temperature);probabilities/=probabilities.sum()
    expected=(np.eye(3)[selected]-probabilities)/temperature
    for k in range(3):
        delta=np.eye(3)[k]*1e-5
        np.testing.assert_allclose((logprob(delta)-logprob(-delta))/2e-5,expected[k],rtol=1e-6)
