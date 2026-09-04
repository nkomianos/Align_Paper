"""Exact finite-channel audit; not a claim about actual LM teachers."""
import argparse
import json
from pathlib import Path
import numpy as np
from .theory import bayes_sdpo_gradient


def gradients(p, likelihood):
    """Stopped Bayesian teacher, observations sampled under current policy.

Returns descent directions in the student's scalar logit. No differentiation
through either teacher or the observation distribution.
"""
    l = np.asarray(likelihood, dtype=float)
    if l.shape != (2, 2) or (l <= 0).any() or not np.allclose(l.sum(1), 1):
        raise ValueError('Strictly positive binary channel required')
    if not 0 < p < 1:
        raise ValueError('Interior policy required')
    pi = np.array([1-p, p])
    m = pi @ l
    posterior = pi[:, None]*l/m
    forward_descent = float(np.sum(m*(posterior[1]-p)))
    ratio = np.log(pi[:, None]/posterior)
    reverse_descent = float(-p*(1-p)*np.sum(m*(ratio[1]-ratio[0])))
    # Reverse KL is averaged over observations; this differs from a sampled
    # action's own feedback conditioned on that same action.
    joint_score_descent = bayes_sdpo_gradient(p, l[:, 1])
    return dict(forward_kl=forward_descent, reverse_kl=reverse_descent,
                own_action_logratio=joint_score_descent)


def audit():
    rows=[]
    for name, l in [('uninformative', [[.3,.7],[.3,.7]]),
                    ('symmetric', [[.9,.1],[.1,.9]]),
                    ('asymmetric', [[.8,.2],[.05,.95]])]:
        for p in [.1,.3,.5,.7,.9]:
            rows.append(dict(channel=name, p=p, directions=gradients(p,l)))
    return dict(scope='Exact stopped-Bayesian-teacher finite model, not neural SDPO',
                rows=rows,
                qualification='No action-independent feedback channel can demonstrate adaptation under these assumptions.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=False)
    result=audit()
    with (args.out/'audit.json').open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
