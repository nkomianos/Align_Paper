"""Designed exact population model, not learned human preferences or LM results."""
import argparse
import json
from pathlib import Path
import numpy as np
from .bayes_hindsight_objective_audit import gradients


def direction(p, q, rho, weight):
    # Initial preference Z~Bernoulli(q); exposure copies action with probability rho.
    # Observations truthfully report post-exposure preference.
    message=np.array([(1-rho)*q, (1-rho)*q+rho])
    likelihood=np.stack([1-message,message],axis=1)
    feedback=gradients(p,likelihood)['reverse_kl']
    # Independent supervised initial-preference anchors. q is known exactly here.
    anchor=q-p
    return (1-weight)*anchor + weight*feedback


def trajectory(q,rho,weight,p0,steps=3000,lr=.2):
    t=float(np.log(p0/(1-p0)))
    for _ in range(steps):
        p=float(1/(1+np.exp(-t)))
        t+=lr*direction(p,q,rho,weight)
    p=float(1/(1+np.exp(-t)))
    return dict(q=q,rho=rho,feedback_weight=weight,p0=p0,p_final=p,
                initial_anchor_utility=p0*q+(1-p0)*(1-q),
                final_anchor_utility=p*q+(1-p)*(1-q),
                observed_agreement=rho+(1-rho)*(p*q+(1-p)*(1-q)))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
    rows=[trajectory(q,rho,w,p) for q in [.3,.7] for rho in [0,.3,.6,.9]
          for w in [0,.5,.9,.99] for p in [.49,.51]]
    result=dict(scope='Designed population simulation; known perfect anchors; not a novel theorem',
                update='logit SGD on weighted anchor cross-entropy and stopped Bayesian reverse KL',
                learning_rate=.2,steps=3000,rows=rows)
    with (a.out/'phase.json').open('x') as f:json.dump(result,f,indent=2)
    harmed=[r for r in rows if r['final_anchor_utility']<r['initial_anchor_utility']-.01]
    print(json.dumps(dict(cases=len(rows),harm_cases=len(harmed),examples=harmed[:6]),indent=2))


if __name__=='__main__': main()
