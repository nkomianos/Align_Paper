"""Exact population identification under bounded action-copying feedback.

Elementary sensitivity model, not a novelty claim or finite-sample certificate.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def preference_interval(m0, m1, bound):
    if not all(np.isfinite(x) and 0<=x<=1 for x in (m0,m1,bound)):
        raise ValueError('Probabilities and bound must be finite in [0,1]')
    lo,hi=m0,m1
    if bound<1:
        lo=max(lo,(m1-bound)/(1-bound))
        hi=min(hi,m0/(1-bound))
    if lo>hi+1e-12:raise ValueError('Channel incompatible with bound')
    if lo>hi:lo=hi=(lo+hi)/2
    return float(lo),float(hi)


def witness(m0,m1,q):
    return (0. if q==0 else 1-m0/q,
            0. if q==1 else (m1-q)/(1-q))


def minimum_utility_change(p_before,p_after,interval):
    """Utility = probability recommendation matches pre-interaction preference."""
    if not (0<=p_before<=1 and 0<=p_after<=1):raise ValueError('Invalid policy')
    lo,hi=interval
    if not 0<=lo<=hi<=1:raise ValueError('Invalid identified set')
    return min((p_after-p_before)*(2*lo-1),(p_after-p_before)*(2*hi-1))


def audit():
    rows=[]
    for bound in (0.,.2,.5,.8,1.):
        counts=dict(cases=0,raw_harm=0,accepted=0,accepted_harm=0,raw_beneficial=0,beneficial_retained=0)
        for q in np.linspace(.1,.9,9):
            for e0 in np.linspace(0,bound,5):
                for e1 in np.linspace(0,bound,5):
                    m0=(1-e0)*q;m1=(1-e1)*q+e1
                    interval=preference_interval(m0,m1,bound)
                    # Transparent plug-in update, NOT SDPO: match mean feedback
                    # under balanced randomized logging, starting from p=0.5.
                    p_new=(m0+m1)/2
                    gain=(p_new-.5)*(2*q-1)
                    accept=minimum_utility_change(.5,p_new,interval)>=-1e-12
                    counts['cases']+=1
                    counts['raw_harm']+=int(gain < -1e-12)
                    counts['accepted']+=int(accept)
                    counts['accepted_harm']+=int(accept and gain < -1e-12)
                    counts['raw_beneficial']+=int(gain > 1e-12)
                    counts['beneficial_retained']+=int(accept and gain > 1e-12)
        rows.append(dict(bound=bound,**counts))
    return dict(scope='Designed population grid with known valid influence bounds; no LM training, sampled-data inference, or human welfare claim.',
                update='Mean-feedback plug-in from balanced logging; identified-set filter accepts only nonnegative worst-case initial-preference utility change.',
                rows=rows,grid_note='225 cells per bound; zero-bound cells repeat. Counts are not prevalence estimates.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=audit()
    with a.out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
