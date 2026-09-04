"""Finite-sample designed comparison; no LLM or human data."""
import argparse
import json
from pathlib import Path
import numpy as np


def interval(mean,n,alpha,kind='hoeffding'):
    if kind=='clopper_pearson':
        from scipy.stats import beta
        k=np.rint(np.asarray(mean)*n).astype(int)
        values,inverse=np.unique(k,return_inverse=True)
        lower=np.zeros(len(values));upper=np.ones(len(values))
        mask=values>0
        lower[mask]=beta.ppf(alpha/2,values[mask],n-values[mask]+1)
        mask=values<n
        upper[mask]=beta.ppf(1-alpha/2,values[mask]+1,n-values[mask])
        return lower[inverse].reshape(k.shape),upper[inverse].reshape(k.shape)
    if kind!='hoeffding':raise ValueError('Unknown interval')
    radius=np.sqrt(np.log(2/alpha)/(2*n))
    return np.maximum(0,mean-radius),np.minimum(1,mean+radius)


def decide(lo,hi):
    # Incompatibility is never interpreted as a vacuous safety certificate.
    return np.where(lo>hi,.5,np.where(lo>.5,1.,np.where(hi<.5,0.,.5)))


def policies(anchor,m0,m1,n_anchor,n_feedback,alpha=.05,kind='hoeffding',slack=0.):
    if not 0<=slack<=1:raise ValueError('Slack must be in [0,1]')
    alo,ahi=interval(anchor,n_anchor,alpha,kind)
    # A total alpha budget, not three uncorrected 95% intervals.
    jlo,jhi=interval(anchor,n_anchor,alpha/2,kind)
    m0lo,_=interval(m0,n_feedback,alpha/4,kind)
    _,m1hi=interval(m1,n_feedback,alpha/4,kind)
    lo=np.maximum(jlo,m0lo-slack);hi=np.minimum(jhi,m1hi+slack)
    flo,_=interval(m0,n_feedback,alpha/2,kind)
    _,fhi=interval(m1,n_feedback,alpha/2,kind)
    return dict(anchor_plugin=np.where(anchor>.5,1.,np.where(anchor<.5,0.,.5)),
                anchor_conservative=decide(alo,ahi),
                feedback_conservative=decide(np.maximum(0,flo-slack),np.minimum(1,fhi+slack)),
                combined_conservative=decide(lo,hi)),lo,hi


def audit(seed=20260905,repeats=2000,kind='hoeffding'):
    rng=np.random.default_rng(seed);rows=[]
    for q in (.2,.4,.6,.8):
        for e0 in (0.,.3,.8):
            for e1 in (0.,.3,.8):
                for n in (16,64,256):
                    anchor=rng.binomial(n,q,repeats)/n
                    m0=rng.binomial(500,(1-e0)*q,repeats)/500
                    m1=rng.binomial(500,(1-e1)*q+e1,repeats)/500
                    methods,lo,hi=policies(anchor,m0,m1,n,500,kind=kind)
                    scores={}
                    for name,p in methods.items():
                        utility=p*q+(1-p)*(1-q)
                        scores[name]=dict(mean_utility=float(utility.mean()),
                            harm_rate=float(np.mean(utility<.5-1e-12)),
                            decision_rate=float(np.mean(p!=.5)))
                    rows.append(dict(q=q,e0=e0,e1=e1,anchors=n,methods=scores,
                        combined_set_miss_rate=float(np.mean((q<lo)|(q>hi))),
                        incompatible_rate=float(np.mean(lo>hi))))
    aggregate=[]
    for n in (16,64,256):
        cells=[r for r in rows if r['anchors']==n]
        aggregate.append(dict(anchors=n,methods={name:{metric:float(np.mean([
            r['methods'][name][metric] for r in cells])) for metric in
            ('mean_utility','harm_rate','decision_rate')} for name in cells[0]['methods']}))
    return dict(seed=seed,repeats_per_cell=repeats,feedback_per_action=500,alpha=.05,interval_kind=kind,
        scope='Designed independent Bernoulli users; equal-cell averages are not human-population prevalence. No neural learner or cost parity: combined methods receive 1000 extra feedback samples.',
        assumptions='Randomized actions, truthful pre-treatment anchors, action-copying expression/transition only. No tight influence bound: uses m0<=q<=m1.',
        rows=rows,aggregate=aggregate)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--interval',choices=['hoeffding','clopper_pearson'],default='hoeffding');a=p.parse_args()
    result=audit(kind=a.interval)
    with a.out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result['aggregate'],indent=2))
