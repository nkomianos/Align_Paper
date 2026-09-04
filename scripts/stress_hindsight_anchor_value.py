"""Designed measurement-misspecification and hypothetical cost tests."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.audit_hindsight_anchor_value import policies,interval,decide


def score(p,q):
    utility=p*q+(1-p)*(1-q)
    return dict(utility=float(utility.mean()),harm=float(np.mean(utility<.5-1e-12)),
                decisions=float(np.mean(p!=.5)))


def audit(repeats=2000,seed=20260906):
    rng=np.random.default_rng(seed);stress=[];cost=[]
    for q in (.2,.4,.45,.55,.6,.8):
        for e0 in (0.,.3,.8):
            for e1 in (0.,.3,.8):
                k_anchor=rng.binomial(16,q,repeats);anchor=k_anchor/16
                for delta in (0.,.05,.1,.2):
                    # Deliberately adverse measurement shift, NOT observed users.
                    shift=delta if q<.5 else -delta
                    m0=rng.binomial(500,np.clip((1-e0)*q+shift,0,1),repeats)/500
                    m1=rng.binomial(500,np.clip((1-e1)*q+e1+shift,0,1),repeats)/500
                    naive,_,_=policies(anchor,m0,m1,16,500,kind='clopper_pearson')
                    robust,_,_=policies(anchor,m0,m1,16,500,kind='clopper_pearson',slack=delta)
                    stress.append(dict(q=q,e0=e0,e1=e1,delta=delta,
                        anchor=score(naive['anchor_conservative'],q),
                        naive=score(naive['combined_conservative'],q),
                        known_slack=score(robust['combined_conservative'],q)))
                    if delta==0:
                        for ratio in (0.,.001,.01,.05,.1,1.):
                            extra=int(1000*ratio)
                            ka=k_anchor+rng.binomial(extra,q,repeats)
                            lo,hi=interval(ka/(16+extra),16+extra,.05,'clopper_pearson')
                            cost.append(dict(q=q,e0=e0,e1=e1,ratio=ratio,extra_anchors=extra,
                                anchor=score(decide(lo,hi),q),
                                combined=score(naive['combined_conservative'],q)))
    def aggregate(rows,key,methods):
        return [{key:value,**{method:{metric:float(np.mean([r[method][metric] for r in rows if r[key]==value]))
                for metric in ('utility','harm','decisions')} for method in methods}}
                for value in sorted({r[key] for r in rows})]
    return dict(seed=seed,repeats=repeats,scope='Designed cells, no human or LM measurement; cost ratios hypothetical. Truthful iid pre-treatment anchors remain assumed.',
                stress=stress,cost=cost,
                stress_summary=aggregate(stress,'delta',('anchor','naive','known_slack')),
                cost_summary=aggregate(cost,'ratio',('anchor','combined')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=audit()
    with a.out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:result[k] for k in ('stress_summary','cost_summary')},indent=2))
