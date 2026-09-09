"""Descriptive paired base-task intervals; no new gate or seed-level inference."""
import argparse
import json
from pathlib import Path
import numpy as np


def summarize(root):
    read=lambda name:json.loads((root/name).read_text(encoding='utf8'))
    result=read('RESULT.json')
    if result['decision']=='STOP_INVALID_INITIAL_TEACHER':
        return {'decision':result['decision'],'comparisons':[]}
    names=['baseline_student']+[a+'_final_student' for a in ('supervised','frozen_teacher','current_teacher')]
    metrics={n:read(n+'.json')['summary'] for n in names}
    bases=sorted(metrics[names[0]]['base_means'])
    if any(set(m['base_means'])!=set(bases) for m in metrics.values()):raise ValueError('unpaired task sets')
    rng=np.random.default_rng(2026090901)
    indices=rng.integers(len(bases),size=(20000,len(bases)))
    pairs=[(n,names[0]) for n in names[1:]]+[(names[3],names[2])]
    comparisons=[]
    for treatment,control in pairs:
        for field in ('nll','correct','conditional_probability','probability'):
            delta=np.array([metrics[treatment]['base_means'][b][field]-metrics[control]['base_means'][b][field] for b in bases])
            interval=np.quantile(delta[indices].mean(1),[.025,.975]).tolist()
            comparisons.append({'treatment':treatment,'control':control,'metric':field,
                'mean_treatment_minus_control':float(delta.mean()),'pointwise_95_percentile_interval':interval,
                'improved_bases':int(((delta<0) if field=='nll' else (delta>0)).sum()),
                'worsened_bases':int(((delta>0) if field=='nll' else (delta<0)).sum())})
    fields=('nll','correct','conditional_probability','probability','choice_mass','mean_position_range')
    return {'decision':result['decision'],'bases':len(bases),'bootstrap_draws':20000,
        'seed':2026090901,'summaries':{n:{k:m[k] for k in fields} for n,m in metrics.items()},
        'comparisons':comparisons,'scope':'Descriptive task resampling within one exposed developmental population and one training seed. Pointwise intervals are not multiplicity-adjusted, confirmatory tests, or uncertainty across model trainings.',
        'paper_green_light':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.resolve().is_relative_to(a.root.resolve()):raise ValueError('report must be outside sealed evidence')
    report=summarize(a.root)
    with a.out.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({'decision':report['decision'],'comparisons':len(report['comparisons'])}))
