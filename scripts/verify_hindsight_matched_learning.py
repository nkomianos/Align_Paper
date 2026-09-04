"""Read-only hashes, design, loss arithmetic, outcomes and saved-state checks.

Does not independently rerun neural backwards or optimizer updates.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from interaction_sprint.hindsight_matched_learning import data_and_schedule,ARMS,SCORE_SHA,STEPS


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();root=a.root
    manifest=json.loads((root/'MANIFEST.json').read_text())
    for name,digest in manifest.items():
        f=(root/name).resolve()
        if f.parent!=root.resolve() or hashlib.sha256(f.read_bytes()).hexdigest()!=digest:raise ValueError('Manifest mismatch')
    cases,schedule=data_and_schedule();lookup={c['id']:c for c in cases}
    if json.loads((root/'cases.json').read_text())!=cases or json.loads((root/'schedule.json').read_text())!=schedule:raise ValueError('Design mismatch')
    raw=(root/'teacher_scores.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SCORE_SHA:raise ValueError('Teacher mismatch')
    teacher_rows=json.loads(raw)
    teachers={(r['id'],r['target']):np.array(r['probabilities']) for r in teacher_rows if r['kind']=='published'}
    baseline=json.loads((root/'baseline.json').read_text())
    initial_p={r['id']:np.array(r['probabilities']) for r in baseline if r['view']=='train'}
    original_base={r['id']:np.array(r['probabilities']) for r in teacher_rows if r['kind']=='base'}
    replay=max(float(np.max(np.abs(initial_p[c]-original_base[c]))) for c in initial_p)
    logs=json.loads((root/'steps.json').read_text())
    if len(logs)!=6*STEPS:raise ValueError('Step count')
    first={r['arm']:r for r in logs if r['step']==1}
    if first['copying_kl']['examples']!=first['fixed_marginal_kl']['examples'] or first['copying_kl']['gradient_norm']!=first['fixed_marginal_kl']['gradient_norm']:
        raise ValueError('Initial matched-feedback control differs')
    max_loss_error=0.;exposures=Counter()
    for log in logs:
        arm=log['arm'];batch=schedule[log['step']-1]
        expected=[cid for cid in batch if arm!='anchors_only' or lookup[cid]['anchor']]
        if [r['id'] for r in log['examples']]!=expected:raise ValueError('Batch mismatch')
        for row in log['examples']:
            cid=row['id'];c=lookup[cid];prob=np.array(row['probabilities'])
            if not np.isfinite(prob).all() or (prob<=0).any() or not np.isclose(prob.sum(),1):raise ValueError('Invalid probability')
            if arm in ('truthful_direct','anchors_only'):
                loss=-np.log(prob[c['target']])
                if row['marginal'] is not None:raise ValueError('Unexpected marginal')
            else:
                rho=0 if arm=='truthful_kl' else .9
                marginal=rho*(initial_p[cid] if arm=='fixed_marginal_kl' else prob)
                marginal[c['target']]+=1-rho
                if not np.allclose(marginal,row['marginal'],atol=1e-12,rtol=1e-12):raise ValueError('Feedback law mismatch')
                q=np.stack([teachers[cid,o] for o in (0,1)])
                loss=float(np.sum(marginal[:,None]*prob[None,:]*(np.log(prob)[None,:]-np.log(q))))
                if arm=='copying_plus_anchors' and c['anchor']:loss-=4*np.log(prob[c['target']])
            max_loss_error=max(max_loss_error,abs(float(loss)-row['loss']))
            if abs(float(loss)-row['loss'])>1e-9:raise ValueError('Loss mismatch')
            exposures[arm]+=1
    if dict(exposures)!={arm:24 if arm=='anchors_only' else 96 for arm in ARMS}:raise ValueError('Exposure count')
    result=json.loads((root/'RESULT.json').read_text())
    if result['counts']!=dict(forwards=728,backwards=504,updates=144):raise ValueError('Execution count')
    initial=torch.load(root/'initial_adapter.pt',weights_only=True,map_location='cpu')
    for arm in ARMS:
        saved=torch.load(root/(arm+'_adapter.pt'),weights_only=True,map_location='cpu')
        if saved.keys()!=initial.keys() or any(saved[k].shape!=initial[k].shape or not torch.isfinite(saved[k]).all() for k in initial):raise ValueError('Bad adapter')
        opt=torch.load(root/(arm+'_optimizer.pt'),weights_only=True,map_location='cpu')
        if not opt['state'] or any(int(s['step'])!=24 for s in opt['state'].values()):raise ValueError('Optimizer step mismatch')
    report=dict(status='HASH_DESIGN_LOSS_ARITHMETIC_AND_STATES_VERIFIED_NOT_OPTIMIZER_REPLAY',
        counts=result['counts'],elapsed_seconds=result['elapsed'],max_loss_error=max_loss_error,
        baseline_vs_previous_score_max_difference=replay,metrics={})
    for arm in ('no_adaptation',)+ARMS:
        rows=baseline if arm=='no_adaptation' else json.loads((root/(arm+'_eval.json')).read_text())
        if {(r['id'],r['view']) for r in rows}!={(c['id'],v) for c in cases for v in ('train','eval')} or len(rows)!=32:raise ValueError('Evaluation coverage')
        report['metrics'][arm]={}
        for view in ('train','eval'):
            rs=[r for r in rows if r['view']==view]
            metric=dict(n=16,correct=sum(int(np.argmax(r['probabilities'])==r['target']) for r in rs),
                nll=float(np.mean([r['nll'] for r in rs])),min_AB_mass=min(r['AB_mass'] for r in rs),
                mean_target_probability=float(np.mean([r['probabilities'][r['target']] for r in rs])),
                unanchored_correct=sum(int(np.argmax(r['probabilities'])==r['target']) for r in rs if not r['anchor']),unanchored_n=12)
            metric['by_target']={str(z):dict(n=sum(r['target']==z for r in rs),
                correct=sum(int(np.argmax(r['probabilities'])==z) for r in rs if r['target']==z)) for z in (0,1)}
            metric['anchored_correct']=metric['correct']-metric['unanchored_correct'];metric['anchored_n']=4
            for r in rs:
                if r['target']!=lookup[r['id']]['target'] or not np.isclose(-np.log(r['probabilities'][r['target']]),r['nll'],atol=1e-10):raise ValueError('Outcome mismatch')
            if arm!='no_adaptation' and (metric['correct']!=result['results'][arm][view]['correct'] or not np.isclose(metric['nll'],result['results'][arm][view]['nll'])):raise ValueError('Summary mismatch')
            report['metrics'][arm][view]=metric
    with a.out.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
