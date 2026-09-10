"""Finite-bank Bernoulli inclusion MSE with calibration-only allocation."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from analyze_reasoning_bank import read_bank,features
from run_unexplored_screens import sha,dump


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--gradients',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    rows,questions,prefixes=read_bank(a.bank)
    for name,h in json.loads((a.gradients/'MANIFEST.json').read_text()).items():assert sha(a.gradients/name)==h
    protocol=json.loads((a.gradients/'PROTOCOL.json').read_text());assert protocol['bank_manifest_sha256']==sha(a.bank/'MANIFEST.json')
    meta=json.loads((a.gradients/'ROWS.json').read_text());tensors=torch.load(a.gradients/'GRADIENTS.pt',map_location='cpu',weights_only=True)
    assert len(rows)==len(meta)==tensors['full'].shape[0]==tensors['early'].shape[0]
    assert tensors['full'].shape==tensors['early'].shape
    assert torch.isfinite(tensors['full']).all() and torch.isfinite(tensors['early']).all()
    for row,m in zip(rows,meta):
        assert all(row[k]==m[k] for k in ('base','prefix_index','sample','length'))
        assert row['sampling_logprob']==m['stored_logprob']
    discrepancy=max(abs(m['replayed_logprob']-m['stored_logprob'])/m['length'] for m in meta)
    indices=[i for i,r in enumerate(rows) if not r['prefix_ended'] and not r['prefix_has_answer']]
    cal=[i for i in indices if rows[i]['split']=='calibration'];dev=[i for i in indices if rows[i]['split']=='dev']
    rewards=np.array([r['reward'] for r in rows],dtype=float);lengths=np.array([rows[i]['length'] for i in dev],dtype=float)
    norms=tensors['full'].double().square().sum(1).numpy()
    x=np.asarray([features(r,questions,prefixes)+[meta[i]['early_logprob']/min(128,r['length'])/5] for i,r in enumerate(rows)])
    results={}
    for label,baseline in [('fixed_half',.5),('calibration_mean',float(rewards[cal].mean()))]:
        n2=(rewards-baseline)**2*norms
        target=np.log(1e-8+n2[cal]/np.maximum(1,[rows[i]['length']-128 for i in cal]))
        penalty=np.eye(x.shape[1])*10;penalty[0,0]=1e-8
        beta=np.linalg.solve(x[cal].T@x[cal]+penalty,x[cal].T@target)
        raw_cal=np.exp(np.clip(x[cal]@beta/2,-10,10));raw_dev=np.exp(np.clip(x[dev]@beta/2,-10,10))
        pdev=np.clip(.25/max(float(raw_cal.mean()),1e-8)*raw_dev,.1,1)
        initial=np.minimum(lengths,128);remaining=np.maximum(lengths-128,0)
        cost=float((initial+pdev*remaining).sum());full_cost=float(lengths.sum())
        p_uniform=(cost-float(initial.sum()))/max(float(remaining.sum()),1)
        includes={'adaptive':np.where(lengths<=128,1,pdev),
                  'uniform':np.where(lengths<=128,1,p_uniform),
                  'fewer_full':np.full(len(dev),cost/full_cost)}
        mse={name:float((((1-prob)/prob)*n2[dev]).sum()/len(dev)**2) for name,prob in includes.items()}
        results[label]={'reward_baseline':baseline,'mse':mse,'beta':beta.tolist(),
            'adaptive_probabilities':pdev.tolist(),'expected_tokens':cost,'full_tokens':full_cost,
            'uniform_continuation_probability':p_uniform,
            'passes_20pct_rule':mse['adaptive']<.8*min(mse['uniform'],mse['fewer_full'])}
    qualified=discrepancy<=.02 and len({rows[i]['base'] for i in cal})>=4 and len({rows[i]['base'] for i in dev})>=8
    route='INVALID_GRADIENT_REPLAY' if not qualified else ('FRESH_BANK_AND_TIME_VALIDATION_NEEDED' if all(r['passes_20pct_rule'] for r in results.values()) else 'STOP_NO_ROBUST_ADAPTIVE_VARIANCE_ADVANTAGE')
    report={'classification':'DEVELOPMENTAL_FINITE_BANK_PARAMETER_GRADIENT_AUDIT','route':route,
        'max_per_token_logprob_discrepancy':discrepancy,'cal_questions':len({rows[i]['base'] for i in cal}),
        'dev_questions':len({rows[i]['base'] for i in dev}),'results':results,
        'calibration_tokens':sum(rows[i]['length'] for i in cal),'gradient_manifest_sha256':sha(a.gradients/'MANIFEST.json'),
        'scope':'Conditional finite-bank design variance, not population gradient error, full-model training or GPU-time efficiency. Baselines define distinct empirical targets; compare inclusion methods only within each baseline.',
        'cost':'DEV lengths match comparator expected token costs retrospectively; they do not determine adaptive probabilities. Calibration and gradient replay costs reported separately; no optimizer step.'}
    dump(a.out,report);print(json.dumps({**{k:v for k,v in report.items() if k!='results'},'results':{k:{j:w for j,w in v.items() if j not in ('beta','adaptive_probabilities')} for k,v in results.items()}},indent=2))


if __name__=='__main__':main()
