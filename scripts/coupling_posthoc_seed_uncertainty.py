"""Declared post-hoc sensitivity analysis; does not change frozen primary tests.

Seed resampling is paired across policies/models within each question. Nested
question+seed bootstrap is descriptive with eight question clusters, not a new
significance gate or calibrated population interval.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from interaction_sprint.squad_coupling import score
from interaction_sprint.byte_clock_coupling import digest


def summarize(samples):
    finite=samples[np.isfinite(samples)]
    return {'descriptive_percentiles_2_5_50_97_5':np.quantile(finite,[.025,.5,.975]).tolist() if len(finite) else None,
            'undefined_draws':int(len(samples)-len(finite))}


def analyze(root,key_path,primary_path,out,draws=10000):
    assert not out.exists() and not out.resolve().is_relative_to(root.resolve())
    manifest=json.loads((root/'MANIFEST.json').read_text())
    assert all(digest(root/name)==sha for name,sha in manifest.items())
    primary=json.loads(primary_path.read_text());assert primary['manifest_sha']==digest(root/'MANIFEST.json')
    freeze=json.loads((root/'FROZEN.json').read_text());cfg=freeze['config'];key=json.loads(key_path.read_text())
    records=[json.loads(line) for line in (root/'outputs.jsonl').read_text().splitlines()]
    lookup={(r['model'],r['case_id'],r['policy'],r['seed']):r for r in records}
    models=[m['name'] for m in cfg['models']];policies=cfg['policies'];questions=[c['id'] for c in freeze['cases']]
    assert len(models)==2 and len(lookup)==len(records)
    # question x seed x policy x model
    x=np.array([[[[score(lookup[m,q,p,s]['text'],key[q])['f1'] for m in models]
                  for p in policies] for s in cfg['seeds']] for q in questions])
    def statistics(y):
        # sample x question x seed x policy x model
        centered=y-y.mean(axis=2,keepdims=True)
        cov=(centered[...,0]*centered[...,1]).sum(axis=2)/(y.shape[2]-1)
        variance=np.var(y[...,0]-y[...,1],axis=2,ddof=1)
        return variance.mean(axis=1),cov.mean(axis=1)
    actual_var,actual_cov=statistics(x[None])
    for i,p in enumerate(policies):
        assert np.isclose(actual_var[0,i],primary['policies'][p]['f1']['variance_difference'])
        assert np.isclose(actual_cov[0,i],primary['policies'][p]['f1']['covariance'])
    rng=np.random.default_rng(940904);result={};candidate=policies.index('byte_hierarchical')
    for mode in ['paired_seed_fixed_questions','nested_question_and_paired_seed']:
        variance=[];covariance=[]
        for begin in range(0,draws,100):
            n=min(100,draws-begin)
            q=rng.integers(len(questions),size=(n,len(questions))) if mode.startswith('nested') else np.tile(np.arange(len(questions)),(n,1))
            seeds=rng.integers(len(cfg['seeds']),size=(n,len(questions),len(cfg['seeds'])))
            sampled=x[q[:,:,None],seeds]
            v,c=statistics(sampled);variance.append(v);covariance.append(c)
        variance=np.concatenate(variance);covariance=np.concatenate(covariance)
        result[mode]={}
        for baseline in ['independent','token_clock','byte_clock']:
            i=policies.index(baseline)
            ratio=np.divide(variance[:,candidate],variance[:,i],out=np.full(draws,np.nan),where=variance[:,i]>0)
            result[mode][baseline]={'variance_ratio':summarize(ratio),
                 'twice_covariance_gain':summarize(2*(covariance[:,candidate]-covariance[:,i]))}
    result.update({'posthoc':True,'metric':'full-output SQuAD F1','draws':draws,
                   'manifest_sha':digest(root/'MANIFEST.json'),'primary_sha':digest(primary_path),
                   'key_sha':digest(key_path),'source_sha':digest(__file__),
                   'scope':'Uncertainty sensitivity only; keeps all policies and model pairs; no updated gate or significance claim.'})
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('key',type=Path)
    p.add_argument('primary',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();analyze(a.root,a.key,a.primary,a.out)
