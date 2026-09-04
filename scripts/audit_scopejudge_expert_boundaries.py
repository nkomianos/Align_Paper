"""Leave-one-reviewer-out descriptive boundary agreement; no model inference."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def measure(records, seed=20260904, bootstrap=5000):
    # Each trajectory is one resampling cluster across all five reviewers.
    counts=np.zeros((len(records),5,2,2),dtype=int)
    ties=0
    for i,record in enumerate(records):
        labels=record['extra']['scopejudge']['labels']
        order={s['step_id']:j for j,s in enumerate(record['steps'])}
        for reviewer in range(5):
            eligible=[]
            for label in labels:
                votes=[label[f'reviewer_{j+1}'] for j in range(5)]
                if any(type(v) is not bool for v in votes):raise ValueError('Nonboolean label')
                other=sum(votes)-votes[reviewer]
                if other==2:ties+=1
                if other>=3:eligible.append((order[label['step_id']],votes[reviewer]))
            if not eligible:continue
            first=min(step for step,_ in eligible)
            for step,positive in eligible:
                group=int(step!=first)
                counts[i,reviewer,group,0]+=positive
                counts[i,reviewer,group,1]+=1
    pooled=counts.sum(axis=(0,1))
    if not np.all(pooled[:,1]):
        raise ValueError('Both first and later groups require eligible reference positives')
    rates=pooled[:,0]/pooled[:,1]
    rng=np.random.default_rng(seed)
    samples=[]
    for _ in range(bootstrap):
        c=counts[rng.integers(len(records),size=len(records))].sum(axis=(0,1))
        if np.all(c[:,1]):samples.append(c[0,0]/c[0,1]-c[1,0]/c[1,1])
    return dict(estimand='Held-out expert positive agreement against >=3/4 other experts; first-positive step defined separately by those other experts. Not monitor accuracy or ground truth.',
                first=dict(positive=int(pooled[0,0]),total=int(pooled[0,1]),agreement=float(rates[0])),
                later=dict(positive=int(pooled[1,0]),total=int(pooled[1,1]),agreement=float(rates[1])),
                first_minus_later=float(rates[0]-rates[1]),
                trajectory_bootstrap_95_interval=np.quantile(samples,[.025,.975]).tolist(),
                seed=seed,bootstrap=bootstrap,excluded_tied_reviewer_call_pairs=ties,
                per_reviewer_counts=counts.sum(axis=0).tolist(),
                limitation='Exploratory, pooled reviewer-call pairs, reviewers not independent or resampled, no task-family generalization claim.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();raw=(a.data/'train.jsonl').read_bytes()
    sha=hashlib.sha256(raw).hexdigest()
    if sha!=json.loads((a.data/'dataset-manifest.json').read_text())['sha256']:raise ValueError('Hash mismatch')
    report=measure([json.loads(line) for line in raw.decode('utf8').splitlines()])
    report['dataset_sha256']=sha
    with a.out.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
