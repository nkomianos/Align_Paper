"""Verify saved files, exact prompt construction, and descriptive arithmetic."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from interaction_sprint.hindsight_revelation_probe import fresh_cases, published_template, published_text


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--upstream-config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();template=published_template(a.upstream_config)
    manifest=json.loads((a.root/'MANIFEST.json').read_text())
    for name,digest in manifest.items():
        f=(a.root/name).resolve()
        if f.parent!=a.root.resolve() or hashlib.sha256(f.read_bytes()).hexdigest()!=digest:raise ValueError('Bad manifest')
    rows=json.loads((a.root/'scores.json').read_text())
    lookup={(r['id'],r['target'],r['kind']):r for r in rows}
    if len(rows)!=128 or len(lookup)!=128:raise ValueError('Wrong count')
    for c in fresh_cases():
        for mode in ('base','published_null'):
            expected=c['prompt']+(template.format(follow_up='Thanks for responding.') if mode=='published_null' else '')
            if lookup[(c['id'],None,mode)]['prompt']!=expected:raise ValueError('Wrong control')
        for target in (0,1):
            for mode in ('published','direct','redacted'):
                r=lookup[(c['id'],target,mode)]
                if r['prompt']!=published_text(c,target,mode,template) or r['action'] is not None:raise ValueError('Wrong prompt')
    for r in rows:
        probs=np.asarray(r['probabilities'])
        if probs.shape!=(2,) or not np.isfinite(probs).all() or (probs<0).any() or not np.isclose(probs.sum(),1):raise ValueError('Invalid probabilities')
        if not 0<=r['AB_mass']<=1+1e-10:raise ValueError('Invalid mass')
    report=dict(status='MANIFEST_PROMPT_CONTRACT_AND_ARITHMETIC_VERIFIED_NOT_NEURAL_REPLAY',
        forwards=128,updates=0,minimum_AB_mass=min(r['AB_mass'] for r in rows),arms={})
    original=json.loads((a.root/'RESULT.json').read_text())
    for mode in ('published','direct','redacted'):
        rs=[r for r in rows if r['kind']==mode]
        correct=sum(int(np.argmax(r['probabilities'])==r['target']) for r in rs)
        mean=float(np.mean([r['probabilities'][r['target']] for r in rs]))
        if original['arms'][mode]['correct']!=correct or not np.isclose(original['arms'][mode]['mean_target_probability'],mean):raise ValueError('Wrong summary')
        report['arms'][mode]=dict(n=32,correct=correct,mean_target_probability=mean,
            by_target={str(t):sum(int(np.argmax(r['probabilities'])==t) for r in rs if r['target']==t) for t in (0,1)},
            recoveries_vs_direct=0,regressions_vs_direct=0)
        for r in rs:
            direct=lookup[(r['id'],r['target'],'direct')]
            was=np.argmax(direct['probabilities'])==r['target'];now=np.argmax(r['probabilities'])==r['target']
            report['arms'][mode]['recoveries_vs_direct']+=int(now and not was)
            report['arms'][mode]['regressions_vs_direct']+=int(was and not now)
    for mode in ('base','published_null'):
        rs=[r for r in rows if r['kind']==mode]
        report[mode]=dict(n=len(rs),choices_A=sum(int(np.argmax(r['probabilities'])==0) for r in rs),
                          mean_probability_A=float(np.mean([r['probabilities'][0] for r in rs])))
    with a.out.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
