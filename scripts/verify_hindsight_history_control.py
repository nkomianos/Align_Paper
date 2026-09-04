"""Read-only checksum, paired-design, replay and arithmetic audit."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from interaction_sprint.hindsight_revelation_probe import cases, history_text


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--previous-scores',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    manifest=json.loads((a.root/'MANIFEST.json').read_text())
    for name,digest in manifest.items():
        f=(a.root/name).resolve()
        if f.parent!=a.root.resolve() or hashlib.sha256(f.read_bytes()).hexdigest()!=digest:
            raise ValueError('Invalid manifest')
    oldraw=a.previous_scores.read_bytes()
    if hashlib.sha256(oldraw).hexdigest()!='8579416b911679aa688f4bae60f6d8b78ef2cd54fddfe7264cbd78e7048ccba4':
        raise ValueError('Previous scores mismatch')
    old={(r['id'],r['target']):r for r in json.loads(oldraw) if r['kind']=='truthful' and r['action']!=r['target']}
    rows=json.loads((a.root/'scores.json').read_text())
    lookup={(r['id'],r['target'],r['kind']):r for r in rows}
    if len(rows)!=96 or len(lookup)!=96:raise ValueError('Wrong or duplicated count')
    diffs=[]
    for c in cases():
        for target in (0,1):
            for mode in ('shown','omitted','redacted'):
                r=lookup[(c['id'],target,mode)]
                if r['action']!=1-target or r['prompt']!=history_text(c,target,mode):raise ValueError('Changed design')
                probs=np.asarray(r['probabilities'])
                if probs.shape!=(2,) or not np.isfinite(probs).all() or (probs<0).any() or not np.isclose(probs.sum(),1):raise ValueError('Invalid probabilities')
                if not 0<=r['AB_mass']<=1+1e-10:raise ValueError('Invalid probability mass')
                if mode=='shown':
                    prior=old[(c['id'],target)]
                    if r['token_ids']!=prior['token_ids']:raise ValueError('Replay token mismatch')
                    diffs.extend(abs(np.array(r['probabilities'])-np.array(prior['probabilities'])))
    result=dict(status='MANIFEST_DESIGN_ARITHMETIC_AND_32_FORWARD_REPLAY_AUDITED',
        forwards=96,updates=0,manifest_files=len(manifest),max_replay_probability_difference=float(max(diffs)),
        minimum_AB_mass=min(r['AB_mass'] for r in rows),arms={})
    reported=json.loads((a.root/'RESULT.json').read_text())
    for mode in ('shown','omitted','redacted'):
        rs=[r for r in rows if r['kind']==mode]
        correct=sum(int(np.argmax(r['probabilities'])==r['target']) for r in rs)
        mean=float(np.mean([r['probabilities'][r['target']] for r in rs]))
        if reported['arms'][mode]['correct']!=correct or not np.isclose(reported['arms'][mode]['mean_target_probability'],mean):
            raise ValueError('Summary mismatch')
        result['arms'][mode]=dict(n=32,correct=correct,mean_target_probability=mean,
            by_target={str(z):sum(int(np.argmax(r['probabilities'])==z) for r in rs if r['target']==z) for z in (0,1)},
            recoveries_vs_shown=0,regressions_vs_shown=0)
        for r in rs:
            shown=lookup[(r['id'],r['target'],'shown')]
            was=np.argmax(shown['probabilities'])==r['target'];now=np.argmax(r['probabilities'])==r['target']
            result['arms'][mode]['recoveries_vs_shown']+=int(now and not was)
            result['arms'][mode]['regressions_vs_shown']+=int(was and not now)
    with a.out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
