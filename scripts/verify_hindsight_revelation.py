"""Read-only receipt, design and arithmetic verification; no neural replay."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--recover-unsealed',action='store_true')
    a=p.parse_args()
    if a.recover_unsealed:
        failure=json.loads((a.root/'FAILED.json').read_text())
        if failure!={'error':'TypeError','message':'Object of type int64 is not JSON serializable'}:
            raise ValueError('Not the known summary serialization failure')
        manifest={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in a.root.iterdir() if f.is_file()}
    else:
        manifest=json.loads((a.root/'MANIFEST.json').read_text())
    for name,digest in manifest.items():
        target=(a.root/name).resolve()
        if target.parent!=a.root.resolve():raise ValueError('Unsafe manifest entry')
        if hashlib.sha256(target.read_bytes()).hexdigest()!=digest:raise ValueError('Checksum mismatch')
    rows=json.loads((a.root/'scores.json').read_text())
    if a.recover_unsealed and rows!=json.loads((a.root/'partial_scores.json').read_text()):
        raise ValueError('Saved and recovery scores differ')
    if Counter(r['kind'] for r in rows)!=dict(base=16,truthful=64,endorsement=32,null=32):raise ValueError('Wrong counts')
    keys=[(r['id'],r['kind'],r['action'],r['target']) for r in rows]
    if len(set(keys))!=144:raise ValueError('Duplicate key')
    for r in rows:
        probs=np.array(r['probabilities'])
        if probs.shape!=(2,) or not np.isfinite(probs).all() or (probs<0).any() or not np.isclose(probs.sum(),1):raise ValueError('Invalid probabilities')
        if not 0<=r['AB_mass']<=1+1e-10:raise ValueError('Invalid mass')
    truthful=[r for r in rows if r['kind']=='truthful']
    def correct(rs):return sum(int(np.argmax(r['probabilities'])==r['target']) for r in rs)
    if not a.recover_unsealed:
        original=json.loads((a.root/'RESULT.json').read_text())
        if original['truthful_correct']!=correct(truthful) or original['forwards']!=144 or original['updates']!=0:raise ValueError('Summary mismatch')
    opposite=[r for r in truthful if r['action']!=r['target']]
    domains={d:dict(correct=correct([r for r in truthful if r['domain']==d]),n=8) for d in sorted({r['domain'] for r in truthful})}
    lookup={(r['id'],r['kind'],r['action']):r for r in rows if r['kind'] in ('endorsement','null')}
    differences=[r['probabilities'][r['action']]-lookup[(r['id'],'null',r['action'])]['probabilities'][r['action']]
                 for r in rows if r['kind']=='endorsement']
    result=dict(status=('UNSEALED_RUN_RECOVERED_FROM_COMPLETE_SCORES_NOT_NEURAL_REPLAY' if a.recover_unsealed else 'CHECKSUM_DESIGN_AND_ARITHMETIC_VERIFIED_NOT_NEURAL_REPLAY'),
        evidence_sha256=manifest,
        manifest_files=len(manifest),truthful_correct=correct(truthful),truthful_n=64,
        opposite_previous_action_correct=correct(opposite),opposite_previous_action_n=len(opposite),
        domains=domains,endorsement_minus_null_previous_action_probability=float(np.mean(differences)),
        minimum_AB_mass=min(r['AB_mass'] for r in rows),paper_green_light=False)
    for kind in ('endorsement','null'):
        rs=[r for r in rows if r['kind']==kind]
        result[kind+'_mean_previous_action_probability']=float(np.mean([r['probabilities'][r['action']] for r in rs]))
    with a.out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
