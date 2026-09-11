"""Authenticate saved score trajectory and split semantic from interface NLL."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads((args.root/'MANIFEST.json').read_text())
    names=['baseline_student.json']+[f'{arm}_{step}_student.json' for arm in
           ['supervised','frozen_teacher','current_teacher'] for step in ['8','16','final']]
    result=[]
    ids=None
    for name in names:
        p=args.root/name
        assert hashlib.sha256(p.read_bytes()).hexdigest()==manifest[name]
        rows=json.loads(p.read_text())['rows']
        current={r['id'] for r in rows}
        assert len(rows)==len(current)==128
        assert len({r['base_id'] for r in rows})==32
        if ids is None:ids=current
        assert current==ids
        for base in {r['base_id'] for r in rows}:
            assert {r['label_rotation'] for r in rows if r['base_id']==base}=={0,1,2,3}
        semantic=[];interface=[];errors=[]
        for r in rows:
            semantic.append(-math.log(r['conditional_probability']))
            interface.append(-math.log(r['choice_mass']))
            errors.append(abs(r['nll']-semantic[-1]-interface[-1]))
            assert errors[-1]<2e-6
            assert abs(r['probability']-r['conditional_probability']*r['choice_mass'])<2e-6
        result.append({'file':name,'nll':sum(r['nll'] for r in rows)/128,
                       'conditional_nll':sum(semantic)/128,'interface_nll':sum(interface)/128,
                       'accuracy':sum(r['correct'] for r in rows)/128,
                       'max_identity_error':max(errors),'sha256':manifest[name]})
    baseline=result[0]
    for r in result[1:]:
        r['conditional_nll_gain']=baseline['conditional_nll']-r['conditional_nll']
        r['interface_nll_gain']=baseline['interface_nll']-r['interface_nll']
    report={'classification':'posthoc_saved_score_trajectory_audit','bases':32,'rotations_per_base':4,
            'rows':result,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope':'Per-row saved-score replay; no new neural forwards, raw-logit recomputation or checkpoint selection.'}
    with args.out.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
