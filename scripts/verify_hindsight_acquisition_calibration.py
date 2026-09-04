"""Read-only final acquisition audit; not a backward/optimizer replay."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from interaction_sprint.hindsight_acquisition_calibration import dataset


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();root=a.root
    manifest=json.loads((root/'MANIFEST.json').read_text())
    for name,digest in manifest.items():
        f=(root/name).resolve()
        if f.parent!=root.resolve() or hashlib.sha256(f.read_bytes()).hexdigest()!=digest:raise ValueError('Manifest mismatch')
    cases,schedule=dataset();byid={c['id']:c for c in cases}
    if json.loads((root/'cases.json').read_text())!=cases or json.loads((root/'schedule.json').read_text())!=schedule:raise ValueError('Design mismatch')
    runtime=json.loads((root/'runtime.json').read_text())
    if not 0<=runtime['batch_single_max_logit_difference']<=.001:raise ValueError('Batch equivalence failed')
    initial=torch.load(root/'initial_adapter.pt',weights_only=True,map_location='cpu')
    state_changes={}
    for step in (32,64,96):
        state=torch.load(root/f'adapter_{step}.pt',weights_only=True,map_location='cpu')
        if state.keys()!=initial.keys() or any(state[k].shape!=initial[k].shape or not torch.isfinite(state[k]).all() for k in initial):raise ValueError('Invalid adapter')
        state_changes[str(step)]=float(sum((state[k].double()-initial[k].double()).square().sum() for k in initial).sqrt())
        opt=torch.load(root/f'optimizer_{step}.pt',weights_only=True,map_location='cpu')
        if not opt['state'] or any(int(s['step'])!=step for s in opt['state'].values()):raise ValueError('Optimizer step mismatch')
        for s in opt['state'].values():
            if any(not torch.isfinite(t).all() for t in s.values() if isinstance(t,torch.Tensor)):raise ValueError('Nonfinite optimizer')
    steps=json.loads((root/'steps.json').read_text())
    if [s['step'] for s in steps]!=list(range(1,97)) or any(not np.isfinite(s['loss']) or not np.isfinite(s['gradient_norm']) for s in steps):raise ValueError('Incomplete steps')
    report=dict(status='MANIFEST_DESIGN_OUTCOMES_AND_STATES_VERIFIED_NOT_NEURAL_REPLAY',
        manifest_files=len(manifest),batch_single_max_logit_difference=runtime['batch_single_max_logit_difference'],
        adapter_L2_change=state_changes,metrics={})
    for phase in ('baseline','final'):
        rows=json.loads((root/f'{phase}.json').read_text())
        if len(rows)!=96 or {r['id'] for r in rows}!=set(byid):raise ValueError('Wrong outcome coverage')
        tokens={z:{r['target_token'] for r in rows if r['target']==z} for z in (0,1)}
        if any(len(t)!=1 for t in tokens.values()) or tokens[0]==tokens[1]:raise ValueError('Invalid answer labels')
        for r in rows:
            c=byid[r['id']]
            if any(r[k]!=c[k] for k in ('domain','split','target')):raise ValueError('Row metadata mismatch')
            prob=np.array(r['probabilities'])
            if prob.shape!=(2,) or not np.isfinite(prob).all() or (prob<=0).any() or not np.isclose(prob.sum(),1):raise ValueError('Probability mismatch')
            if not 0<r['AB_mass']<=1+1e-10:raise ValueError('Mass mismatch')
            if not np.isclose(prob[r['target']]*r['AB_mass'],r['target_probability'],atol=1e-12):raise ValueError('Full probability mismatch')
            if not np.isclose(-np.log(r['target_probability']),r['nll'],atol=1e-10):raise ValueError('NLL mismatch')
        report['metrics'][phase]={}
        for split in ('train','eval'):
            rs=[r for r in rows if r['split']==split]
            report['metrics'][phase][split]=dict(n=len(rs),correct=sum(int(r['full_argmax']==r['target_token']) for r in rs),
                nll=float(np.mean([r['nll'] for r in rs])),mean_target_probability=float(np.mean([r['target_probability'] for r in rs])),
                min_AB_mass=min(r['AB_mass'] for r in rs),
                domains={d:dict(n=sum(r['domain']==d for r in rs),correct=sum(int(r['full_argmax']==r['target_token']) for r in rs if r['domain']==d)) for d in sorted({r['domain'] for r in rs})})
    final=report['metrics']['final']['eval']
    qualified=final['correct']/final['n']>=.9 and final['min_AB_mass']>=.95 and all(d['correct']/d['n']>=.75 for d in final['domains'].values())
    report['decision']='ACQUISITION_QUALIFIED' if qualified else 'ACQUISITION_NOT_QUALIFIED'
    result=json.loads((root/'RESULT.json').read_text())
    if result['status']!=report['decision'] or result['counts']!=dict(forward_batches=291,forward_examples=964,backwards=96,updates=96):raise ValueError('Decision or count mismatch')
    for split in ('train','eval'):
        expected=report['metrics']['final'][split]
        if any(result['metrics'][split][k]!=expected[k] for k in ('n','correct','domains')) or not np.isclose(result['metrics'][split]['nll'],expected['nll']):raise ValueError('Reported arithmetic mismatch')
    report['counts']=result['counts'];report['elapsed_seconds']=result['elapsed']
    with a.out.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
