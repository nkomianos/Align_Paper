"""Recompute acquisition decisions and outcomes from saved probabilities; no neural replay."""
import argparse
import json
from pathlib import Path
import numpy as np
from research_pilots.common import check_manifest
from research_pilots.tabular_drift import validate,acquisition,METHODS,summarize,probabilities
from verify_research_pilot import close


def verify(root):
    check_manifest(root)
    read=lambda name:json.loads((root/name).read_text())
    data=validate(read('INPUTS.json'));provenance=read('PROVENANCE.json')
    cases=data['cases'][:1] if provenance['smoke'] else data['cases']
    count=1 if provenance['smoke'] else 4
    outputs=[]
    if len(list(root.glob('case_*.json')))!=len(cases):raise ValueError('case population differs')
    for i,c in enumerate(cases):
        o=read(f'case_{i:02d}.json');outputs.append(o)
        if o['id']!=c['id'] or set(o['arms'])!=set(METHODS):raise ValueError('case/arm mismatch')
        for method,records in o['arms'].items():
            if len(records)!=count:raise ValueError('round population differs')
            context=list(c['initial'])
            for step,r in enumerate(records):
                shortlist=c['shortlists'][step]
                if r['shortlist']!=shortlist or r['step']!=step:raise ValueError('shortlist mismatch')
                p=probabilities(r['candidate_p'])
                if p.shape!=(8,len(set(c['y']))):raise ValueError('candidate probability shape differs')
                if method in ('raw','information','projected'):
                    block=r['lookahead'];close(block['p'],r['candidate_p'])
                    scores=[acquisition(p[j],block['q'],block['qa'][j]) for j in range(8)]
                    close(scores,r['scores'])
                    index=max(range(8),key=lambda j:(scores[j][method],-j))
                elif method=='entropy':index=int(np.argmax(-np.sum(p*np.log(p),axis=1)))
                else:index=int(np.random.default_rng(c['seed']+1000+step).integers(8))
                if r['selected']!=shortlist[index]:raise ValueError('acquisition decision differs')
                context.append(shortlist[index])
                if r['context']!=context:raise ValueError('label history differs')
                ev=probabilities(r['evaluation_p'])
                if ev.shape!=(len(c['evaluation']),len(set(c['y']))):raise ValueError('evaluation shape differs')
    expected=summarize(cases,outputs,provenance['backend'] if not provenance['smoke'] else 'smoke')
    result=read('RESULT.json')
    close(expected,{k:result[k] for k in expected})
    return {'verified':True,'decision':result['decision'],'neural_replay':False,'paper_green_light':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    print(json.dumps(verify(p.parse_args().root),indent=2))
