"""Paired ID-only audit using the original prompts and inference implementation."""
import argparse
import copy
import json
from pathlib import Path
import re
import sys
import run_benign_memory_extraction as base
from run_unexplored_screens import dump,sha

original_inputs=base.inputs

def inputs():
    result=[]
    for old in original_inputs()[::2]:
        for renamed in (False,True):
            row=copy.deepcopy(old);n=len(row['values'])
            mapping=list(reversed(range(n))) if renamed else list(range(n))
            values=[0]*n
            for i,v in enumerate(row['values']):values[mapping[i]]=v
            row.update(id=f"{old['base']}_{int(renamed)}",presentation=int(renamed),
                       values=values,edges=[[mapping[a],mapping[b]] for a,b in row['edges']],
                       text=re.sub(r'(?i)(event )(\d+)',lambda m:m[1]+str(mapping[int(m[2])]),row['text']))
            assert base.solve(row['values'],row['edges'],row['threshold'])==old['gold']
            row['id_mapping']=mapping;result.append(row)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','score'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--snapshot',type=Path)
    p.add_argument('--report',type=Path);a=p.parse_args()
    if a.mode=='prepare':
        a.out.parent.mkdir(parents=True,exist_ok=True);dump(a.out,inputs());return
    base.inputs=inputs
    if a.mode=='run':
        if a.snapshot is None:p.error('snapshot required')
        sys.argv=[sys.argv[0],'--out',str(a.out),'--snapshot',str(a.snapshot)]
        base.main()
        dump(a.out/'RENAMING_PROTOCOL.json',dict(wrapper_sha256=sha(Path(__file__)),
             intervention='Reverse event-ID assignment only, preserving record order and wording',
             units='24 old exposed bases, original presentation0, identity/reversed IDs',
             classification='Posthoc paired attribution audit, not fresh confirmation',
             decision='Report both arms and changes; no training or repair queue'))
        dump(a.out/'MANIFEST.json',{f.name:sha(f) for f in a.out.iterdir() if f.is_file() and f.name!='MANIFEST.json'})
    else:
        if a.report is None:p.error('report required')
        import verify_benign_memory_extraction as verify
        verify.inputs=inputs
        sys.argv=[sys.argv[0],'--run',str(a.out),'--out',str(a.report)]
        verify.main()

if __name__=='__main__':main()
