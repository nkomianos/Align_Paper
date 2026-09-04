"""Download pinned public research records; never execute trace content."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import requests

REV='b8b06a65a09e39a4fe1682ef56f48fd14ab74800'


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False); receipts=[]
    for name in ['README.md','dataset-manifest.json','train.jsonl']:
        url=f'https://huggingface.co/datasets/dreadnode/scopejudge/resolve/{REV}/{name}'
        r=requests.get(url,timeout=60);r.raise_for_status()
        if len(r.content)>30_000_000: raise ValueError('Unexpected file size')
        with (a.out/name).open('xb') as f:f.write(r.content)
        receipts.append(dict(path=name,bytes=len(r.content),sha256=hashlib.sha256(r.content).hexdigest()))
    records=[json.loads(line) for line in (a.out/'train.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    keys=collections.Counter();sources=collections.Counter();labels=collections.Counter()
    step_keys=collections.Counter();call_keys=collections.Counter();label_keys=collections.Counter()
    tool_steps=0;multi=0;unmatched=0;total_calls=0
    for row in records:
        keys.update(row.keys()); calls=set()
        for step in row['steps']:
            step_keys.update(step.keys());sources.update([step.get('source')])
            tc=step.get('tool_calls') or []
            tool_steps+=bool(tc);multi+=len(tc)>1;total_calls+=len(tc)
            for call in tc:
                call_keys.update(call.keys())
                calls.add((step.get('step_id'),call.get('tool_call_id')))
        for lab in row['extra']['scopejudge']['labels']:
            label_keys.update(lab.keys());labels.update([lab['golden_label']])
            unmatched+=(lab['step_id'],lab['tool_call_id']) not in calls
    report=dict(revision=REV,receipts=receipts,trajectories=len(records),root_keys=dict(keys),
                step_keys=dict(step_keys),call_keys=dict(call_keys),label_keys=dict(label_keys),
                sources=dict(sources),labels=dict(labels),tool_steps=tool_steps,
                multicall_steps=multi,tool_calls=total_calls,unmatched_labels=unmatched,
                scope='Schema only; no classifier, no tool execution, no exposure of trace text')
    with (a.out/'schema_audit.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
