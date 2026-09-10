"""Verify native temporal patch timestamps, saved pixels and exact-count routing."""
import argparse
import json
from pathlib import Path
import re
from run_unexplored_screens import sha


def verify(root):
    out=root/'svc_native_v2';old=root/'svc_v1'
    for folder in (out,old):
        for name,h in json.loads((folder/'MANIFEST.json').read_text()).items():assert sha(folder/name)==h
    assert json.loads((out/'PROTOCOL.json').read_text())['parent_manifest_sha256']==sha(old/'MANIFEST.json')
    inputs={r['id']:r for r in json.loads((old/'INPUTS.json').read_text()) if r['category']=='E1-Action'}
    views=json.loads((old/'VIEWS.json').read_text())
    rows=[json.loads(x) for x in (out/'OUTPUTS.jsonl').read_text().splitlines()]
    assert len(rows)==12 and {r['id'] for r in rows}==set(inputs)
    correct=parsed=censored=0
    for r in rows:
        row=inputs[r['id']];assert row['target']==r['target'];assert r['timestamps']==views[r['id']]['timestamps']
        ticks=[round(t*1_000_000) for t in r['timestamps']]
        assert r['metadata']['frames_indices']==ticks and r['metadata']['fps']==1_000_000
        observed=[float(v) for v in re.findall(r'<([0-9.]+) seconds>',r['decoded_prompt'])]
        # Native processor converts each index to seconds before averaging.
        # Moving division after addition changes decimal half-tie rounding.
        expected=[float(f'{(a/1e6+b/1e6)/2:.1f}') for a,b in zip(ticks[::2],ticks[1::2])]
        assert observed==expected and len(observed)==32
        assert r['video_grid_thw'][0][0]==32
        assert r['image_sha256']==[sha(old/r['id']/f'{i:02d}.png') for i in range(64)]
        match=re.fullmatch(r'\s*(\d+)\s*',r['text']);pred=int(match[1]) if match else None
        assert pred==r['prediction'];assert (pred==row['target'])==r['correct']
        correct+=r['correct'];parsed+=pred is not None;censored+=r['censored']
    saved=json.loads((out/'SUMMARY.json').read_text())
    assert (correct,parsed,censored)==(saved['correct'],saved['parse_count'],saved['censored'])
    route='CAPABILITY_QUALIFIED_DEV_ONLY' if correct>=8 and parsed==12 and censored==0 else 'STOP_INVALID_CAPABILITY'
    assert route==saved['route']
    return {'verified':True,'correct':correct,'n':12,'parse_count':parsed,'route':route,
        'scope':'pixel identity, timestamp transport and saved output replay; not neural replay',
        'interpretation':'native dense64 capability check on sparse sampled evidence; no disproof of observation-aware video'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();print(json.dumps(verify(a.root),indent=2))
