"""Read-only video screen integrity, causality, selection and metric replay."""
import argparse
import json
import re
from pathlib import Path
from run_unexplored_screens import sha


def verify(root):
    for name,h in json.loads((root/'MANIFEST.json').read_text()).items():
        path=root/name;assert path.resolve().is_relative_to(root.resolve());assert sha(path)==h,name
    inputs=json.loads((root/'INPUTS.json').read_text())
    assert len(inputs)==24 and len({r['source_video'] for r in inputs})==24
    indexed={r['id']:r for r in inputs};views=json.loads((root/'VIEWS.json').read_text())
    records=[json.loads(x) for x in (root/'OUTPUTS.jsonl').read_text().splitlines()]
    seen=set();metrics={};correct={};parsed=0;censored=0
    for r in records:
        source=indexed[r['id']];v=views[r['id']];key=(r['id'],r['mode']);assert key not in seen;seen.add(key)
        assert r['target']==source['target']
        assert len(v['timestamps'])==len(v['requested'])==64
        assert all(0<=t<=req+1e-8 and req<=source['query_time']+1e-8 for t,req in zip(v['timestamps'],v['requested']))
        assert v['timestamps']==sorted(v['timestamps'])
        mode=r['mode']
        if mode in ('snapshot','blank'): expected=[63]
        elif mode=='uniform16':expected=[round(i*63/15) for i in range(16)]
        elif mode=='dense64':expected=list(range(64))
        elif mode=='change16':expected=sorted(i for block in range(8) for i in sorted(range(8*block,8*(block+1)),key=lambda j:(-v['changes'][j],j))[:2])
        else:raise AssertionError(mode)
        assert r['indices']==expected
        paths=[root/r['id']/'blank.png'] if mode=='blank' else [root/r['id']/f'{i:02d}.png' for i in expected]
        assert [sha(p) for p in paths]==r['image_sha256']
        match=re.fullmatch(r'\s*(\d+)\s*',r['text']);pred=int(match[1]) if match else None
        assert pred==r['prediction'];assert r['correct']==(pred==source['target'])
        parsed+=pred is not None;censored+=r['censored'];correct.setdefault(mode,[]).append(pred==source['target'])
    expected_keys={(r['id'],mode) for r in inputs for mode in (['snapshot','blank'] if r['category']=='O1-Snap' else ['uniform16','change16','dense64'])}
    assert seen==expected_keys and len(records)==60
    metrics={k:sum(v)/len(v) for k,v in correct.items()}
    saved=json.loads((root/'SUMMARY.json').read_text());assert saved['accuracy']==metrics
    assert saved['parse_coverage']==parsed/60 and saved['censored']==censored
    qualified=metrics['snapshot']>=.75 and metrics['dense64']>=.60 and parsed/60>=.95
    signal=(metrics['change16']-metrics['uniform16'])*12>=3-1e-8
    route='INVALID_PERCEPTION_OR_FORMAT' if not qualified else ('DEV_SIGNAL' if signal else 'STOP_NO_DECISIVE_SELECTION_GAIN')
    assert saved['route']==route
    return {'verified':True,'accuracy':metrics,'parse_coverage':parsed/60,'route':route,
            'scope':'saved evidence replay; no neural replay or independent human relabeling',
            'n_source_videos':24,'n_action_comparison_units':12}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    print(json.dumps(verify(a.root),indent=2))
