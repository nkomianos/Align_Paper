"""Independently reconstruct every view from pinned downloaded raw traces."""
import argparse
from collections import Counter
import json
from pathlib import Path
from run_unexplored_screens import sha,dump


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--views',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    download=json.loads((a.source/'DOWNLOAD.json').read_text())
    assert sha(a.source/'SELECTION.json')==download['selection_sha256']
    for r in download['files']:assert sha(a.source/r['file'])==r['sha256']
    selection=json.loads((a.source/'SELECTION.json').read_text())['rows']
    inputs=json.loads((a.views/'INPUTS.json').read_text());lookup={r['id']:r for r in inputs}
    assert len(inputs)==len(lookup)==len(selection)*4
    observed_keys=Counter();user_counts=Counter();records=[]
    for i,selected in enumerate(selection):
        instruction=(a.source/f'{i:02d}_instruction.md').read_text(encoding='utf8')
        for cohort in ('hack','baseline'):
            raw=json.loads((a.source/f'{i:02d}_{cohort}.json').read_text(encoding='utf8'))
            user_counts[sum(s['source']=='user' for s in raw['steps'])]+=1
            agents=[s for s in raw['steps'] if s['source']=='agent']
            assert all(s['source'] in ('user','agent') for s in raw['steps'])
            for mode in ('full','stripped'):
                row=lookup[f'{i:02d}_{cohort}_{mode}'];prefix='TASK INSTRUCTION\n'+instruction+'\nOBSERVED TRAJECTORY\n'
                assert row['input'].startswith(prefix)
                payload=json.loads(row['input'][len(prefix):]);assert len(payload)==len(agents)
                assert row['task']==selected['task'] and row['split']==selected['split']
                for source,view in zip(agents,payload):
                    expected={k:source[k] for k in ('step_id','tool_calls','observation') if k in source}
                    if mode=='full' and source.get('message'):expected['message']=source['message']
                    assert view==expected
                    observed_keys.update(view.keys())
                records.append({'id':row['id'],'sha256':sha_text(row['input']),'agent_steps':len(agents)})
    report={'verified':True,'views':len(inputs),'tasks':len(selection),'user_step_count_distribution':dict(user_counts),
        'top_level_view_key_counts':dict(observed_keys),'records':records,
        'scope':'Exact reconstruction and unchanged action/output objects, not a semantic label audit',
        'source_receipt_sha256':sha(a.source/'DOWNLOAD.json'),'input_sha256':sha(a.views/'INPUTS.json')}
    dump(a.out,report);print(json.dumps({k:v for k,v in report.items() if k!='records'}))


def sha_text(value):
    import hashlib
    return hashlib.sha256(value.encode()).hexdigest()


if __name__=='__main__':main()
