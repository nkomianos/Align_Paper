"""Independent saved-output replay of the benign temporal extraction gate."""
import argparse
import itertools
import json
from pathlib import Path
from run_benign_memory_extraction import inputs,parse,solve
from run_unexplored_screens import dump,sha


def orders(values,edges):
    return {p for p in itertools.permutations(range(len(values))) if all(p.index(a)<p.index(b) for a,b in edges)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    for name,h in json.loads((a.run/'MANIFEST.json').read_text()).items():assert Path(name).name==name and sha(a.run/name)==h
    rows=json.loads((a.run/'INPUTS.json').read_text());assert rows==inputs();byid={r['id']:r for r in rows}
    outputs=[json.loads(x) for x in (a.run/'OUTPUTS.jsonl').read_text().splitlines()]
    keys=[(o['id'],o['arm']) for o in outputs];assert len(keys)==len(set(keys))==96
    assert set(keys)=={(r['id'],arm) for r in rows for arm in ['direct','extract']}
    records=[]
    for o in outputs:
        r=byid[o['id']];v=parse(o['text']);assert v==o['parsed'];answer=None;semantic=False;valid=False
        if o['arm']=='direct':
            valid=isinstance(v,dict) and set(v)=={'answer'} and v['answer'] in ['YES','NO','CLARIFY']
            if valid:answer=v['answer']
        elif isinstance(v,dict) and set(v)=={'values','edges'} and isinstance(v['values'],list) and isinstance(v['edges'],list):
            try:
                answer=solve(v['values'],v['edges'],r['threshold']);valid=True
                semantic=v['values']==r['values'] and orders(v['values'],v['edges'])==orders(r['values'],r['edges'])
            except (ValueError,TypeError,IndexError):pass
        records.append({'id':o['id'],'base':r['base'],'mechanism':r['mechanism'],'arm':o['arm'],'valid':valid,'eos':o['eos'],'answer':answer,'correct':answer==r['gold'],'semantic_graph_correct':semantic})
    metrics={}
    for arm in ['direct','extract']:
        selected=[r for r in records if r['arm']==arm]
        metrics[arm]={k:sum(r[k] for r in selected)/len(selected) for k in ['valid','eos','correct','semantic_graph_correct']}
    qualified=all(metrics[arm][k]>=.95 for arm in metrics for k in ['valid','eos'])
    signal=metrics['extract']['semantic_graph_correct']>=.95 and metrics['extract']['correct']-metrics['direct']['correct']>=.10
    report={'classification':'DEVELOPMENTAL_SYNTHETIC_NEURAL_EXTRACTION','route':('INVALID_INTERFACE' if not qualified else 'FRESH_BENIGN_DATA_REQUIRED' if signal else 'STOP_NO_DECISIVE_EXTRACTION_ADVANTAGE'),
            'metrics':metrics,'records':records,'manifest_sha256':sha(a.run/'MANIFEST.json'),
            'scope':'Four synthetic mechanisms with parameter and presentation repeats; no independent-task confidence interval, natural-memory, learned update, or novelty claim.'}
    dump(a.out,report);print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))


if __name__=='__main__':main()
