"""Posthoc saved-graph query-family audit; no neural calls or corrected outputs."""
import argparse
import json
from pathlib import Path
import memory_encoding_followup as m

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    for name,digest in json.loads((a.run/'MANIFEST.json').read_text()).items():
        assert Path(name).name==name and m.sha(a.run/name)==digest
    rows=json.loads((a.run/'INPUTS.json').read_text());assert rows==m.prepare()
    byid={r['id']:r for r in rows}; records=[]
    for line in (a.run/'OUTPUTS.jsonl').read_text().splitlines():
        o=json.loads(line)
        if o['arm']=='reasoning':continue
        r=byid[o['id']]
        item=dict(id=r['id'],base=r['base'],arm=o['arm'],valid=False,
                  original_correct=False,all_thresholds_correct=False)
        try:
            values,edges=m.decode(m.parse(o['text']),o['arm'],len(r['values']))
            # Integer values: transitions can occur only just above a value.
            thresholds=sorted({min(values+r['values'])-1} | {v+1 for v in values+r['values']})
            gold=[m.decision(r['values'],r['edges'],t) for t in thresholds]
            predicted=[m.decision(values,edges,t) for t in thresholds]
            item.update(valid=True,original_correct=m.decision(values,edges,r['threshold'])==r['gold'],
                        all_thresholds_correct=gold==predicted,thresholds=thresholds,
                        gold=gold,predicted=predicted)
            # Independent min/max of values at maximal events is sufficient
            # for every threshold query in this specific overwrite task.
            terminal=[values[i] for i in range(len(values)) if i not in {x for x,y in edges}]
            oracle=['YES' if min(terminal)>=t else 'NO' if max(terminal)<t else 'CLARIFY' for t in thresholds]
            assert oracle==predicted
        except (ValueError,TypeError):
            pass
        records.append(item)
    summary={arm:{
        'valid':sum(r['valid'] for r in records if r['arm']==arm),
        'all_thresholds_correct':sum(r['all_thresholds_correct'] for r in records if r['arm']==arm),
        'original_correct_but_reuse_fails':sum(r['original_correct'] and not r['all_thresholds_correct'] for r in records if r['arm']==arm)
    } for arm in m.ARMS[1:]}
    m.dump(a.out,dict(classification='POSTHOC_QUERY_FAMILY_AUDIT',summary=summary,records=records,
        input_manifest_sha256=m.sha(a.run/'MANIFEST.json'),
        scope='Saved-graph counterfactual scoring only; no causal query-conditioning effect, independent replication or novel theorem.'))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
