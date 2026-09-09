"""Audit overlap for nuisance-matched MALT monitoring, without reading traces."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def audit(rows):
    positive={'bypass_constraints','ignores_task_instructions'}
    rows=[dict(r,label=int(bool(set(r['labels']) & positive))) for r in rows
          if r['manually_reviewed'] and r['run_source']=='unprompted'
          and (set(r['labels']) & positive or r['labels']==['normal'])]
    if len({r['run_id'] for r in rows})!=len(rows):
        raise ValueError('duplicate run IDs')
    result={}
    for grouping in ('task_model','family_model'):
        groups=defaultdict(lambda: [[],[]])
        for r in rows:
            task=r['task_id'] if grouping=='task_model' else r['task_id'].split('/')[0]
            groups[(task,r['model'])][r['label']].append(r['run_id'])
        cells=[{'task':k[0],'model':k[1],'negative_ids':sorted(v[0]),'positive_ids':sorted(v[1])}
               for k,v in sorted(groups.items())]
        overlap=[c for c in cells if c['negative_ids'] and c['positive_ids']]
        result[grouping]={'cells':cells,'overlap_cells':len(overlap),
            'overlap_positives':sum(len(c['positive_ids']) for c in overlap),
            'overlap_negatives':sum(len(c['negative_ids']) for c in overlap),
            'maximum_disjoint_pairs':sum(min(len(c['positive_ids']),len(c['negative_ids'])) for c in overlap)}
    # A simultaneous family-and-model disjoint partition must keep every
    # connected component of this incidence graph intact.
    adjacency=defaultdict(set)
    for r in rows:
        f=('family',r['task_id'].split('/')[0]);m=('model',r['model'])
        adjacency[f].add(m);adjacency[m].add(f)
    remaining=set(adjacency);components=[]
    while remaining:
        todo=[min(remaining)];found=set()
        while todo:
            node=todo.pop()
            if node in found:continue
            found.add(node);todo.extend(adjacency[node]-found)
        remaining-=found
        selected=[r for r in rows if ('family',r['task_id'].split('/')[0]) in found]
        components.append({'families':sorted(n[1] for n in found if n[0]=='family'),
            'models':sorted(n[1] for n in found if n[0]=='model'),
            'runs':len(selected),'positives':sum(r['label'] for r in selected)})
    result['joint_disjoint_components']=components
    result['scope']='Overlap audit only; no matched sample selected, split frozen, or neural experiment run.'
    result['limitations']=['Exact model identifiers are not independently verified model lineages.',
        'Task-family prefixes are provisional.',
        'Matching observed metadata does not remove unobserved selection bias.',
        'Disjoint pairs within one task remain clustered by task.']
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();raw=a.metadata.read_bytes();report=audit(json.loads(raw))
    report['metadata_sha256']=hashlib.sha256(raw).hexdigest()
    with a.out.open('x',encoding='utf8') as f:json.dump(report,f,indent=2)
    print(json.dumps({k:({x:y for x,y in v.items() if x!='cells'} if isinstance(v,dict) else v)
                      for k,v in report.items()},indent=2))


if __name__=='__main__':main()
