"""Streaming task/model/harness overlap audit before selecting an experiment."""
import argparse,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
import pyarrow.parquet as pq


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError('fresh output required')
    records=[];groups=defaultdict(lambda:defaultdict(list));seen=set()
    for batch in pq.ParquetFile(a.data).iter_batches(batch_size=1):
        r=batch.to_pylist()[0]
        if r['trace_id'] in seen:raise ValueError('duplicate ID')
        seen.add(r['trace_id']);t=json.loads(r['trace'])
        task=t.get('task',{});task_id=task.get('id') if isinstance(task,dict) else None
        task_source='task.id'
        if not task_id:task_id=t.get('task_name');task_source='task_name'
        # An unexplained top-level id might identify a run, not a task. Keep it
        # unresolved rather than silently claiming task matching.
        key=(r['benchmark'],str(task_id),r['model'],r['harness']) if task_id else None
        files=t.get('files') or []
        has_payload=bool(t.get('events') or t.get('messages') or any(isinstance(f,dict) and f.get('content') for f in files))
        row={'id':r['trace_id'],'benchmark':r['benchmark'],'model':r['model'],'harness':r['harness'],
            'label':r['label'],'provisional_task_id':task_id,'task_id_field':task_source if task_id else None,
            'has_standard_payload':has_payload,'trace_sha256':hashlib.sha256(r['trace'].encode()).hexdigest()}
        records.append(row)
        if key and has_payload:groups[key][r['label']].append(r['trace_id'])
    cells=[]
    for key,v in sorted(groups.items()):
        positives=v.get('cheating',[])+v.get('attempt',[]);negatives=v.get('benign',[])
        if positives and negatives:cells.append({'key':key,'positive_ids':positives,'negative_ids':negatives})
    with a.data.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    summary={'rows':len(records),'labels':dict(Counter(r['label'] for r in records)),
        'unknown_task_ids':sum(not r['provisional_task_id'] for r in records),
        'no_standard_payload':dict(Counter(r['label'] for r in records if not r['has_standard_payload'])),
        'overlap_cells':len(cells),'overlap_positives':sum(len(c['positive_ids']) for c in cells),
        'overlap_negatives':sum(len(c['negative_ids']) for c in cells),
        'maximum_disjoint_pairs':sum(min(len(c['positive_ids']),len(c['negative_ids'])) for c in cells)}
    report={'data_sha256':sha,'summary':summary,'overlap_cells':cells,'records':records,
        'scope':'Exact provisional task/model/harness overlap. No semantic task independence, label replay, or monitor scores.'}
    with a.out.open('x',encoding='utf8') as f:json.dump(report,f,indent=2)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
