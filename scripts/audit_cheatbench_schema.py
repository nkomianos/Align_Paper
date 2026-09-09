"""Inventory heterogeneous trace fields without treating audit metadata as behavior."""
import argparse
from collections import Counter,defaultdict
import hashlib,json
from pathlib import Path
import pyarrow.parquet as pq


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError('fresh report required')
    records=[];summary=defaultdict(Counter)
    for r in pq.read_table(a.data).to_pylist():
        t=json.loads(r['trace']);extra=t.get('extra');events=t.get('events')
        extra_keys=sorted(extra) if isinstance(extra,dict) else []
        prior_monitor=[k for k in extra_keys if k.startswith('monitor_')]
        item={'trace_id':r['trace_id'],'benchmark':r['benchmark'],'label':r['label'],
            'trace_sha256':hashlib.sha256(r['trace'].encode()).hexdigest(),
            'top_keys':sorted(t),'extra_keys':extra_keys,'prior_monitor_extra_keys':prior_monitor,
            'event_count':len(events) if isinstance(events,list) else None,
            'event_schema_counts':dict(Counter(','.join(sorted(e)) if isinstance(e,dict) else type(e).__name__ for e in events)) if isinstance(events,list) else {},
            'task_present':bool(t.get('task'))}
        records.append(item);s=summary[r['benchmark']];s['rows']+=1
        s['prior_monitor_extra']+=bool(prior_monitor)
        s['empty_or_missing_events']+=not bool(events)
        s['missing_task_object']+=not item['task_present']
    with a.data.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    report={'data_sha256':sha,'records':records,'by_benchmark':dict(summary),
        'scope':'Field inventory only. Missing standardized events/task does not imply absent content in alternate fields.',
        'monitor_inputs_qualified':False,'gpu_used':False}
    with a.out.open('x',encoding='utf8') as f:json.dump(report,f,indent=2)
    print(json.dumps(report['by_benchmark'],indent=2))


if __name__=='__main__':main()
