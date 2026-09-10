"""Observer-view/source-label audit; no labels certified and no model input selection by gold spans."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import shlex
import pyarrow.parquet as pq
from run_unexplored_screens import sha,dump


def task_instruction(trace):
    found=[]
    for event in trace.get('events',[]):
        if not isinstance(event,dict) or event.get('type')!='agent_command':continue
        command=event.get('command')
        if not isinstance(command,str):continue
        try:words=shlex.split(command)
        except ValueError:continue
        if '-p' in words and words.index('-p')+1<len(words):found.append(words[words.index('-p')+1])
    unique=list(dict.fromkeys(found))
    return unique[0] if len(unique)==1 else None


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--views',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=False)
    manifest=json.loads((a.views/'MANIFEST.json').read_text());assert sha(a.data)==manifest['data_sha256']
    views={r['id']:r for r in manifest['records']};records=[];groups=defaultdict(list);review=[]
    for row in pq.read_table(a.data).to_pylist():
        trace=json.loads(row['trace']);view=views[row['trace_id']]
        assert sha_bytes(row['trace'].encode())==view['source_trace_sha256']
        instruction=task_instruction(trace);task=trace.get('task_name')
        record={'id':row['trace_id'],'benchmark':row['benchmark'],'task':task,'model':row['model'],'harness':row['harness'],
            'released_label':row['label'],'released_type':row['cheating_type'],'released_mechanism':row['cheating_mechanism'],
            'label_source':row['label_source'],'annotator_id':row['annotator_id'],'loc_evidence_source':row['loc_evidence_source'],
            'observer_status':view['status'],'instruction_extracted':instruction is not None,
            'raw_tokens':view.get('tokens_without_prompt_overhead'),
            'label_semantics':'not independently certified; benign release includes unauditable traces'}
        if view['status']=='EVENT_VIEW_PREPARED':
            assert sha(a.views/view['file'])==view['sha256']
            # Gold location is used solely for AFTER-projection coverage audit.
            def strings(v):
                if isinstance(v,str):yield v
                elif isinstance(v,list):
                    for item in v:yield from strings(item)
                elif isinstance(v,dict):
                    for item in v.values():yield from strings(item)
            payload=json.loads((a.views/view['file']).read_text(encoding='utf8'))
            snippet=row['loc_snippet'];record['gold_snippet_retained']=any(snippet in s for s in strings(payload)) if snippet else None
        if task and instruction and view['status']=='EVENT_VIEW_PREPARED' and record['raw_tokens']<=30000:
            groups[(row['benchmark'],task,row['model'],row['harness'],instruction)].append(record)
        if row['label']!='benign' and row['cheating_type']=='task' and instruction:
            review.append({'id':row['trace_id'],'task':task,'instruction':instruction,'released_label':row['label'],
                'rationale':row['rationale'],'snippet':row['loc_snippet'],'evidence_source':row['loc_evidence_source'],
                'status':'UNREVIEWED; read complete trace before any independent label'})
        records.append(record)
    cells=[]
    for (benchmark,task,model,harness,instruction),rows in groups.items():
        positive=[r['id'] for r in rows if r['released_label'] in ('cheating','attempt')];negative=[r['id'] for r in rows if r['released_label']=='benign']
        if positive and negative:cells.append({'benchmark':benchmark,'task':task,'model':model,'harness':harness,
            'instruction_sha256':sha_bytes(instruction.encode()),'positives':positive,'benign':negative})
    summary={'rows':len(records),'observer_status':dict(Counter(r['observer_status'] for r in records)),
        'explicit_instructions':sum(r['instruction_extracted'] for r in records),
        'exact_instruction_matched_short_cells':len(cells),'distinct_provisional_tasks':len({(c['benchmark'],c['task']) for c in cells}),
        'matched_positives':sum(len(c['positives']) for c in cells),'matched_benign':sum(len(c['benign']) for c in cells),
        'positive_evidence_sources':dict(Counter(str(r['loc_evidence_source']) for r in records if r['released_label']!='benign')),
        'neural_admitted':False,'reason':'Released labels are not independently validated safe/violation controls; manual complete-trace and policy review outstanding.'}
    dump(a.out/'AUDIT.json',{'source_sha256':sha(a.data),'projection_manifest_sha256':sha(a.views/'MANIFEST.json'),
        'script_sha256':sha(Path(__file__)),'records':records,'cells':cells,'summary':summary})
    dump(a.out/'TASK_POSITIVE_REVIEW_QUEUE.json',review)
    print(json.dumps(summary,indent=2))


def sha_bytes(value):
    import hashlib
    return hashlib.sha256(value).hexdigest()


if __name__=='__main__':main()
