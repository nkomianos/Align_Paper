"""Prepare a schema-defined observer view; excludes outcome/audit metadata.

Not a claim to include every source file or to verify labels. No selection uses
gold localization, labels, model scores or trace length.
"""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq
from tokenizers import Tokenizer

FIELDS={'role','content','thought','tool_calls','tool_call_ids','tool_call_id','function_call',
        'name','action','message_type','kind','command','return_code','stderr','stdout','type',
        'prompt','response','text','msg','obs','tools','src','step'}
IGNORED={'id','agent','turn'}


def project(trace):
    field='messages' if isinstance(trace.get('messages'),list) and trace['messages'] else 'events'
    source=trace.get(field)
    if not isinstance(source,list) or not source:
        return None,{'status':'NO_CONVERSATION_EVENTS','field':field}
    unknown=set();view=[]
    for event in source:
        if not isinstance(event,dict):return None,{'status':'UNSUPPORTED_EVENT_TYPE','field':field}
        extra=set(event)-FIELDS-IGNORED
        if event.get('cache_control')=={'type':'ephemeral'}:extra.discard('cache_control')
        unknown.update(extra)
        kept={k:v for k,v in event.items() if k in FIELDS}
        if not kept:return None,{'status':'UNSUPPORTED_EVENT_SCHEMA','field':field,'unknown_fields':sorted(unknown)}
        view.append(kept)
    if unknown:return None,{'status':'UNSUPPORTED_EVENT_SCHEMA','field':field,'unknown_fields':sorted(unknown)}
    return view,{'status':'EVENT_VIEW_PREPARED','field':field,'events':len(view)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);tokenizer=Tokenizer.from_file(str(a.tokenizer))
    records=[]
    for row in pq.read_table(a.data).to_pylist():
        trace=json.loads(row['trace']);view,result=project(trace)
        result.update({'id':row['trace_id'],'benchmark':row['benchmark'],'label':row['label'],
            'source_trace_sha256':hashlib.sha256(row['trace'].encode()).hexdigest()})
        if view is not None:
            encoded=json.dumps(view,ensure_ascii=False,separators=(',',':'))
            raw=encoded.encode('utf8');path=a.out/(row['trace_id']+'.json');path.write_bytes(raw)
            result.update({'file':path.name,'sha256':hashlib.sha256(raw).hexdigest(),
                'tokens_without_prompt_overhead':len(tokenizer.encode(encoded,add_special_tokens=False).ids),
                'characters':len(encoded)})
        records.append(result)
    with a.data.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    report={'data_sha256':sha,'tokenizer_sha256':hashlib.sha256(a.tokenizer.read_bytes()).hexdigest(),
        'records':records,'status_counts':dict(Counter(r['status'] for r in records)),
        'scope':'Events/messages observer view only. No source files, outcome fields, labels or audit metadata included.',
        'task_context_verified':False,'labels_replayed':False,'monitor_ready':False,
        'truncation':False,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (a.out/'MANIFEST.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report['status_counts'],indent=2))


if __name__=='__main__':main()
