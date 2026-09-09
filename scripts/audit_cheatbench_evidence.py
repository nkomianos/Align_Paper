"""Resolve released localization pointers and distinguish messages from empty records."""
import argparse,hashlib,json,re
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq


def resolve(obj,path):
    tokens=re.findall(r'[^.\[\]]+|\[\d+\]',path)
    if not tokens:raise ValueError('empty path')
    for token in tokens:
        obj=obj[int(token[1:-1])] if token.startswith('[') else obj[token]
    return obj


def locate_snippet(obj,snippet,path=''):
    """Audit-only relocation; annotation text must never guide monitor inputs."""
    found=[]
    if not snippet:return found
    if isinstance(obj,str):
        start=obj.find(snippet)
        while start>=0:
            found.append({'path':path,'char_start':start,'char_end':start+len(snippet)})
            start=obj.find(snippet,start+1)
    elif isinstance(obj,dict):
        for key,value in obj.items():found.extend(locate_snippet(value,snippet,f'{path}.{key}' if path else key))
    elif isinstance(obj,list):
        for i,value in enumerate(obj):found.extend(locate_snippet(value,snippet,f'{path}[{i}]'))
    return found


def inspect(row):
    trace=json.loads(row['trace']);events=trace.get('events');messages=trace.get('messages')
    files=trace.get('files') or []
    file_text=sum(len(f.get('content') or '') for f in files if isinstance(f,dict))
    no_behavior=not events and not messages and not file_text
    result={'trace_id':row['trace_id'],'label':row['label'],'benchmark':row['benchmark'],
        'trace_sha256':hashlib.sha256(row['trace'].encode()).hexdigest(),
        'events':len(events) if isinstance(events,list) else 0,
        'messages':len(messages) if isinstance(messages,list) else 0,
        'file_content_characters':file_text,'no_standard_behavior_payload':no_behavior,
        'loc_kind':row['loc_kind'],'localization_check':'not_available'}
    if row['loc_field_path'] and row['loc_line_start'] and row['loc_line_end']:
        try:
            text=resolve(trace,row['loc_field_path'])
            if not isinstance(text,str):raise ValueError('localization field is not text')
            lo=int(row['loc_line_start']);hi=int(row['loc_line_end'])
            if lo<1 or hi<lo or hi>len(text.splitlines()):raise ValueError('bad line range')
            extracted='\n'.join(text.splitlines()[lo-1:hi])
            result['localization_check']='exact_match' if extracted==row['loc_snippet'] else 'mismatch'
            if extracted!=row['loc_snippet']:
                result['equal_ignoring_terminal_line_endings']=extracted.rstrip('\r\n')==row['loc_snippet'].rstrip('\r\n')
                result['equal_ignoring_outer_whitespace']=extracted.strip()==row['loc_snippet'].strip()
        except (KeyError,IndexError,ValueError,TypeError) as e:
            result['localization_check']='unresolvable';result['localization_error']=type(e).__name__
    if result['localization_check'] in ('mismatch','unresolvable') and not result.get('equal_ignoring_outer_whitespace'):
        result['audit_only_snippet_matches']=locate_snippet(trace,(row['loc_snippet'] or '').strip())
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    rows=[inspect(r) for r in pq.read_table(a.data).to_pylist()]
    report={'rows':rows,'localization_counts':dict(Counter(r['localization_check'] for r in rows)),
        'no_behavior_by_label':dict(Counter(r['label'] for r in rows if r['no_standard_behavior_payload'])),
        'mismatch_terminal_line_endings_only':sum(r.get('equal_ignoring_terminal_line_endings',False) for r in rows),
        'mismatch_outer_whitespace_only':sum(r.get('equal_ignoring_outer_whitespace',False) for r in rows),
        'scope':'Structural source audit; exact span reproduction is not semantic label verification.',
        'monitor_ready':False}
    with a.out.open('x',encoding='utf8') as f:json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
