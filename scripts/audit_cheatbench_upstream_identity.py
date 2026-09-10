"""Compare released observer events to their pinned original trajectory rows."""
import argparse
from collections import defaultdict,Counter
import hashlib
import json
from pathlib import Path
import re
import pyarrow.parquet as pq
from run_unexplored_screens import sha,dump


def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def canonical_steps(steps):return [{k:v for k,v in step.items() if v is not None and k!='step'} for step in steps]


def main():
    p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    receipt=json.loads((a.source/'DOWNLOAD.json').read_text())
    for row in receipt['files']:assert sha(a.source/row['path'])==row['sha256']
    cohort=[];names=set()
    for row in pq.read_table(a.cohort).to_pylist():
        trace=json.loads(row['trace'])
        if trace.get('source_dataset')=='yoonholee/terminalbench-trajectories':
            cohort.append((row,trace));names.add(trace['trial_name'])
    originals=defaultdict(list);invalid_original_payloads=Counter()
    for file in a.source.glob('*.parquet'):
        for batch in pq.ParquetFile(file).iter_batches(batch_size=128):
            for row in batch.to_pylist():
                if row['trial_name'] not in names:continue
                steps=json.loads(row['steps']) if row['steps'] else []
                if not isinstance(steps,list) or any(not isinstance(step,dict) for step in steps):
                    invalid_original_payloads[type(steps).__name__]+=1
                    continue
                row['canonical_hash']=fingerprint(canonical_steps(steps))
                row['steps_sha256']=hashlib.sha256((row['steps'] or '').encode()).hexdigest()
                row['n_steps']=len(steps);row['shard']=file.name
                row['unresolved_user']=any(x.get('src')=='user' and isinstance(x.get('msg'),str) and re.fullmatch(r'\$\d+',x['msg']) for x in steps)
                originals[row['trial_name']].append(row)
    records=[];by_trial=defaultdict(list)
    for row,trace in cohort:
        exact=[r for r in originals[trace['trial_name']] if r['canonical_hash']==fingerprint(canonical_steps(trace['events']))]
        candidates=[r for r in originals[trace['trial_name']] if r['trial_id'] and r['model']==trace['model'] and not r['unresolved_user']]
        record={'id':row['trace_id'],'released_label':row['label'],'trial_name':trace['trial_name'],
            'released_harness':row['harness'],'released_model':row['model'],
            'exact_source_matches':len(exact),'source_agents':sorted({r['agent'] for r in exact}),
            'source_models':sorted({r['model'] for r in exact}),
            'source_unresolved_user':any(r['unresolved_user'] for r in exact),
            'source_rows':[{'shard':r['shard'],'steps_sha256':r['steps_sha256'],'trial_id':r['trial_id']} for r in exact],
            'complete_original_candidates':[{'trial_id':r['trial_id'],'steps_sha256':r['steps_sha256'],'agent':r['agent'],
                'model':r['model'],'shard':r['shard'],'n_steps':r['n_steps']} for r in candidates],
            'scope':'Candidate recovery is not automatic equivalence; full behavioral differences require review.'}
        records.append(record);by_trial[(trace['trial_name'],trace['model'])].append(record)
    repeats=[{'trial_name':key[0],'model':key[1],'ids':[r['id'] for r in rows],
        'labels':[r['released_label'] for r in rows],'event_identity_proven':False}
        for key,rows in by_trial.items() if len(rows)>1]
    summary={'cohort_rows':len(cohort),'unique_trial_name_model_keys':len(by_trial),
        'matching_trial_original_rows_with_invalid_step_payload':dict(invalid_original_payloads),
        'rows_with_exact_original_match':sum(r['exact_source_matches']>0 for r in records),
        'repeated_trial_keys':len(repeats),'repeated_trial_keys_with_different_labels':sum(len(set(r['labels']))>1 for r in repeats),
        'rows_with_unresolved_user_in_original':sum(r['source_unresolved_user'] for r in records),
        'unresolved_rows_with_one_complete_candidate':sum(r['source_unresolved_user'] and len(r['complete_original_candidates'])==1 for r in records),
        'released_harness_original_agent_pairs':dict(Counter(r['released_harness']+' / '+','.join(r['source_agents']) for r in records)),
        'neural_admitted':False}
    dump(a.out,{'summary':summary,'records':records,'repeated_trial_keys':repeats,
        'cohort_sha256':sha(a.cohort),'original_download_receipt_sha256':sha(a.source/'DOWNLOAD.json'),
        'script_sha256':sha(Path(__file__)),'scope':'Source reanalysis; no duplicate removal, label correction or neural result.'})
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
