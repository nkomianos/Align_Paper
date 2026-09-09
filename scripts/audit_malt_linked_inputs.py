"""Measure reconstruction limits of ordered MALT API samples without scoring labels."""
import argparse
import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq


def signature(message):
    fields={k:message.get(k) for k in ('role','content','name','function_call')}
    return hashlib.sha256(json.dumps(fields,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def inspect(row):
    nodes={};conflicts=set();unsigned_inputs=0;output_signatures=set();input_signatures=set()
    unmatched=0;multiple_outputs=0;max_input_chars=0
    for sample in row['samples']:
        unmatched+=bool(sample['metadata'].get('unmatched'))
        multiple_outputs+=len(sample['output'])>1
        max_input_chars=max(max_input_chars,sum(len(m.get('content') or '') for m in sample['input']))
        for message in sample['input']:
            sig=signature(message);input_signatures.add(sig)
            meta=message.get('metadata') or {};node=meta.get('node_id')
            if node is None:unsigned_inputs+=1;continue
            item={'signature':sig,'parent':meta.get('parent_node_id'),'branch':meta.get('branch_id'),
                  'timestamp':meta.get('timestamp'),'chars':len(message.get('content') or '')}
            if node in nodes and nodes[node]!=item:conflicts.add(node)
            nodes[node]=item
        for group in sample['output']:
            output_signatures.update(signature(message) for message in group)
    missing_parents={v['parent'] for v in nodes.values() if v['parent'] is not None and v['parent'] not in nodes}
    final=row['samples'][-1] if row['samples'] else None
    final_outputs=[signature(m) for group in final['output'] for m in group] if final else []
    return {'run_id':row['metadata']['run_id'],'task_id':row['metadata']['task_id'],
            'samples':len(row['samples']),'unique_input_nodes':len(nodes),'conflicting_node_ids':len(conflicts),
            'missing_parent_ids':len(missing_parents),'input_occurrences_without_node_id':unsigned_inputs,
            'samples_with_multiple_output_candidates':multiple_outputs,'unmatched_samples':unmatched,
            'unique_output_messages':len(output_signatures),
            'outputs_not_observed_as_input':len(output_signatures-input_signatures),
            'final_outputs_not_observed_as_input':sum(s not in input_signatures for s in final_outputs),
            'max_single_sample_input_characters':max_input_chars,
            'complete_executed_transcript_certified':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--metadata',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();metadata=json.loads(a.metadata.read_text(encoding='utf8'))
    positive={'bypass_constraints','ignores_task_instructions'}
    selected={r['run_id'] for r in metadata if r['manually_reviewed'] and r['run_source']=='unprompted'
              and (set(r['labels'])&positive or r['labels']==['normal'])}
    if a.out.exists():raise FileExistsError('fresh output required')
    a.out.mkdir(parents=True);reports=[];sources={}
    for path in sorted(a.data.glob('*.parquet')):
        digest=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
        sources[path.name]=digest.hexdigest()
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1):
            meta=batch.column('metadata')[0].as_py()
            if meta['run_id'] not in selected:continue
            result=inspect(batch.to_pylist()[0]);reports.append(result)
            (a.out/f'run_{result["run_id"]}.json').write_text(json.dumps(result,indent=2),encoding='utf8')
        print(json.dumps({'shard':path.name,'reviewed_runs_scanned':len(reports)}),flush=True)
    if len(reports)!=len(selected) or {r['run_id'] for r in reports}!=selected:
        raise ValueError('missing or duplicate selected runs')
    summary={'runs':len(reports),'sources':sources,
        'runs_with_node_conflicts':sum(r['conflicting_node_ids']>0 for r in reports),
        'runs_with_missing_parents':sum(r['missing_parent_ids']>0 for r in reports),
        'runs_with_no_node_ids':sum(r['unique_input_nodes']==0 for r in reports),
        'runs_with_unobserved_final_outputs':sum(r['final_outputs_not_observed_as_input']>0 for r in reports),
        'complete_executed_transcript_certified':False,'gpu_admitted':False,
        'scope':'Conservative reconstruction feasibility audit; outputs absent from all observed inputs may be unselected candidates or terminal actions, not automatically errors. Signature membership does not establish chronology or execution.'}
    (a.out/'SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
