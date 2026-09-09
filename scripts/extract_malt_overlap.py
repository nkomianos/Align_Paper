"""Preserve complete API records for the fixed metadata-overlap cohort.

This is a source extraction, not an executed-transcript normalizer. No candidates
are selected, no source labels are rewritten, and no monitor split is assigned.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--matching',type=Path,required=True)
    p.add_argument('--source-audit',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();matching_raw=a.matching.read_bytes()
    matching=json.loads(matching_raw);sources=json.loads(a.source_audit.read_bytes())['sources']
    cells=[c for c in matching['task_model']['cells'] if c['positive_ids'] and c['negative_ids']]
    wanted={i for c in cells for i in c['positive_ids']+c['negative_ids']}
    a.out.mkdir(parents=True,exist_ok=False);records={}
    for path in sorted(a.data.glob('*.parquet')):
        # Read only metadata first, avoiding decompression of irrelevant transcripts.
        metas=pq.read_table(path,columns=['metadata']).to_pylist()
        indexes={i:r['metadata']['run_id'] for i,r in enumerate(metas) if r['metadata']['run_id'] in wanted}
        if not indexes:continue
        with path.open('rb') as source_file:
            sha=hashlib.file_digest(source_file,'sha256').hexdigest()
        if sha!=sources[path.name]:raise ValueError('source changed: '+path.name)
        for index,batch in enumerate(pq.ParquetFile(path).iter_batches(batch_size=1)):
            if index not in indexes:continue
            row=batch.to_pylist()[0];run=row['metadata']['run_id']
            if run in records:raise ValueError('duplicate selected run')
            raw=json.dumps(row,ensure_ascii=False,separators=(',',':')).encode('utf8')
            target=a.out/f'run_{run}.json.gz'
            target.write_bytes(gzip.compress(raw,mtime=0))
            # Roundtrip equality checks all nested fields, including nulls.
            if json.loads(gzip.decompress(target.read_bytes()))!=row:
                raise ValueError('extraction changed nested source values')
            records[run]={'file':target.name,'source_shard':path.name,'source_row':index,
                'source_sha256':sha,'canonical_json_sha256':hashlib.sha256(raw).hexdigest(),
                'compressed_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                'samples':len(row['samples']),'canonical_bytes':len(raw)}
        print(json.dumps({'shard':path.name,'extracted':len(records),'target':len(wanted)}),flush=True)
    if set(records)!=wanted:raise ValueError('selected source rows missing')
    manifest={'records':records,'selection_sha256':hashlib.sha256(matching_raw).hexdigest(),
        'complete_api_records':True,'executed_path_certified':False,'monitor_ready':False,
        'scope':'Metadata-overlap cohort; preserves all API candidates without assigning execution.'}
    (a.out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print(json.dumps({'complete_runs':len(records),'monitor_ready':False}),flush=True)


if __name__=='__main__':main()
