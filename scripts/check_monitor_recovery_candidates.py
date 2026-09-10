"""Check fuller upstream transcript candidates against every retained field."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import pyarrow.parquet as pq
from run_unexplored_screens import dump,sha


def identity_matches(rows,trial_name,model,agents,trial_id=None):
    return [r for r in rows if r['trial_name']==trial_name and r['model']==model
            and r['agent'] in agents and (trial_id is None or r['trial_id']==trial_id)]


def differences(partial,full,path=''):
    if partial==full:return []
    if isinstance(partial,str) and re.fullmatch(r'\$\d+',partial):
        return [{'path':path,'kind':'serialized_reference','partial':partial,'full_preview':str(full)[:100]}]
    if partial=='' and '.tools[' in path and path.endswith('.cmd') and isinstance(full,str):
        return [{'path':path,'kind':'empty_tool_argument','full_preview':full[:100]}]
    if isinstance(partial,dict) and isinstance(full,dict):
        out=[]
        for key in sorted(set(partial)|set(full)):
            out.extend(differences(partial.get(key),full.get(key),path+'.'+key))
        return out
    if isinstance(partial,list) and isinstance(full,list) and len(partial)==len(full):
        return [d for i,(a,b) in enumerate(zip(partial,full)) for d in differences(a,b,f'{path}[{i}]')]
    return [{'path':path,'kind':'unexplained_difference','partial_preview':str(partial)[:200],'full_preview':str(full)[:200]}]


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=False)
    audit=json.loads(a.audit.read_text());download=json.loads((a.source/'DOWNLOAD.json').read_text())
    assert sha(a.source/'DOWNLOAD.json')==audit['original_download_receipt_sha256']
    wanted=set();selected=[]
    for row in audit['records']:
        if row['source_unresolved_user'] and len(row['complete_original_candidates'])==1 and len(row['source_rows'])==1:
            wanted.add(row['source_rows'][0]['steps_sha256']);wanted.add(row['complete_original_candidates'][0]['steps_sha256']);selected.append(row)
    originals={}
    for receipt in download['files']:
        file=a.source/receipt['path'];assert sha(file)==receipt['sha256']
        for batch in pq.ParquetFile(file).iter_batches(batch_size=128):
            for row in batch.to_pylist():
                h=hashlib.sha256((row['steps'] or '').encode()).hexdigest()
                if h in wanted:originals.setdefault(h,[]).append(row)
    assert set(originals)==wanted
    results=[]
    for record in selected:
        partial_matches=identity_matches(originals[record['source_rows'][0]['steps_sha256']],
                         record['trial_name'],record['released_model'],record['source_agents'])
        candidate=record['complete_original_candidates'][0]
        full_matches=identity_matches(originals[candidate['steps_sha256']],record['trial_name'],
                         candidate['model'],[candidate['agent']],candidate['trial_id'])
        if len(partial_matches)!=1 or len(full_matches)!=1 or partial_matches[0]['agent']!=full_matches[0]['agent']:
            results.append({'id':record['id'],'compatible_retained_fields':False,'differences':[],
                'classification':'SOURCE_IDENTITY_NOT_UNIQUE_OR_HARNESS_MISMATCH',
                'partial_matches':len(partial_matches),'full_matches':len(full_matches),
                'independent_label_verified':False})
            continue
        partial=partial_matches[0];full=full_matches[0]
        left=json.loads(partial['steps']);right=json.loads(full['steps']);diff=differences(left,right,'steps')
        unexplained=sum(d['kind']=='unexplained_difference' for d in diff)
        accepted=unexplained==0 and bool(full['trial_id']) and len(left)==len(right)
        row={'id':record['id'],'candidate_trial_uuid':full['trial_id'],'n_steps':len(right),
            'partial_steps_sha256':record['source_rows'][0]['steps_sha256'],
            'full_steps_sha256':record['complete_original_candidates'][0]['steps_sha256'],
            'differences':diff,'compatible_retained_fields':accepted,
            'classification':'DEVELOPMENTAL_SOURCE_RECOVERY_CHECK','independent_label_verified':False}
        if accepted:
            # Separate new artifact: never overwrite released traces or labels.
            dump(a.out/(record['id']+'_full_candidate.json'),{'trial_uuid':full['trial_id'],
                'trial_name':full['trial_name'],'model':full['model'],'agent':full['agent'],
                'events':right,'source_steps_sha256':row['full_steps_sha256'],
                'scope':'Original fuller transcript; compatible with retained fields of damaged version. Not independently labeled.'})
        results.append(row)
    summary={'candidates':len(results),'retained_field_compatible':sum(r['compatible_retained_fields'] for r in results),
        'identity_rejections':sum(r['classification']=='SOURCE_IDENTITY_NOT_UNIQUE_OR_HARNESS_MISMATCH' for r in results),
        'difference_kinds':dict(Counter(d['kind'] for r in results for d in r['differences'])),'neural_admitted':False}
    dump(a.out/'AUDIT.json',{'summary':summary,'records':results,'source_audit_sha256':sha(a.audit),'script_sha256':sha(Path(__file__))})
    dump(a.out/'MANIFEST.json',{p.name:sha(p) for p in a.out.iterdir() if p.is_file()})
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
