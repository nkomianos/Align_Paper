"""Offline prompt qualification using the archived, hash-checked model tokenizer."""
import argparse
import hashlib
import json
from pathlib import Path
from research_pilots.common import write,digest
from research_pilots.data import build,validate
from research_pilots.neural import REVISION
from research_pilots.reference import WITHHOLD


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archived-run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    manifest=json.loads((a.archived_run/'MANIFEST.json').read_text())
    for name in ('model_provenance.json','tokenizer_backend.json'):
        if hashlib.sha256((a.archived_run/name).read_bytes()).hexdigest()!=manifest[name]:
            raise ValueError('archived tokenizer hash differs')
    provenance=json.loads((a.archived_run/'model_provenance.json').read_text())
    if provenance['resolved_revision']!=REVISION:raise ValueError('revision differs')
    from transformers import PreTrainedTokenizerFast
    effective=provenance['effective_tokenizer']
    tokenizer=PreTrainedTokenizerFast(tokenizer_file=str(a.archived_run/'tokenizer_backend.json'),
        chat_template=effective['chat_template'],**effective['special_tokens_map'])
    if hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()!=effective['backend_sha256']:
        raise ValueError('effective tokenizer differs')
    data=validate(build());checks=[]
    for split in ('train','dev','holdout','utility'):
        for row in data[split]:
            for prefix in ('',WITHHOLD):
                rendered=tokenizer.apply_chat_template([{'role':'user','content':prefix+row['prompt']}],
                    tokenize=False,add_generation_prompt=True,enable_thinking=False)
                tokens=tokenizer.encode(rendered,add_special_tokens=False)
                if len(tokens)>4096:raise ValueError('token overflow')
                checks.append({'id':row['id'],'withholding':bool(prefix),'tokens':len(tokens),
                    'rendered_sha256':digest(rendered),'input_ids_sha256':digest(tokens)})
    ids=[tokenizer.encode(x,add_special_tokens=False) for x in ('A','B')]
    if any(len(x)!=1 for x in ids) or ids[0]==ids[1]:raise ValueError('answer tokens invalid')
    a.out.mkdir(parents=True,exist_ok=False)
    write(a.out/'INPUTS.json',data)
    report={'status':'ARCHIVED_TOKENIZER_PASS','answer_ids':ids,'checks':checks,
        'max_tokens':max(r['tokens'] for r in checks),'model_loaded':False,
        'monitor_data_checked':False,'paper_confirmation_accessed':False,
        'revision':REVISION,'archive_manifest_sha256':hashlib.sha256((a.archived_run/'MANIFEST.json').read_bytes()).hexdigest()}
    write(a.out/'TOKEN_PREFLIGHT.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))


if __name__=='__main__':main()
