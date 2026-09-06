"""Tokenize learning-only prompts with archived pinned tokenizer; no model/DEV reads."""
import argparse
import json
from pathlib import Path
from interaction_sprint.hindsight_calibration import prepare,digest
from interaction_sprint.hindsight_pahf_reduced import HINDSIGHT_BLOCK


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--learning',type=Path,required=True)
    p.add_argument('--archived-run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();repo=Path(__file__).resolve().parents[1]
    manifest=json.loads((a.archived_run/'MANIFEST.json').read_text())
    for name in ('model_provenance.json','tokenizer_backend.json'):
        if digest((a.archived_run/name).read_bytes())!=manifest[name]:raise ValueError('archived tokenizer provenance differs')
    provenance=json.loads((a.archived_run/'model_provenance.json').read_text())
    effective=provenance['effective_tokenizer']
    cfg=json.loads((repo/'configs/hindsight_calibration_v1.json').read_text())
    if provenance['resolved_revision']!=cfg['model_revision']:raise ValueError('revision differs')
    from transformers import PreTrainedTokenizerFast
    tokenizer=PreTrainedTokenizerFast(tokenizer_file=str(a.archived_run/'tokenizer_backend.json'),
        chat_template=effective['chat_template'],**effective['special_tokens_map'])
    if digest(tokenizer.backend_tokenizer.to_str().encode())!=effective['backend_sha256']:raise ValueError('backend differs')
    selected=prepare(a.learning);checks=[]
    for split in ('train','holdout'):
        for row in selected[split]:
            for teacher in (False,True):
                text=cfg['common_task_prefix']+row['prompt']+(HINDSIGHT_BLOCK.format(follow_up=row['delayed_expression_followup'].strip()) if teacher else '')
                rendered=tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=False,
                    add_generation_prompt=True,enable_thinking=False)
                tokens=tokenizer.encode(rendered,add_special_tokens=False)
                if len(tokens)>1024:raise ValueError('token overflow on '+row['id'])
                checks.append({'id':row['id'],'split':split,'teacher_context':teacher,'tokens':len(tokens),
                    'rendered_sha256':digest(rendered.encode()),'input_ids_sha256':digest(json.dumps(tokens,separators=(',',':')).encode())})
    report={'status':'LEARNING_ONLY_TOKEN_PREFLIGHT_PASS','renderings':len(checks),
        'max_tokens':max(r['tokens'] for r in checks),'checks':checks,
        'old_dev_accessed':False,'confirmation_accessed':False,'model_loaded':False,
        'archive_model_provenance_sha256':manifest['model_provenance.json'],
        'scope':'archived tokenizer/template, local transformers rendering; GPU runtime rechecks exact inputs'}
    with a.out.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))


if __name__=='__main__':main()
