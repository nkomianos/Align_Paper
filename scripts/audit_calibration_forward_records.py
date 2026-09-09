"""Supplemental CPU audit of sealed prompts, tokens and logged training losses.

Does not replay the neural model or change the prospective decision rule.
Write the report outside the immutable experiment root.
"""
import argparse
import json
import math
from pathlib import Path


def audit(root, snapshot):
    import torch
    from transformers import AutoTokenizer
    from interaction_sprint.hindsight_execution_integrity import verify_manifest
    from interaction_sprint.hindsight_pahf_reduced import HINDSIGHT_BLOCK
    verify_manifest(root)
    read=lambda name: json.loads((root/name).read_text(encoding='utf8'))
    cfg=read('CONFIG.json'); selected=read('INPUTS.json')
    lookup={r['id']:r for split in ('train','holdout') for r in selected[split]}
    train=selected['train']; tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
    tokenizer.padding_side='left'
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    answers=[tokenizer.encode(x,add_special_tokens=False) for x in 'ABCD']
    if any(len(x)!=1 for x in answers):raise ValueError('answer token not singular')
    answers=[x[0] for x in answers]
    if answers!=read('RUNTIME.json')['answer_ids']:raise ValueError('answer token binding differs')
    if tokenizer.backend_tokenizer.to_str()!=(root/'tokenizer.json').read_text(encoding='utf8'):
        raise ValueError('tokenizer backend differs')
    # Bind the frozen teacher cache to the original pre-update scored forwards.
    cached={}
    if (root/'baseline_train_teacher.json').exists():
        for r in read('baseline_train_teacher.json')['rows']:
            cached[r['id']]=(r['forward_file'],r['forward_row'])
    last=None; gradient_batches=0; step_losses=[]; total=0
    arms=('supervised','frozen_teacher','current_teacher')
    per_arm=cfg['steps']*cfg['effective_batch']//cfg['microbatch']
    for path in sorted(root.glob('forward_*.pt')):
        raw=torch.load(path,map_location='cpu',weights_only=True); total+=1
        rows=[lookup[i] for i in raw['ids']]
        rendered=[]
        for row in rows:
            text=cfg['common_task_prefix']+row['prompt']
            if raw['teacher_context']:
                text+=HINDSIGHT_BLOCK.format(follow_up=row['delayed_expression_followup'].strip())
            rendered.append(tokenizer.apply_chat_template([{'role':'user','content':text}],
                tokenize=False,add_generation_prompt=True,enable_thinking=False))
        if rendered!=raw['rendered']:raise ValueError(f'rendered prompt differs: {path.name}')
        encoded=tokenizer(rendered,return_tensors='pt',padding=True,add_special_tokens=False)
        for key in ('input_ids','attention_mask'):
            if not torch.equal(encoded[key],raw[key]):raise ValueError(f'{key} differs: {path.name}')
        if int(encoded['attention_mask'].sum(-1).max())>1024:raise ValueError('overlength input')
        logits=raw['logits'].double()
        if logits.shape[0]!=len(rows) or not torch.isfinite(logits).all():raise ValueError('invalid logits')
        if raw['student_gradient']:
            if raw['teacher_context']:raise ValueError('teacher leakage into student')
            arm_index=gradient_batches//per_arm
            if arm_index>=len(arms):raise ValueError('extra training batch')
            arm=arms[arm_index]; local=gradient_batches%per_arm
            pos=local*cfg['microbatch']
            expected=[r['id'] for r in train[pos:pos+cfg['microbatch']]]
            if raw['ids']!=expected:raise ValueError('actual gradient schedule differs')
            if arm=='supervised':
                y=torch.tensor([answers['ABCD'.index(r['old_target'])] for r in rows])
                loss=torch.nn.functional.cross_entropy(logits,y)
            else:
                if arm=='current_teacher':
                    if last is None or last['ids']!=raw['ids'] or not last['teacher_context'] or last['student_gradient']:
                        raise ValueError('current teacher not paired with student')
                    target=last['logits'].double()
                else:
                    target=[]
                    for row in rows:
                        name,index=cached[row['id']]
                        original=torch.load(root/name,map_location='cpu',weights_only=True)
                        if original['ids'][index]!=row['id'] or not original['teacher_context']:
                            raise ValueError('frozen teacher binding differs')
                        target.append(original['logits'][index].double())
                    target=torch.stack(target)
                logp=logits.log_softmax(-1); logq=target.log_softmax(-1)
                loss=(logp.exp()*(logp-logq)).sum(-1).mean()
            step_losses.append(float(loss));gradient_batches+=1
            batches_per_step=cfg['effective_batch']//cfg['microbatch']
            if len(step_losses)==batches_per_step:
                step=local//batches_per_step+1
                recorded=read(f'{arm}_step_{step:03d}.json')
                value=sum(step_losses)/len(step_losses)
                if not math.isclose(value,recorded['loss'],rel_tol=1e-5,abs_tol=1e-5):
                    raise ValueError(f'training loss differs: {arm} {step}')
                step_losses=[]
        last=raw
    decision=read('RESULT.json')['decision']
    expected_batches=0 if decision=='STOP_INVALID_INITIAL_TEACHER' else len(arms)*per_arm
    if gradient_batches!=expected_batches or step_losses:raise ValueError('incomplete training forwards')
    if expected_batches:
        for arm in arms:
            initial=torch.load(root/f'{arm}_initial_optimizer.pt',map_location='cpu',weights_only=True)
            final=torch.load(root/f'{arm}_optimizer.pt',map_location='cpu',weights_only=True)
            if initial['state']:raise ValueError('optimizer did not start fresh')
            if not final['state'] or any(float(v['step'])!=cfg['steps'] for v in final['state'].values()):
                raise ValueError('optimizer steps differ')
    return {'verified':True,'forward_records':total,'gradient_batches':gradient_batches,
            'prompts_and_token_ids_replayed':True,'training_losses_recomputed':True,
            'optimizer_steps_checked':bool(expected_batches),'neural_replay':False,
            'scope':'Supplemental integrity audit; no new scientific endpoint or causal claim'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.resolve().is_relative_to(a.root.resolve()):raise ValueError('report must be outside evidence')
    result=audit(a.root,a.snapshot)
    with a.out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
