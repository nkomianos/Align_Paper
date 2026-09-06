"""Bounded three-arm diagnostic, not a scientific confirmation or SDPO replication."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from interaction_sprint.hindsight_calibration import ARMS, prepare, summarize, qualified_teacher, route, digest
from interaction_sprint.hindsight_execution_integrity import capture_execution_provenance, capture_model_provenance, seal_artifacts, tensor_tree_sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--learning', type=Path, required=True)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--preflight-only', action='store_true')
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    cfg = json.loads((repo/'configs/hindsight_calibration_v1.json').read_text())
    selected = prepare(args.learning)
    if args.snapshot.name != cfg['model_revision']:
        raise ValueError('wrong immutable snapshot revision')
    if args.preflight_only:
        print(json.dumps({'status':'CPU_INPUT_CHECK_PASS', 'train_rows':len(selected['train']),
            'holdout_rows':len(selected['holdout']), 'model_loaded':False}))
        return
    if os.environ.get('CALIBRATION_SUPERVISED_LAUNCH') != '1':
        raise RuntimeError('use launch_hindsight_calibration.py for bounded execution')
    for source in (args.learning.parent, args.snapshot):
        if args.out.resolve().is_relative_to(source.resolve()):
            raise ValueError('output cannot modify inputs/model')
    args.out.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        with (args.out/name).open('x', encoding='utf8') as f:
            json.dump(value, f, indent=2, allow_nan=False)
    started = time.monotonic()
    provenance = capture_execution_provenance(repo, [Path(__file__), 'configs/hindsight_calibration_v1.json',
        'docs/HINDSIGHT_CALIBRATION_PROTOCOL_20260905.md'], sys.argv, cfg)
    write('PROVENANCE.json', provenance)
    write('INPUTS.json', selected)
    write('CONFIG.json', cfg)
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM
    from interaction_sprint.hindsight_neural_anchor import install_qwen35_lora, reverse_kl_per_example
    from interaction_sprint.hindsight_pahf_reduced import HINDSIGHT_BLOCK
    from latent_contract.sender_update import adapter_state, load_adapter
    if torch.cuda.device_count() != 1:
        raise RuntimeError('exactly one GPU required')
    if torch.cuda.get_device_properties(0).total_memory < 90*1024**3:
        raise RuntimeError('requires at least90GiB; no automatic smaller-model substitution')
    write('MODEL.json', capture_model_provenance(args.snapshot, cfg['model_id'], cfg['model_revision']))
    torch.manual_seed(cfg['seed']); torch.cuda.manual_seed_all(cfg['seed'])
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
    tokenizer.padding_side='left'
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    (args.out/'tokenizer.json').write_text(tokenizer.backend_tokenizer.to_str(), encoding='utf8')
    model = Qwen3_5ForCausalLM.from_pretrained(args.snapshot, local_files_only=True, dtype=torch.bfloat16,
        device_map={'':0}, attn_implementation='sdpa', use_kernels=False).eval()
    model.config.use_cache=False
    install_qwen35_lora(model, rank=8, alpha=16)
    parameters=[p for p in model.parameters() if p.requires_grad]
    initial=adapter_state(model)
    torch.save(initial, args.out/'initial_adapter.pt')
    ids=[tokenizer.encode(x, add_special_tokens=False) for x in 'ABCD']
    if any(len(x)!=1 for x in ids) or len({x[0] for x in ids})!=4: raise ValueError('native answer tokens invalid')
    answer_ids=[x[0] for x in ids]
    write('RUNTIME.json', {'torch':torch.__version__, 'gpu':torch.cuda.get_device_name(0),
        'answer_ids':answer_ids, 'initial_adapter_hash':tensor_tree_sha256(initial)})
    serial=0
    def forward(rows, teacher=False, grad=False):
        nonlocal serial
        texts=[cfg['common_task_prefix']+r['prompt']+(HINDSIGHT_BLOCK.format(follow_up=r['delayed_expression_followup'].strip()) if teacher else '') for r in rows]
        rendered=[tokenizer.apply_chat_template([{'role':'user','content':s}],tokenize=False,add_generation_prompt=True,enable_thinking=False) for s in texts]
        encoded=tokenizer(rendered,return_tensors='pt',padding=True,add_special_tokens=False).to('cuda')
        if int(encoded['attention_mask'].sum(-1).max())>1024: raise ValueError('truncation forbidden')
        with torch.set_grad_enabled(grad):
            output=model(**encoded,use_cache=False,logits_to_keep=1).logits[:,-1,:].float()
        if not torch.isfinite(output).all(): raise ValueError('nonfinite logits')
        serial+=1
        # Every actual forward is replayable, including training and teacher targets.
        torch.save({'ids':[r['id'] for r in rows], 'teacher_context':teacher, 'student_gradient':grad,
            'input_ids':encoded['input_ids'].cpu(), 'attention_mask':encoded['attention_mask'].cpu(),
            'rendered':rendered, 'logits':output.detach().cpu()}, args.out/f'forward_{serial:06d}.pt')
        return output
    def score(rows, name, teacher=False, cache=None):
        result=[]
        for start in range(0,len(rows),cfg['microbatch']):
            batch=rows[start:start+cfg['microbatch']]
            output=forward(batch,teacher)
            if cache is not None:cache.append(output.cpu())
            log=output.log_softmax(-1); choice=log[:,answer_ids]; cond=choice.log_softmax(-1)
            for i,r in enumerate(batch):
                target='ABCD'.index(r['old_target'])
                result.append({'id':r['id'],'base_id':r['base_id'],'label_rotation':r['label_rotation'],
                    'forward_file':f'forward_{serial:06d}.pt','forward_row':i,
                    'nll':-float(choice[i,target]),'probability':float(choice[i,target].exp()),
                    'conditional_probability':float(cond[i,target].exp()),
                    'correct':int(int(choice[i].argmax())==target),'choice_mass':float(choice[i].exp().sum())})
        summary=summarize(result)
        write(name+'.json', {'rows':result,'summary':summary})
        return summary
    train, held=selected['train'], selected['holdout']
    baseline=score(held,'baseline_student')
    teacher_initial=score(held,'baseline_teacher',True)
    if not qualified_teacher(teacher_initial):
        write('RESULT.json',{'decision':'STOP_INVALID_INITIAL_TEACHER','paper_green_light':False})
        seal_artifacts(args.out); return
    # Cache teacher at exactly theta0 before any optimizer is created.
    frozen=[]
    initial_train_teacher=score(train,'baseline_train_teacher',True,cache=frozen)
    if not qualified_teacher(initial_train_teacher):
        write('RESULT.json',{'decision':'STOP_INVALID_INITIAL_TEACHER','paper_green_light':False})
        seal_artifacts(args.out); return
    final, teachers={}, {}
    for arm in ARMS:
        load_adapter(model, initial)
        if tensor_tree_sha256(adapter_state(model))!=tensor_tree_sha256(initial): raise ValueError('reset failed')
        optimizer=torch.optim.AdamW(parameters,lr=cfg['learning_rate'],betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False)
        if optimizer.state: raise ValueError('optimizer is not fresh')
        torch.save(optimizer.state_dict(),args.out/f'{arm}_initial_optimizer.pt')
        updates=[]
        for step in range(cfg['steps']):
            start=step*cfg['effective_batch']; optimizer.zero_grad(set_to_none=True)
            losses=[]
            for offset in range(0,cfg['effective_batch'],cfg['microbatch']):
                pos=start+offset; batch=train[pos:pos+cfg['microbatch']]
                target=None
                if arm=='frozen_teacher': target=frozen[pos//cfg['microbatch']].to('cuda')
                elif arm=='current_teacher': target=forward(batch,True)
                student=forward(batch,grad=True)
                if arm=='supervised':
                    y=torch.tensor([answer_ids['ABCD'.index(r['old_target'])] for r in batch],device='cuda')
                    loss=torch.nn.functional.cross_entropy(student,y)
                else: loss=reverse_kl_per_example(student,target).mean()
                if not torch.isfinite(loss): raise ValueError('nonfinite objective')
                (loss*(len(batch)/cfg['effective_batch'])).backward()
                losses.append(float(loss.detach()))
                del student,target,loss
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.0,error_if_nonfinite=True)
            optimizer.step()
            updates.append({'step':step+1,'row_ids':[r['id'] for r in train[start:start+cfg['effective_batch']]],
                'loss':sum(losses)/len(losses),'gradient_norm':float(norm)})
            write(f'{arm}_step_{step+1:03d}.json', updates[-1])
            if step+1 in cfg['checkpoints']:
                torch.save(adapter_state(model),args.out/f'{arm}_{step+1}_adapter.pt')
                score(held,f'{arm}_{step+1}_student')
                score(held,f'{arm}_{step+1}_teacher',True)
        final[arm]=score(held,arm+'_final_student')
        teachers[arm]=score(held,arm+'_final_teacher',True)
        score(train,arm+'_train_student')
        torch.save(adapter_state(model),args.out/f'{arm}_adapter.pt')
        torch.save(optimizer.state_dict(),args.out/f'{arm}_optimizer.pt')
        del optimizer
    for rel,sha in provenance['source_sha256'].items():
        if digest((repo/rel).read_bytes())!=sha: raise RuntimeError('source changed during execution')
    write('RESULT.json',{**route(baseline,final,teachers),'wall_seconds':time.monotonic()-started,
        'optimizer_updates':96,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
        'old_dev_accessed':False,'confirmation_accessed':False})
    seal_artifacts(args.out)


if __name__=='__main__': main()
