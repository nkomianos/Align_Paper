"""Staged full-vocabulary oracle-marginal feedback learning, CPU only."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from .parameter_probe import MODEL,REVISION
from .hindsight_acquisition_calibration import dataset,SEED
from .hindsight_matched_learning import data_and_schedule
from .hindsight_revelation_probe import published_template
from latent_contract.sender_update import install_lora,adapter_state,load_adapter

ARMS=('truthful_kl','copying_kl','fixed_marginal_kl','copying_plus_anchors','anchors_only','truthful_direct')


def report_weights(prob,ab,target,rho):
    valid=prob[ab].detach()
    missing=(1-valid.sum()).clamp(min=0)
    weights=rho*torch.cat((valid,missing.view(1)))
    weights=weights.clone();weights[target]+=1-rho
    return weights


def run(root,calibration,config):
    from transformers import AutoTokenizer,AutoModelForCausalLM
    import transformers
    template=published_template(config)
    receipt=json.loads((calibration/'MANIFEST.json').read_text())
    for name,digest in receipt.items():
        f=(calibration/name).resolve()
        if f.parent!=calibration.resolve() or hashlib.sha256(f.read_bytes()).hexdigest()!=digest:raise ValueError('Acquisition evidence mismatch')
    if json.loads((calibration/'RESULT.json').read_text())['status']!='ACQUISITION_QUALIFIED':raise ValueError('Acquisition not qualified')
    root.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    def write(name,value):
        with (root/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
    rows,schedule=dataset();lookup={r['id']:r for r in rows};training=[r for r in rows if r['split']=='train']
    original,_=data_and_schedule();base={r['id']:r for r in original}
    write('spec.json',dict(model=MODEL,revision=REVISION,seed=SEED,steps=96,batch=8,lr=.0003,rank=8,alpha=16,
        arms=ARMS,teacher='Frozen original base, full vocabulary, published hindsight block',
        stop_after_unqualified_truthful=True,calibration_manifest_sha256=hashlib.sha256((calibration/'MANIFEST.json').read_bytes()).hexdigest(),
        initial_adapter_sha256=receipt['initial_adapter.pt'],template=template,paper_green_light=False))
    write('cases.json',rows);write('schedule.json',schedule)
    write('anchor_ids.json',[r['id'] for r in training if base[r['base_id']]['anchor']])
    (root/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    write('dependencies.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__).with_name('hindsight_acquisition_calibration.py'),Path(__file__).with_name('hindsight_matched_learning.py'),
         Path(__file__).with_name('hindsight_revelation_probe.py'),Path(__file__).parents[1]/'latent_contract'/'sender_update.py']})
    torch.set_num_threads(4);torch.manual_seed(SEED)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,local_files_only=True,
        dtype=torch.float32,attn_implementation='eager').eval()
    token_ids=[tok.encode(t,add_special_tokens=False) for t in ('A','B')]
    if any(len(t)!=1 for t in token_ids):raise ValueError('Invalid answer tokens')
    ab=torch.tensor([t[0] for t in token_ids]);encoded={};teacher_prompt_rows=[]
    def encode(text):
        ids=tok.apply_chat_template([dict(role='user',content=text)],tokenize=True,add_generation_prompt=True,
            enable_thinking=False,return_dict=False)
        if not ids or len(ids)>512:raise ValueError('No truncation')
        return ids
    for r in rows:encoded[r['id']]=encode(r['prompt'])
    write('token_ids.json',encoded)
    counts=dict(forward_batches=0,forward_examples=0,backwards=0,updates=0)
    def forward(idslist):
        counts['forward_batches']+=1;counts['forward_examples']+=len(idslist)
        lengths=torch.tensor([len(ids) for ids in idslist]);width=int(lengths.max())
        ids=torch.full((len(idslist),width),tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,dtype=torch.long)
        mask=torch.zeros_like(ids)
        for i,seq in enumerate(idslist):ids[i,:len(seq)]=torch.tensor(seq);mask[i,:len(seq)]=1
        h=model.model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
        return model.get_output_embeddings()(h[torch.arange(len(idslist)),lengths-1])
    teacher={};logs=[];opt=None;current_arm=None
    def evaluate():
        result=[]
        with torch.no_grad():
            for r in rows:
                z=forward([encoded[r['id']]])[0].double();lp=z.log_softmax(0)
                result.append(dict(id=r['id'],domain=r['domain'],split=r['split'],target=r['target'],
                    anchor=base[r['base_id']]['anchor'],full_argmax=int(z.argmax()),target_token=int(ab[r['target']]),
                    target_probability=float(lp[ab[r['target']]].exp()),nll=float(-lp[ab[r['target']]]),
                    AB_mass=float(lp[ab].exp().sum()),AB_probabilities=lp[ab].exp().tolist()))
        return result
    def metrics(rs):
        return dict(n=len(rs),correct=sum(int(r['full_argmax']==r['target_token']) for r in rs),
            nll=float(np.mean([r['nll'] for r in rs])),mean_target_probability=float(np.mean([r['target_probability'] for r in rs])),
            min_AB_mass=min(r['AB_mass'] for r in rs),
            domains={d:dict(n=sum(r['domain']==d for r in rs),correct=sum(int(r['full_argmax']==r['target_token']) for r in rs if r['domain']==d)) for d in sorted({r['domain'] for r in rs})})
    def finish(status,results):
        write('RESULT.json',dict(status=status,results=results,counts=counts,elapsed=time.monotonic()-start,paper_green_light=False))
        write('MANIFEST.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()})
        print(json.dumps(dict(status=status,results=results,elapsed=time.monotonic()-start)),flush=True)
    try:
        with torch.no_grad():
            for r in training:
                opts=base[r['base_id']]['options']
                feedback=[f'My actual preference is {o}. Please use that preference.' for o in opts]+['Please answer using only A or B.']
                idslist=[]
                for index,message in enumerate(feedback):
                    text=r['prompt']+template.format(follow_up=message)
                    ids=encode(text);idslist.append(ids)
                    teacher_prompt_rows.append(dict(id=r['id'],report=index,text=text,token_ids=ids))
                teacher[r['id']]=forward(idslist).double().log_softmax(-1).float().cpu()
        torch.save(teacher,root/'teacher_logprobabilities.pt');write('teacher_prompts.json',teacher_prompt_rows)
        initial_p={}
        for r in training:
            q=teacher[r['id']][r['target']].double().exp()
            initial_p[r['id']]=dict(target_probability=float(q[ab[r['target']]]),AB_mass=float(q[ab].sum()),argmax=int(q.argmax()))
        write('truthful_teacher_diagnostic.json',initial_p)
        print(json.dumps(dict(stage='teacher_prepared',prompts=192,elapsed=time.monotonic()-start)),flush=True)
        install_lora(model,rank=8,alpha=16)
        initial=torch.load(calibration/'initial_adapter.pt',weights_only=True,map_location='cpu');load_adapter(model,initial)
        torch.save(initial,root/'initial_adapter.pt')
        params=[p for p in model.parameters() if p.requires_grad]
        baseline=evaluate();write('baseline.json',baseline)
        baseline_lookup={r['id']:r for r in baseline};fixed={}
        for r in training:
            v=torch.tensor(baseline_lookup[r['id']]['AB_probabilities'],dtype=torch.float64)
            m=.9*torch.cat((v,(1-v.sum()).clamp(min=0).view(1)));m[r['target']]+=.1;fixed[r['id']]=m
        write('fixed_marginals.json',{k:v.tolist() for k,v in fixed.items()})
        write('runtime.json',dict(torch=torch.__version__,transformers=transformers.__version__,vocabulary_size=model.config.vocab_size))
        results={}
        for arm in ARMS:
            current_arm=arm;load_adapter(model,initial);opt=torch.optim.AdamW(params,lr=.0003,weight_decay=0.);arm_logs=[]
            for step,batch_ids in enumerate(schedule,1):
                batch=[lookup[i] for i in batch_ids];opt.zero_grad(set_to_none=True)
                z=forward([encoded[r['id']] for r in batch]);lp=z.double().log_softmax(-1);pi=lp.exp()
                targets=ab[torch.tensor([r['target'] for r in batch])]
                supervised=-lp[torch.arange(len(batch)),targets]
                mask=torch.tensor([base[r['base_id']]['anchor'] for r in batch],dtype=torch.bool)
                if arm=='truthful_direct':loss=supervised.mean();marginals=[]
                elif arm=='anchors_only':
                    loss=supervised[mask].mean() if mask.any() else z.sum()*0;marginals=[]
                else:
                    losses=[];marginals=[]
                    for i,r in enumerate(batch):
                        m=fixed[r['id']] if arm=='fixed_marginal_kl' else report_weights(pi[i],ab,r['target'],0. if arm=='truthful_kl' else .9)
                        q=teacher[r['id']].double()
                        losses.append((m[:,None]*pi[i][None,:]*(lp[i][None,:]-q)).sum());marginals.append(m.tolist())
                    loss=torch.stack(losses).mean()
                    if arm=='copying_plus_anchors' and mask.any():loss=loss+supervised[mask].mean()
                loss.backward();counts['backwards']+=1
                norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step();counts['updates']+=1
                arm_logs.append(dict(step=step,loss=float(loss.detach()),gradient_norm=float(norm),marginals=marginals,
                    AB_probabilities=pi[:,ab].detach().tolist(),anchor_count=int(mask.sum())))
                if step in (32,64,96):print(json.dumps(dict(arm=arm,step=step,elapsed=time.monotonic()-start)),flush=True)
            torch.save(adapter_state(model),root/(arm+'_adapter.pt'));torch.save(opt.state_dict(),root/(arm+'_optimizer.pt'))
            write(arm+'_steps.json',arm_logs);logs.append(arm)
            final=evaluate();write(arm+'_eval.json',final)
            results[arm]={s:metrics([r for r in final if r['split']==s]) for s in ('train','eval')}
            if arm=='truthful_kl':
                e=results[arm]['eval'];qualified=e['correct']/e['n']>=.9 and e['min_AB_mass']>=.95 and all(d['correct']/d['n']>=.75 for d in e['domains'].values())
                if not qualified:finish('STOP_TRUTHFUL_HINDSIGHT_LEARNING_UNQUALIFIED',results);return
        finish('COMPLETED_SIX_ARM_DEVELOPMENTAL_COMPARISON',results)
    except Exception as e:
        write('FAILED.json',dict(error=type(e).__name__,message=str(e),counts=counts,arm=current_arm,completed_arms=logs))
        torch.save(adapter_state(model),root/'failure_adapter.pt')
        if opt is not None:torch.save(opt.state_dict(),root/'failure_optimizer.pt')
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--upstream-config',type=Path,required=True)
    a=p.parse_args();run(a.root,a.calibration,a.upstream_config)
