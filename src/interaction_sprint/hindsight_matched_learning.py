"""Restricted-action, frozen-teacher LoRA learning with known feedback law."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from .parameter_probe import MODEL, REVISION
from .hindsight_revelation_probe import fresh_cases
from latent_contract.sender_update import install_lora, adapter_state, load_adapter

SEED=90411031
STEPS=24
ARMS=('truthful_kl','copying_kl','fixed_marginal_kl','copying_plus_anchors','anchors_only','truthful_direct')
SCORE_SHA='63f5797e4ae678e9765cfeaa3c1f3a3e9b38fc1135dafc6a92a648e240fcc80a'


def data_and_schedule():
    cases=fresh_cases();rng=random.Random(SEED)
    domains=sorted({c['domain'] for c in cases});rng.shuffle(domains)
    anchors=set(domains[:2])
    # One fixed semantic preference per domain, assigned before any training.
    preferred={d:rng.randrange(2) for d in domains}
    for c in cases:
        swap=int(c['id'].rsplit('-',1)[1])
        c['target']=preferred[c['domain']]^swap;c['anchor']=c['domain'] in anchors
        c['eval_prompt']=(f'I need you to pick my {c["domain"]}. The two choices are '
            f'A) {c["options"][0]} and B) {c["options"][1]}. Respond with just A or B.')
    schedule=[]
    for _ in range(STEPS//4):
        marked=[c['id'] for c in cases if c['anchor']];rest=[c['id'] for c in cases if not c['anchor']]
        rng.shuffle(marked);rng.shuffle(rest)
        for i in range(4):
            batch=[marked[i]]+rest[3*i:3*i+3];rng.shuffle(batch);schedule.append(batch)
    return cases,schedule


def feedback_marginal(policy,target,rho):
    truth=torch.zeros_like(policy);truth[target]=1
    return (1-rho)*truth+rho*policy.detach()


def run(root,score_path):
    from transformers import AutoTokenizer,AutoModelForCausalLM
    import transformers
    raw=score_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SCORE_SHA:raise ValueError('Frozen teacher scores mismatch')
    teacher_rows=json.loads(raw)
    teachers={(r['id'],r['target']):r for r in teacher_rows if r['kind']=='published'}
    root.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    def write(name,value):
        with (root/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)
    cases,schedule=data_and_schedule();byid={c['id']:c for c in cases}
    write('spec.json',dict(model=MODEL,revision=REVISION,seed=SEED,steps=STEPS,batch=4,rank=4,alpha=8,
        lr=.0003,clip=1.,arms=ARMS,copy_strength=.9,anchor_weight=1.,teacher_score_sha256=SCORE_SHA,
        teacher='Frozen base-model published-template A/B probabilities',automatic_expansion=False,
        scope='Actual LoRA updates, restricted categorical expectation, not full SDPO or human study'))
    write('cases.json',cases);write('schedule.json',schedule)
    (root/'teacher_scores.json').write_bytes(raw)
    (root/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    write('dependency_hashes.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__).with_name('hindsight_revelation_probe.py'),Path(__file__).parents[1]/'latent_contract'/'sender_update.py']})
    torch.set_num_threads(4);torch.manual_seed(SEED)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,local_files_only=True,
        dtype=torch.float32,attn_implementation='eager').eval()
    modules=install_lora(model,rank=4,alpha=8)
    params=[p for p in model.parameters() if p.requires_grad]
    initial=adapter_state(model);torch.save(initial,root/'initial_adapter.pt')
    ids=[tok.encode(s,add_special_tokens=False) for s in ('A','B')]
    if any(len(i)!=1 for i in ids):raise ValueError('Invalid tokens')
    answer_ids=torch.tensor([i[0] for i in ids]);encoded={};counts=dict(forwards=0,backwards=0,updates=0)
    for c in cases:
        for view,field in [('train','prompt'),('eval','eval_prompt')]:
            ids=tok.apply_chat_template([dict(role='user',content=c[field])],tokenize=True,
                add_generation_prompt=True,enable_thinking=False,return_dict=False)
            if len(ids)>256:raise ValueError('No truncation')
            encoded[c['id'],view]=torch.tensor([ids])
    write('token_ids.json',[dict(id=i,view=v,ids=x[0].tolist()) for (i,v),x in encoded.items()])
    def forward(c,view):
        counts['forwards']+=1
        h=model.model(input_ids=encoded[c['id'],view],use_cache=False).last_hidden_state[:,-1]
        logits=model.get_output_embeddings()(h)[0].double();z=logits[answer_ids]
        return z.log_softmax(0),float((z.detach().logsumexp(0)-logits.detach().logsumexp(0)).exp())
    def evaluate():
        rows=[]
        with torch.no_grad():
            for c in cases:
                for view in ('train','eval'):
                    lp,mass=forward(c,view)
                    rows.append(dict(id=c['id'],domain=c['domain'],anchor=c['anchor'],target=c['target'],view=view,
                        probabilities=lp.exp().tolist(),nll=float(-lp[c['target']]),AB_mass=mass))
        return rows
    results={};logs=[]
    try:
        baseline=evaluate();write('baseline.json',baseline)
        initial_p={r['id']:torch.tensor(r['probabilities'],dtype=torch.float64) for r in baseline if r['view']=='train'}
        write('runtime.json',dict(torch=torch.__version__,transformers=transformers.__version__,modules=modules))
        for arm in ARMS:
            load_adapter(model,initial)
            opt=torch.optim.AdamW(params,lr=.0003,weight_decay=0.)
            for step,batch in enumerate(schedule):
                opt.zero_grad(set_to_none=True);detail=[]
                for cid in batch:
                    c=byid[cid]
                    if arm=='anchors_only' and not c['anchor']:continue
                    lp,mass=forward(c,'train');pi=lp.exp()
                    if arm in ('anchors_only','truthful_direct'):
                        loss=-lp[c['target']]
                        scale=1 if arm=='anchors_only' else 4
                        marginal=None
                    else:
                        rho=0. if arm=='truthful_kl' else .9
                        marginal=feedback_marginal(initial_p[cid] if arm=='fixed_marginal_kl' else pi,c['target'],rho)
                        q=torch.tensor([teachers[cid,o]['probabilities'] for o in (0,1)],dtype=torch.float64)
                        if (q<=0).any():raise ValueError('Zero teacher probability')
                        loss=(marginal[:,None]*pi[None,:]*(lp[None,:]-q.log())).sum()
                        if arm=='copying_plus_anchors' and c['anchor']:
                            loss=loss-4*lp[c['target']]
                        scale=4
                    (loss/scale).backward();counts['backwards']+=1
                    detail.append(dict(id=cid,loss=float(loss.detach()),probabilities=pi.detach().tolist(),
                        marginal=None if marginal is None else marginal.tolist(),AB_mass=mass))
                norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
                opt.step();counts['updates']+=1
                logs.append(dict(arm=arm,step=step+1,gradient_norm=float(norm),examples=detail))
            torch.save(adapter_state(model),root/(arm+'_adapter.pt'));torch.save(opt.state_dict(),root/(arm+'_optimizer.pt'))
            rows=evaluate();write(arm+'_eval.json',rows)
            results[arm]={}
            for view in ('train','eval'):
                rs=[r for r in rows if r['view']==view]
                results[arm][view]=dict(n=len(rs),correct=sum(int(np.argmax(r['probabilities'])==r['target']) for r in rs),
                    nll=float(np.mean([r['nll'] for r in rs])))
            print(json.dumps(dict(arm=arm,result=results[arm],elapsed=time.monotonic()-start)),flush=True)
        write('steps.json',logs)
        write('RESULT.json',dict(results=results,counts=counts,elapsed=time.monotonic()-start,paper_green_light=False))
        write('MANIFEST.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()})
    except Exception as e:
        write('partial_steps.json',logs);write('FAILED.json',dict(error=type(e).__name__,message=str(e),counts=counts))
        torch.save(adapter_state(model),root/'failure_adapter.pt');raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--teacher-scores',type=Path,required=True)
    a=p.parse_args();run(a.root,a.teacher_scores)
