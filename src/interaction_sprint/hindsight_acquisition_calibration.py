"""Single prospective supervised acquisition control; no feedback experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from .parameter_probe import MODEL,REVISION
from .hindsight_matched_learning import data_and_schedule
from latent_contract.sender_update import install_lora,adapter_state

SEED=90412017
STEPS=96


def dataset():
    original,_=data_and_schedule();rows=[]
    for c in original:
        d=c['domain'];a,b=c['options']
        texts=[f'Choose my {d}.\nA: {a}\nB: {b}\nAnswer A or B.',
               f'For my {d}, select one: A) {a}; B) {b}. Reply only with its letter.',
               f'I am choosing a {d}. My options are A: {a} and B: {b}. Pick A or B.',
               f'Which {d} should I select?\n(A) {a}\n(B) {b}\nReturn the option letter only.',
               f'Help me decide on a {d}: option A is {a}, whereas option B is {b}. Give just A or B.',
               f'Please settle my {d} choice. Candidate A: {a}. Candidate B: {b}. Answer using the candidate letter.']
        for renderer,text in enumerate(texts):
            rows.append(dict(id=f'{c["id"]}-r{renderer}',base_id=c['id'],domain=d,
                renderer=renderer,split='train' if renderer<4 else 'eval',target=c['target'],prompt=text))
    rng=random.Random(SEED);schedule=[]
    train=[r['id'] for r in rows if r['split']=='train']
    for _ in range(STEPS//8):
        order=train.copy();rng.shuffle(order)
        schedule.extend(order[i:i+8] for i in range(0,len(order),8))
    return rows,schedule


def run(root):
    import transformers
    from transformers import AutoTokenizer,AutoModelForCausalLM
    root.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    def write(name,value):
        with (root/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
    rows,schedule=dataset();lookup={r['id']:r for r in rows}
    write('spec.json',dict(model=MODEL,revision=REVISION,seed=SEED,steps=STEPS,batch=8,
        rank=8,alpha=16,lr=.0003,loss='full-vocabulary first-answer-token cross entropy',
        checkpoint_steps=[32,64,96],selection='Final step only, no heldout checkpoint selection',
        qualification='Final eval >=90% full-vocabulary argmax accuracy; each domain >=75%; every A/B mass >=.95',
        automatic_expansion=False,paper_green_light=False))
    write('cases.json',rows);write('schedule.json',schedule)
    (root/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    write('dependencies.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__).with_name('hindsight_matched_learning.py'),Path(__file__).with_name('hindsight_revelation_probe.py'),
         Path(__file__).parents[1]/'latent_contract'/'sender_update.py']})
    torch.set_num_threads(4);torch.manual_seed(SEED)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,local_files_only=True,
        dtype=torch.float32,attn_implementation='eager').eval()
    answer_tokens=[tok.encode(t,add_special_tokens=False) for t in ('A','B')]
    if any(len(t)!=1 for t in answer_tokens):raise ValueError('Invalid answer tokens')
    ab=torch.tensor([t[0] for t in answer_tokens]);encoded={}
    for r in rows:
        ids=tok.apply_chat_template([dict(role='user',content=r['prompt'])],tokenize=True,
            add_generation_prompt=True,enable_thinking=False,return_dict=False)
        if not ids or len(ids)>256:raise ValueError('Invalid length, no truncation')
        encoded[r['id']]=ids
    write('token_ids.json',encoded)
    counts=dict(forward_batches=0,forward_examples=0,backwards=0,updates=0)
    def logits(batch):
        counts['forward_batches']+=1;counts['forward_examples']+=len(batch)
        lengths=torch.tensor([len(encoded[r['id']]) for r in batch]);width=int(lengths.max())
        ids=torch.full((len(batch),width),tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,dtype=torch.long)
        mask=torch.zeros_like(ids)
        for i,r in enumerate(batch):
            seq=encoded[r['id']];ids[i,:len(seq)]=torch.tensor(seq);mask[i,:len(seq)]=1
        h=model.model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
        return model.get_output_embeddings()(h[torch.arange(len(batch)),lengths-1])
    def evaluate():
        out=[]
        with torch.no_grad():
            for r in rows:
                z=logits([r])[0].double();lp=z.log_softmax(0);conditional=z[ab].softmax(0)
                out.append(dict(id=r['id'],domain=r['domain'],split=r['split'],target=r['target'],
                    probabilities=conditional.tolist(),full_argmax=int(z.argmax()),
                    target_token=int(ab[r['target']]),target_probability=float(lp[ab[r['target']]].exp()),
                    nll=float(-lp[ab[r['target']]]),AB_mass=float(lp[ab].exp().sum())))
        return out
    logs=[];opt=None
    try:
        # Check right-padded batching against single-example logits before updates.
        with torch.no_grad():
            mini=[rows[0],rows[-1]];batched=logits(mini)
            singles=torch.cat([logits([r]) for r in mini],dim=0)
            max_diff=float((batched-singles).abs().max())
        if max_diff>1e-3:raise ValueError('Batch equivalence exceeds frozen tolerance')
        modules=install_lora(model,rank=8,alpha=16)
        params=[p for p in model.parameters() if p.requires_grad]
        torch.save(adapter_state(model),root/'initial_adapter.pt')
        write('runtime.json',dict(torch=torch.__version__,transformers=transformers.__version__,modules=modules,
            batch_single_max_logit_difference=max_diff))
        write('baseline.json',evaluate())
        opt=torch.optim.AdamW(params,lr=.0003,weight_decay=0.)
        for step,ids in enumerate(schedule,1):
            batch=[lookup[i] for i in ids];opt.zero_grad(set_to_none=True)
            z=logits(batch);target=ab[torch.tensor([r['target'] for r in batch])]
            loss=torch.nn.functional.cross_entropy(z,target)
            loss.backward();counts['backwards']+=1
            norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            opt.step();counts['updates']+=1
            logs.append(dict(step=step,loss=float(loss.detach()),gradient_norm=float(norm)))
            if step in (32,64,96):
                torch.save(adapter_state(model),root/f'adapter_{step}.pt');torch.save(opt.state_dict(),root/f'optimizer_{step}.pt')
                print(json.dumps(dict(step=step,elapsed=time.monotonic()-start)),flush=True)
        final=evaluate();write('final.json',final);write('steps.json',logs)
        def summarize(rs):
            return dict(n=len(rs),correct=sum(int(r['full_argmax']==r['target_token']) for r in rs),
                nll=float(np.mean([r['nll'] for r in rs])),min_AB_mass=min(r['AB_mass'] for r in rs),
                domains={d:dict(n=sum(r['domain']==d for r in rs),correct=sum(int(r['full_argmax']==r['target_token']) for r in rs if r['domain']==d))
                         for d in sorted({r['domain'] for r in rs})})
        metrics={s:summarize([r for r in final if r['split']==s]) for s in ('train','eval')};e=metrics['eval']
        qualified=e['correct']/e['n']>=.9 and e['min_AB_mass']>=.95 and all(d['correct']/d['n']>=.75 for d in e['domains'].values())
        write('RESULT.json',dict(status='ACQUISITION_QUALIFIED' if qualified else 'ACQUISITION_NOT_QUALIFIED',
            metrics=metrics,counts=counts,elapsed=time.monotonic()-start,paper_green_light=False))
        write('MANIFEST.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()})
        print(json.dumps(dict(qualified=qualified,metrics=metrics,elapsed=time.monotonic()-start)),flush=True)
    except Exception as e:
        write('FAILED.json',dict(error=type(e).__name__,message=str(e),counts=counts));write('partial_steps.json',logs)
        torch.save(adapter_state(model),root/'failure_adapter.pt')
        if opt is not None:torch.save(opt.state_dict(),root/'failure_optimizer.pt')
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    run(p.parse_args().root)
