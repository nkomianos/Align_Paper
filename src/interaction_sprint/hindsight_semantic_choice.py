"""Prospective, constrained semantic-choice teacher qualification; no training."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from .parameter_probe import MODEL, REVISION
from .hindsight_revelation_probe import published_template


def dataset():
    groups={
        'development': [('breakfast fruit','apple','pear'),('bedding fabric','cotton','linen'),
                        ('vacation season','summer','winter'),('exercise','swimming','cycling')],
        'confirmation': [('window covering','curtains','blinds'),('floor material','wood','tile'),
                         ('clock display','analog','digital'),('writing tool','pen','pencil')]}
    rows=[]
    for split,domains in groups.items():
        for index,(domain,a,b) in enumerate(domains):
            for order in (0,1):
                options=[a,b] if order==0 else [b,a]
                for wording in (0,1):
                    text=(f'Choose my {domain}: {options[0]} or {options[1]}. Respond with the chosen option.'
                          if wording==0 else
                          f'For my {domain}, the options are {options[0]} and {options[1]}. Which option should I use?')
                    for preferred in (0,1):
                        base=f'{split}-{index}-{order}-{wording}'
                        rows.append(dict(id=f'{base}-{preferred}',base_id=base,split=split,domain=domain,
                            order=order,wording=wording,options=options,preferred=(a,b)[preferred],
                            target=preferred^order,prompt=text))
    return rows


def distribution(scores):
    if len(scores)!=2 or not all(math.isfinite(v) for v in scores):
        raise ValueError('Need two finite sequence log probabilities')
    m=max(scores);weights=[math.exp(s-m) for s in scores];total=sum(weights)
    return [w/total for w in weights]


def summarize(rows):
    metrics={}
    for kind in ('direct','hindsight'):
        metrics[kind]={}
        for split in ('development','confirmation'):
            rs=[r for r in rows if r['kind']==kind and r['split']==split]
            metrics[kind][split]=dict(n=len(rs),correct=sum(r['prediction']==r['target'] for r in rs),
                mean_target_probability=sum(r['probabilities'][r['target']] for r in rs)/len(rs),
                domains={d:dict(n=sum(r['domain']==d for r in rs),
                    correct=sum(r['prediction']==r['target'] for r in rs if r['domain']==d))
                    for d in sorted({r['domain'] for r in rs})})
    qualified=all(m['correct']/m['n']>=.9 and all(d['correct']/d['n']>=.75 for d in m['domains'].values())
                  for bysplit in metrics.values() for m in bysplit.values())
    base={r['base_id']:r for r in rows if r['kind']=='base'}
    shifts={}
    for split in ('development','confirmation'):
        rs=[r for r in rows if r['kind']=='hindsight' and r['split']==split]
        delta=[r['probabilities'][r['target']]-base[r['base_id']]['probabilities'][r['target']] for r in rs]
        shifts[split]=dict(n=len(delta),positive=sum(d>1e-6 for d in delta),mean=sum(delta)/len(delta))
    return dict(status='CONSTRAINED_TEACHER_QUALIFIED' if qualified else 'CONSTRAINED_TEACHER_UNQUALIFIED',
        metrics=metrics,truthful_probability_shifts=shifts,parameter_updates=0,paper_green_light=False)


def run(root,config):
    import torch
    import transformers
    from transformers import AutoTokenizer,AutoModelForCausalLM
    template=published_template(config)
    root.mkdir(parents=True,exist_ok=False)
    def write(name,obj):
        with (root/name).open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)
    data=dataset();write('cases.json',data)
    write('spec.json',dict(model=MODEL,revision=REVISION,seed=90415001,scope='Two candidate semantic continuations; normalized summed token log likelihood; no EOS, length averaging, or generation',
        candidates='Explicitly constrained deployment action space, not free-form generation accuracy',
        template=template,template_source_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        qualification='Both direct and hindsight >=90% in each split, each domain >=75%',
        automatic_expansion=False,parameter_updates=0))
    (root/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(4);torch.manual_seed(90415001)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,local_files_only=True,
        dtype=torch.float32,attn_implementation='eager').eval()
    write('runtime.json',dict(torch=torch.__version__,transformers=transformers.__version__))
    rows=[];seen=set();forwards=0;start=time.monotonic()
    try:
        for c in data:
            candidates=[tok.encode(o,add_special_tokens=False) for o in c['options']]
            if any(not v for v in candidates) or any(candidates[i]==candidates[1-i][:len(candidates[i])] for i in (0,1)):
                raise ValueError('Candidate token strings must be nonempty and prefix-free')
            kinds=['direct','hindsight']
            if c['base_id'] not in seen:kinds=['base']+kinds;seen.add(c['base_id'])
            for kind in kinds:
                feedback=f'My actual preference is {c["preferred"]}. Please use that preference.'
                text=c['prompt']+('' if kind=='base' else '\n'+feedback if kind=='direct' else template.format(follow_up=feedback))
                prompt=tok.apply_chat_template([dict(role='user',content=text)],tokenize=True,
                    add_generation_prompt=True,enable_thinking=False,return_dict=False)
                scores=[];per_token=[]
                for ids in candidates:
                    inputs=prompt+ids[:-1]
                    if not prompt or len(inputs)>512:raise ValueError('No truncation')
                    with torch.no_grad():
                        h=model.model(input_ids=torch.tensor([inputs]),use_cache=False).last_hidden_state[0,len(prompt)-1:]
                        z=model.get_output_embeddings()(h).double()
                        lp=z.log_softmax(-1)[torch.arange(len(ids)),torch.tensor(ids)]
                    forwards+=1;scores.append(float(lp.sum()));per_token.append(lp.tolist())
                p=distribution(scores)
                rows.append(dict(id=c['id'],base_id=c['base_id'],domain=c['domain'],split=c['split'],
                    kind=kind,target=None if kind=='base' else c['target'],options=c['options'],
                    text=text,prompt_tokens=prompt,candidate_tokens=candidates,token_logprobabilities=per_token,
                    logprobabilities=scores,probabilities=p,prediction=0 if p[0]>=p[1] else 1,
                    candidate_prefix_mass=sum(math.exp(s) for s in scores)))
            if len(rows)%40==0:print(json.dumps(dict(scored_prompts=len(rows),sequence_forwards=forwards,elapsed=time.monotonic()-start)),flush=True)
        result=summarize(rows);result.update(scored_prompts=len(rows),sequence_forwards=forwards,elapsed=time.monotonic()-start)
        write('rows.json',rows);write('RESULT.json',result)
        write('MANIFEST.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()})
        print(json.dumps(result),flush=True)
    except Exception as error:
        write('partial_rows.json',rows);write('FAILED.json',dict(error=type(error).__name__,message=str(error),sequence_forwards=forwards))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--upstream-config',type=Path,required=True)
    a=p.parse_args();run(a.root,a.upstream_config)
