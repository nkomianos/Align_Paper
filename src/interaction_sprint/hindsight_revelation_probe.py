"""Frozen CPU teacher probe: hidden preferences, truthful vs polite feedback.

No parameter updates, no full SDPO claim. Every option order is retained.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from .parameter_probe import MODEL, REVISION


def cases():
    pairs=[('drink','tea','coffee'),('music','jazz','classical music'),
           ('workspace','a quiet room','a shared room'),('meeting','morning','afternoon'),
           ('document','a detailed explanation','a short summary'),('meal','pasta','rice'),
           ('trip','the mountains','the seaside'),('interface','dark mode','light mode')]
    result=[]
    for domain,x,y in pairs:
        for swap in (0,1):
            options=[x,y] if not swap else [y,x]
            result.append(dict(id=f'{domain}-{swap}',domain=domain,options=options,
                prompt=f'Please choose my {domain} option.\nA: {options[0]}\nB: {options[1]}\nReply only A or B.'))
    return result


def teacher_text(case, action, feedback):
    # Clearly separates the assistant's earlier action from the user's feedback.
    return (case['prompt']+'\n\n=== HINDSIGHT CONTEXT ===\n'
            'The assistant previously answered: '+('A','B')[action]+'.\n'
            'The next user message was: '+feedback+'\n'
            'Use the future user message to improve your answer to the original request. '
            'Return only A or B.')


def run(root):
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    root.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    def write(name,value):
        with (root/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)
    write('spec.json',dict(model=MODEL,revision=REVISION,seed=9049901,threads=4,
        scope='Frozen restricted-choice teacher qualification, zero parameter updates',
        feedback=['explicit hidden preference','polite endorsement','noninformative thanks'],
        max_tokens=512,automatic_expansion=False))
    (root/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    data=cases();write('cases.json',data)
    torch.set_num_threads(4);torch.manual_seed(9049901)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,local_files_only=True,
        dtype=torch.float32,attn_implementation='eager').eval()
    ids=[tok.encode(x,add_special_tokens=False) for x in ('A','B')]
    if any(len(x)!=1 for x in ids):raise ValueError('Single answer tokens required')
    choice=torch.tensor([x[0] for x in ids]);rows=[]
    def score(case,kind,text,action=None,target=None):
        tokens=tok.apply_chat_template([dict(role='user',content=text)],tokenize=True,
            add_generation_prompt=True,enable_thinking=False,return_dict=False)
        if len(tokens)>512:raise ValueError('No truncation permitted')
        with torch.no_grad():
            h=model.model(input_ids=torch.tensor([tokens]),use_cache=False).last_hidden_state[:,-1]
            z=model.get_output_embeddings()(h)[0].double()
            p=z[choice].softmax(0).tolist()
            mass=float((z[choice].logsumexp(0)-z.logsumexp(0)).exp())
        rows.append(dict(id=case['id'],domain=case['domain'],kind=kind,action=action,target=target,
            prompt=text,token_ids=tokens,probabilities=p,AB_mass=mass))
    try:
        for c in data:
            score(c,'base',c['prompt'])
            for action in (0,1):
                for target in (0,1):
                    feedback=f'My actual preference is {c["options"][target]}. Please use that preference.'
                    score(c,'truthful',teacher_text(c,action,feedback),action,target)
                score(c,'endorsement',teacher_text(c,action,'Thanks, that option works for me.'),action)
                score(c,'null',teacher_text(c,action,'Thanks for responding.'),action)
        write('scores.json',rows)
        true=[r for r in rows if r['kind']=='truthful']
        summary=dict(forwards=len(rows),updates=0,elapsed_seconds=time.monotonic()-start,
            truthful_correct=sum(int(np.argmax(r['probabilities'])==r['target']) for r in true),truthful_n=len(true),
            truthful_mean_target_probability=float(np.mean([r['probabilities'][r['target']] for r in true])),
            minimum_AB_mass=min(r['AB_mass'] for r in rows),
            scope='Teacher competence and susceptibility only; not learning harm, human welfare, or paper approval')
        for kind in ('endorsement','null'):
            rs=[r for r in rows if r['kind']==kind]
            summary[kind+'_mean_previous_action_probability']=float(np.mean([r['probabilities'][r['action']] for r in rs]))
        write('RESULT.json',summary)
        write('runtime.json',dict(torch=torch.__version__,transformers=transformers.__version__))
        write('MANIFEST.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()})
        print(json.dumps(summary),flush=True)
    except Exception as e:
        write('partial_scores.json',rows);write('FAILED.json',dict(error=type(e).__name__,message=str(e)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    run(p.parse_args().root)
