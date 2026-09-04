"""Frozen CPU teacher probe: hidden preferences, truthful vs polite feedback.

No parameter updates, no full SDPO claim. Every option order is retained.
"""
import argparse
import ast
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


def history_text(case, target, mode):
    if mode not in ('shown', 'omitted', 'redacted'):
        raise ValueError('Unknown history mode')
    action = 1-target
    feedback = f'My actual preference is {case["options"][target]}. Please use that preference.'
    text = teacher_text(case, action, feedback)
    line = 'The assistant previously answered: '+('A','B')[action]+'.\n'
    if mode == 'omitted':
        return text.replace(line, '')
    if mode == 'redacted':
        return text.replace(line, 'The assistant previously answered: [redacted].\n')
    return text


def fresh_cases():
    pairs=[('card shape','a circle','a square'),('wallpaper color','blue','green'),
           ('font','serif','sans serif'),('folder name','alpha','omega'),
           ('pet name','Coco','Pip'),('calendar layout','weekly','monthly'),
           ('icon','a star','a heart'),('notification sound','a bell','a chime')]
    result=[]
    for i,(domain,x,y) in enumerate(pairs):
        for swap in (0,1):
            options=[x,y] if swap==0 else [y,x]
            result.append(dict(id=f'fresh-{i}-{swap}',domain=domain,options=options,
                prompt=f'Select a {domain} option for me.\nA: {options[0]}\nB: {options[1]}\nReply only A or B.'))
    return result


def published_template(path):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!='bc835f355c63e00015fdbe396830aff210dcf6efac22cbb91a89e1214e92d2ec':
        raise ValueError('Unpinned upstream config')
    nodes=[n for n in ast.walk(ast.parse(raw)) if isinstance(n,ast.AnnAssign)
           and isinstance(n.target,ast.Name) and n.target.id=='hindsight_block_template']
    if len(nodes)!=1:raise ValueError('Template definition ambiguous')
    return ast.literal_eval(nodes[0].value)


def published_text(case,target,mode,template):
    feedback=f'My actual preference is {case["options"][target]}. Please use that preference.'
    if mode=='published':return case['prompt']+template.format(follow_up=feedback.strip())
    if mode=='direct':return case['prompt']+'\n'+feedback
    if mode=='redacted':return history_text(case,target,'redacted')
    raise ValueError('Unknown published-control mode')


def run(root, history_control=False, published_control=False, upstream_config=None):
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    root.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    def write(name,value):
        with (root/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)
    if history_control and published_control:raise ValueError('Select one control')
    template=published_template(upstream_config) if published_control else None
    write('spec.json',dict(model=MODEL,revision=REVISION,seed=9049901,threads=4,
        scope='Frozen restricted-choice teacher qualification, zero parameter updates',
        feedback=['explicit hidden preference','polite endorsement','noninformative thanks'],
        max_tokens=512,history_control=history_control,published_control=published_control,
        published_template=template,automatic_expansion=False))
    (root/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    data=fresh_cases() if published_control else cases();write('cases.json',data)
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
            if published_control:
                score(c,'base',c['prompt'])
                score(c,'published_null',c['prompt']+template.format(follow_up='Thanks for responding.'))
                for target in (0,1):
                    for mode in ('published','direct','redacted'):
                        score(c,mode,published_text(c,target,mode,template),None,target)
                continue
            if history_control:
                for target in (0,1):
                    for mode in ('shown','omitted','redacted'):
                        score(c,mode,history_text(c,target,mode),1-target,target)
                continue
            score(c,'base',c['prompt'])
            for action in (0,1):
                for target in (0,1):
                    feedback=f'My actual preference is {c["options"][target]}. Please use that preference.'
                    score(c,'truthful',teacher_text(c,action,feedback),action,target)
                score(c,'endorsement',teacher_text(c,action,'Thanks, that option works for me.'),action)
                score(c,'null',teacher_text(c,action,'Thanks for responding.'),action)
        write('scores.json',rows)
        summary=dict(forwards=len(rows),updates=0,elapsed_seconds=time.monotonic()-start)
        if history_control or published_control:
            summary['arms']={}
            for mode in (('published','direct','redacted') if published_control else ('shown','omitted','redacted')):
                rs=[r for r in rows if r['kind']==mode]
                summary['arms'][mode]=dict(n=len(rs),
                    correct=sum(int(np.argmax(r['probabilities'])==r['target']) for r in rs),
                    mean_target_probability=float(np.mean([r['probabilities'][r['target']] for r in rs])))
            summary['scope']=('Fresh preference teacher qualification; published prompt only, not full SDPO learning'
                              if published_control else 'Paired diagnostic on previous cases, not independent confirmation or learning')
        else:
            true=[r for r in rows if r['kind']=='truthful']
            summary.update(dict(
            truthful_correct=sum(int(np.argmax(r['probabilities'])==r['target']) for r in true),truthful_n=len(true),
            truthful_mean_target_probability=float(np.mean([r['probabilities'][r['target']] for r in true])),
            minimum_AB_mass=min(r['AB_mass'] for r in rows),
            scope='Teacher competence and susceptibility only; not learning harm, human welfare, or paper approval'))
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
    p.add_argument('--history-control',action='store_true')
    p.add_argument('--published-control',action='store_true');p.add_argument('--upstream-config',type=Path)
    args=p.parse_args()
    run(args.root,args.history_control,args.published_control,args.upstream_config)
