"""Frozen synthetic temporal extraction feasibility, no security inputs."""
import argparse
import itertools
import json
from pathlib import Path
import random
import subprocess
import time
from run_unexplored_screens import dump,sha

RULE=('A laboratory records a numeric setting. Each event replaces the previous setting. '
      'Only the stated before/after relations constrain chronology. Record display order carries no time information. '
      'Consider every chronology consistent with those relations. Answer YES if the final setting is at least the threshold in every chronology, '
      'NO if below it in every chronology, and CLARIFY otherwise. Conflicting values alone do not require clarification.')


def solve(values,edges,threshold):
    if not 1<=len(values)<=3 or any(type(v) is not int for v in values):raise ValueError('invalid values')
    if any(len(e)!=2 or any(type(v) is not int or not 0<=v<len(values) for v in e) for e in edges):raise ValueError('invalid edges')
    answers=set()
    for order in itertools.permutations(range(len(values))):
        position={v:i for i,v in enumerate(order)}
        if all(position[a]<position[b] for a,b in edges):answers.add(values[order[-1]]>=threshold)
    if not answers:raise ValueError('cycle')
    return 'CLARIFY' if len(answers)>1 else ('YES' if True in answers else 'NO')


def parse(text):
    text=text.strip()
    if text.startswith('```json\n') and text.endswith('\n```'):text=text[8:-4]
    elif text.startswith('```\n') and text.endswith('\n```'):text=text[4:-4]
    try:return json.loads(text)
    except ValueError:return None


def inputs():
    rng=random.Random(2026091091);rows=[]
    for i in range(24):
        threshold=rng.randrange(30,100);kind=i%4
        values=[threshold-5,threshold+3,threshold+7]
        edges=[[0,1],[1,2]] if kind==0 else [[0,1],[0,2]]
        if kind==2:values[1]=threshold-3
        if kind==3:values=[threshold+2,threshold-4];edges=[[0,1]]
        for presentation in range(2):
            order=list(range(len(values)));rng.shuffle(order)
            records=[f'Event {j} set the value to {values[j]}.' for j in order]
            relations=[f'Event {b} occurred after event {a}.' for a,b in edges]
            text='\n'.join(records+relations)+f'\nThreshold: {threshold}.'
            rows.append({'id':f'{i}_{presentation}','base':i,'mechanism':kind,'presentation':presentation,
                         'values':values,'edges':edges,'threshold':threshold,'text':text,'gold':solve(values,edges,threshold)})
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);rows=inputs();dump(a.out/'INPUTS.json',rows)
    dump(a.out/'PROTOCOL.json',{'scope':'Synthetic DEV apparatus;24 parameter cases of4 mechanisms, not24 independent mechanisms or a paper claim',
        'source_sha256':sha(Path(__file__)),'gate':'parse and EOS >=.95 each arm; extracted graph semantic accuracy >=.95; extraction+solver action accuracy at least .10 above direct explicit-rule baseline; otherwise stop',
        'sampling':'greedy nonthinking native chat, max192 tokens; exact outer JSON fence supported prospectively',
        'followup':'Positive only admits independently sourced benign ambiguity data and second-family test; no automatic training',
        'estimate':'5-15 minutes including load unbenchmarked; no midrun deadline'})
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,GenerationConfig
    tok=AutoTokenizer.from_pretrained(a.snapshot,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(a.snapshot,local_files_only=True,dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa').eval();model.requires_grad_(False)
    dump(a.out/'MODEL.json',{'snapshot':str(a.snapshot),'files':{p.name:sha(p) for p in a.snapshot.iterdir() if p.is_file() and p.suffix in {'.json','.safetensors'}}})
    config=GenerationConfig(do_sample=False,max_new_tokens=192,eos_token_id=tok.eos_token_id,pad_token_id=tok.eos_token_id,use_cache=True)
    started=time.monotonic()
    with (a.out/'OUTPUTS.jsonl').open('x') as f,torch.inference_mode():
        for row in rows:
            for arm in ['direct','extract']:
                instruction=RULE+'\n'+('Return only JSON with the one key answer and value YES, NO, or CLARIFY.' if arm=='direct' else
                    'Extract only the stated facts. Return JSON with exactly two keys: values (integer array in event ID order) and edges (pairs [earlier event ID, later event ID]). Do not add unstated order from display position. Include every explicitly stated relation.')
                rendered=tok.apply_chat_template([{'role':'system','content':instruction},{'role':'user','content':row['text']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                encoded=tok(rendered,return_tensors='pt',add_special_tokens=False).to('cuda')
                ids=model.generate(**encoded,generation_config=config)[0,encoded.input_ids.shape[1]:].tolist()
                text=tok.decode(ids,skip_special_tokens=True)
                f.write(json.dumps({'id':row['id'],'arm':arm,'rendered':rendered,'input_ids':encoded.input_ids[0].tolist(),'output_ids':ids,'text':text,'parsed':parse(text),'eos':bool(ids and ids[-1]==tok.eos_token_id)})+'\n');f.flush()
            print(json.dumps({'completed':row['id'],'seconds':time.monotonic()-started}),flush=True)
    dump(a.out/'TIMING.json',{'inference_seconds':time.monotonic()-started})
    dump(a.out/'MANIFEST.json',{p.name:sha(p) for p in a.out.iterdir() if p.is_file()})


if __name__=='__main__':main()
