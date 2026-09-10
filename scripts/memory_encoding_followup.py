"""Fresh developmental codec comparison; never repairs historical predictions."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
import subprocess
import time

ARMS = ('reasoning', 'pairs', 'named', 'reverse_pairs')
RULE = ('Each event replaces a numeric laboratory setting. Display order is not chronology. '
        'Only explicit relations constrain chronology. Consider every consistent chronology. '
        'Return YES if the final value is at least the threshold in all of them, NO if below '
        'in all of them, and CLARIFY otherwise.')
INSTRUCTIONS = {
    'reasoning': RULE + ' Return JSON with reasoning (at most 60 words) and answer.',
    'pairs': 'Extract stated facts only. Return JSON with values in event ID order and edges '
             'as [earlier event ID, later event ID] pairs. Include all explicit relations only.',
    'named': 'Extract stated facts only. Return JSON with values in event ID order and edges '
             'as objects with keys earlier and later, each holding an event ID. Include all explicit relations only.',
    'reverse_pairs': 'Extract stated facts only. Return JSON with values in event ID order and edges '
                     'as [later event ID, earlier event ID] pairs. Include all explicit relations only.',
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')

def orders(n, edges):
    return {p for p in itertools.permutations(range(n))
            if all(p.index(a) < p.index(b) for a, b in edges)}

def decision(values, edges, threshold):
    endings = {values[p[-1]] >= threshold for p in orders(len(values), edges)}
    if not endings:
        raise ValueError('cycle')
    return 'CLARIFY' if len(endings) == 2 else 'YES' if True in endings else 'NO'

def prepare():
    rng = random.Random(202609101727)
    rows = []
    for base in range(24):
        n = 3 + base % 3
        chronology = list(range(n))
        rng.shuffle(chronology)
        edges = [[chronology[i], chronology[j]] for i in range(n) for j in range(i+1,n)
                 if rng.random() < .38]
        threshold = rng.randrange(200, 800)
        values = [threshold + rng.choice((-13, -7, 4, 11)) for _ in range(n)]
        for view in range(2):
            display = list(range(n))
            rng.shuffle(display)
            relations = [f'Event {b} occurred after event {a}.' for a,b in edges]
            rng.shuffle(relations)
            text = '\n'.join([f'Event {i} set the value to {values[i]}.' for i in display] + relations
                             + [f'Threshold: {threshold}.'])
            rows.append(dict(id=f'{base}_{view}', base=base, view=view, values=values,
                             edges=edges, threshold=threshold, text=text,
                             gold=decision(values,edges,threshold)))
    return rows

def parse(text):
    text = text.strip()
    if text.startswith('```json\n') and text.endswith('\n```'):
        text = text[8:-4]
    elif text.startswith('```\n') and text.endswith('\n```'):
        text = text[4:-4]
    try:
        return json.loads(text)
    except ValueError:
        return None

def decode(value, arm, n):
    if not isinstance(value, dict) or set(value) != {'values','edges'}:
        raise ValueError('schema')
    values, raw = value['values'], value['edges']
    if not isinstance(values,list) or len(values)!=n or any(type(x)!=int for x in values):
        raise ValueError('values')
    if not isinstance(raw,list):
        raise ValueError('edges')
    edges=[]
    for e in raw:
        if arm=='named':
            if not isinstance(e,dict) or set(e)!={'earlier','later'}:
                raise ValueError('named edge')
            e=[e['earlier'],e['later']]
        if not isinstance(e,list) or len(e)!=2 or any(type(x)!=int or not 0<=x<n for x in e):
            raise ValueError('edge')
        edges.append(e[::-1] if arm=='reverse_pairs' else e)
    return values, edges

def score(root):
    for name, digest in json.loads((root/'MANIFEST.json').read_text()).items():
        assert Path(name).name == name and sha(root/name)==digest
    rows=json.loads((root/'INPUTS.json').read_text())
    assert rows==prepare()
    byid={r['id']:r for r in rows}
    outputs=[json.loads(s) for s in (root/'OUTPUTS.jsonl').read_text().splitlines()]
    keys=[(o['id'],o['arm']) for o in outputs]
    assert len(keys)==len(set(keys))==192
    assert set(keys)=={(r['id'],a) for r in rows for a in ARMS}
    records=[]
    for o in outputs:
        r=byid[o['id']]; arm=o['arm']; v=parse(o['text'])
        valid=False; semantic=False; answer=None
        try:
            if arm=='reasoning':
                valid=isinstance(v,dict) and set(v)=={'reasoning','answer'} and isinstance(v['reasoning'],str) and v['answer'] in ('YES','NO','CLARIFY')
                answer=v['answer'] if valid else None
            else:
                values,edges=decode(v,arm,len(r['values']))
                answer=decision(values,edges,r['threshold']); valid=True
                semantic=values==r['values'] and orders(len(values),edges)==orders(len(values),r['edges'])
        except (ValueError,TypeError,KeyError):
            pass
        records.append(dict(id=o['id'],base=r['base'],arm=arm,valid=valid,eos=o['eos'],
                            semantic=semantic,correct=answer==r['gold']))
    metrics={a:{k:sum(x[k] for x in records if x['arm']==a)/48
                for k in ('valid','eos','semantic','correct')} for a in ARMS}
    dump(root/'VERIFIED.json',dict(classification='FRESH_DEVELOPMENTAL_CODEC_COMPARISON',
        metrics=metrics, records=records, unit='24 generated graph cases, two dependent views each; not natural tasks',
        qualified={a:metrics[a]['valid']>=.95 and metrics[a]['eos']>=.95 and
                   (a=='reasoning' or metrics[a]['semantic']>=.95) for a in ARMS},
        scope='No best-arm selection, historical repair, confirmation or training admission.'))
    print(json.dumps(metrics),flush=True)

def run(args):
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    args.out.mkdir(parents=True,exist_ok=False)
    rows=prepare(); dump(args.out/'INPUTS.json',rows)
    dump(args.out/'CONFIG.json',dict(source_sha256=sha(__file__),instructions=INSTRUCTIONS,
         arms=ARMS,max_new_tokens=192,model=str(args.snapshot),dtype='bfloat16',thinking=False))
    dump(args.out/'MODEL.json',{p.name:sha(p) for p in args.snapshot.iterdir()
         if p.is_file() and p.suffix in ('.json','.safetensors','.jinja')})
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,GenerationConfig
    started=time.monotonic()
    tok=AutoTokenizer.from_pretrained(args.snapshot,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(args.snapshot,local_files_only=True,dtype=torch.bfloat16,
        device_map={'':0},attn_implementation='sdpa').eval(); model.requires_grad_(False)
    loaded=time.monotonic()
    cfg=GenerationConfig(do_sample=False,max_new_tokens=192,eos_token_id=tok.eos_token_id,
                         pad_token_id=tok.eos_token_id,use_cache=True)
    with (args.out/'OUTPUTS.jsonl').open('x',encoding='utf-8') as f,torch.inference_mode():
        for r in rows:
            for arm in ARMS:
                prompt=tok.apply_chat_template([{'role':'system','content':INSTRUCTIONS[arm]},
                    {'role':'user','content':r['text']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                encoded=tok(prompt,add_special_tokens=False,return_tensors='pt').to('cuda')
                ids=model.generate(**encoded,generation_config=cfg)[0,encoded.input_ids.shape[1]:].tolist()
                f.write(json.dumps(dict(id=r['id'],arm=arm,prompt=prompt,input_ids=encoded.input_ids[0].tolist(),
                    output_ids=ids,text=tok.decode(ids,skip_special_tokens=True),eos=bool(ids and ids[-1]==tok.eos_token_id)))+'\n');f.flush()
            print(json.dumps(dict(completed=r['id'],seconds=time.monotonic()-loaded)),flush=True)
    dump(args.out/'TIMING.json',dict(load_seconds=loaded-started,inference_seconds=time.monotonic()-loaded))
    dump(args.out/'MANIFEST.json',{p.name:sha(p) for p in args.out.iterdir() if p.is_file()})
    score(args.out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','score'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--snapshot',type=Path)
    a=p.parse_args()
    if a.mode=='prepare':
        a.out.parent.mkdir(parents=True,exist_ok=True);dump(a.out,prepare())
    elif a.mode=='run':
        if a.snapshot is None:p.error('--snapshot required')
        run(a)
    else:score(a.out)
