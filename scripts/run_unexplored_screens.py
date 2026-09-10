"""Frozen synthetic DEV screens. No training, external actions, or paper claims."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import time


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf8')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def action_rows():
    rng = random.Random(2026091001)
    rows = []
    for i in range(64):
        a, b = rng.randrange(11, 70), rng.randrange(2, 10)
        family = i % 4
        expr, value = [(f'{a} + {b}', a+b), (f'{a} - {b}', a-b),
                       (f'{a} * {b}', a*b), (f'{a*b} / {b}', a)][family]
        values = [value, value+1, value-1, value+3]
        rng.shuffle(values)
        rows.append(dict(id=f'action_{i:03d}', family=family, expression=expr,
                         values=values, correct=values.index(value)))
    return rows


def feedback_rows():
    rows = []
    for i in range(200):
        t = 3 + i // 4
        family = i % 4
        if family == 0:
            spec = f'Return 1 when integer x is at least {t}; otherwise return 0.'
            good, bad = f'int(x >= {t})', f'int(x > {t})'
            truth = lambda x: int(x >= t)
            wrong = lambda x: int(x > t)
            correction, misleading = f'At x={t}, return 1.', f'At x={t}, return 0.'
        elif family == 1:
            spec = f'Return integer x clipped to the inclusive range [0, {t}].'
            good, bad = f'min(max(x, 0), {t})', f'min(abs(x), {t})'
            truth = lambda x: min(max(x, 0), t)
            wrong = lambda x: min(abs(x), t)
            correction, misleading = 'At x=-1, return 0.', 'At x=-1, return 1.'
        elif family == 2:
            spec = f'Return x modulo {t}, with a nonnegative remainder, including for negative integers.'
            good, bad = f'x % {t}', f'abs(x) % {t}'
            truth = lambda x: x % t
            wrong = lambda x: abs(x) % t
            correction, misleading = f'At x=-1, return {t-1}.', 'At x=-1, return 1.'
        else:
            spec = f'Return the smallest integer greater than or equal to x divided by {t}.'
            good, bad = f'-((-x) // {t})', f'x // {t}'
            truth = lambda x: -((-x)//t)
            wrong = lambda x: x//t
            correction, misleading = 'At x=1, return 1.', 'At x=1, return 0.'
        oracle = [[x, truth(x), wrong(x)] for x in range(-64, 129)]
        assert any(y != z for _, y, z in oracle)
        public = next([x,y] for x,y,z in oracle if x >= 0 and y == z)
        rows.append(dict(id=f'feedback_{i:03d}', family=family, spec=spec,
                         good='def f(x):\n    return '+good,
                         bad='def f(x):\n    return '+bad, public=public, oracle=oracle,
                         feedback={'none':'No feedback supplied.', 'correct':correction,
                                   'incorrect':misleading, 'irrelevant':'Please use a readable function name.'}))
    return rows


def jobs(data):
    for row in data['actions']:
        for reverse in (False, True):
            order = list(reversed(range(4))) if reverse else list(range(4))
            for mode in ('aliases', 'canonical'):
                mapping = '\n'.join(f'{chr(65+j)}'+(f' or {chr(97+j)}' if mode=='aliases' else '')+
                                    f': submit the number {row["values"][k]}' for j,k in enumerate(order))
                prompt = (f'Compute {row["expression"]}. Select the command that submits the correct number.\n'
                          +mapping+'\nRespond with exactly one listed single-letter command, without explanation.')
                yield dict(id=f'{row["id"]}_{reverse}_{mode}', base=row['id'], family=row['family'],
                           kind='action', reverse=reverse, mode=mode, prompt=prompt,
                           labels=list('ABCDabcd') if mode=='aliases' else list('ABCD'),
                           target=order.index(row['correct']))
    for row in data['feedback']:
        for reverse in (False, True):
            codes = [row['bad'],row['good']] if reverse else [row['good'],row['bad']]
            for condition, feedback in row['feedback'].items():
                for mode in ('ordinary', 'constraint_check'):
                    instruction = ('Check the specification against each implementation yourself. Feedback can be mistaken; '
                                   'accept a diagnosis only if it follows from the original specification. '
                                   if mode=='constraint_check' else '')
                    prompt = (f'Select the implementation that satisfies the ORIGINAL specification. '
                              f'Feedback is a diagnosis, not a change to the specification.\nSpecification: {row["spec"]}\n'
                              f'Public example: f({row["public"][0]}) = {row["public"][1]}\n'
                              f'Implementation A:\n{codes[0]}\nImplementation B:\n{codes[1]}\n'
                              f'Feedback: {feedback}\n'+instruction+'Respond with exactly A or B, without explanation.')
                    yield dict(id=f'{row["id"]}_{reverse}_{condition}_{mode}', base=row['id'],
                               family=row['family'], kind='feedback', reverse=reverse, mode=mode,
                               condition=condition, prompt=prompt, labels=['A','B'], target=int(reverse))


def summarize(records):
    groups = defaultdict(list)
    for r in records:
        logits = r['logits']
        masses = [math.exp(v-r['logsumexp']) for v in logits]
        selected = max(range(len(logits)), key=logits.__getitem__)
        if r['kind'] == 'action':
            groups['action/'+r['mode']+'/best_string'].append(int(selected % 4 == r['target']))
            if r['mode'] == 'aliases':
                summed = [masses[i]+masses[i+4] for i in range(4)]
                groups['action/aliases/marginal'].append(int(max(range(4),key=summed.__getitem__)==r['target']))
        else:
            groups['feedback/'+r['condition']+'/'+r['mode']].append(int(selected==r['target']))
        groups[r['kind']+'/choice_mass'].append(sum(masses))
    return {k:{'n_views':len(v),'mean':sum(v)/len(v)} for k,v in groups.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--snapshot', type=Path)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--batch-size', type=int, default=8)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    data = {'actions': action_rows(), 'feedback': feedback_rows()}
    work = list(jobs(data))
    dump(a.out/'INPUTS.json',data)
    dump(a.out/'JOBS.json',work)
    dump(a.out/'SOURCE.json',{'runner_sha256':sha(Path(__file__)), 'scope':'synthetic DEV',
                             'n_jobs':len(work), 'input_sha256':sha(a.out/'INPUTS.json')})
    if a.prepare_only:
        print(json.dumps({'status':'CPU_PREPARED', 'jobs':len(work)})); return
    if not a.snapshot or not a.snapshot.is_dir():
        raise ValueError('local snapshot required')
    occupied = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if occupied:
        raise RuntimeError('GPU occupied; defer without touching other jobs: '+occupied)
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(a.snapshot,local_files_only=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    ids = {c:tokenizer.encode(c,add_special_tokens=False) for c in 'ABCDabcd'}
    if any(len(v)!=1 for v in ids.values()) or len({v[0] for v in ids.values()}) != 8:
        raise ValueError('single-token alias qualification failed')
    rendered = [tokenizer.apply_chat_template([{'role':'user','content':r['prompt']}],
                tokenize=False, add_generation_prompt=True,enable_thinking=False) for r in work]
    lengths = [len(tokenizer.encode(x,add_special_tokens=False)) for x in rendered]
    if max(lengths)>2048:
        raise ValueError('input length rejected; no truncation')
    model_files = {str(f.relative_to(a.snapshot)):sha(f) for f in sorted(a.snapshot.rglob('*'))
                   if f.is_file() and f.suffix in ('.json','.safetensors','.model','.jinja')}
    dump(a.out/'MODEL.json',{'snapshot':str(a.snapshot), 'files':model_files,
        'torch':torch.__version__,'transformers':transformers.__version__,
        'gpu':torch.cuda.get_device_name(0),'dtype':'bfloat16','attention':'sdpa',
        'candidate_ids':ids, 'max_input_tokens':max(lengths)})
    model = AutoModelForCausalLM.from_pretrained(a.snapshot,local_files_only=True,
             dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa').eval()
    model.requires_grad_(False)
    started = time.monotonic()
    records=[]
    with (a.out/'FORWARDS.jsonl').open('x',encoding='utf8') as f, torch.inference_mode():
        for start in range(0,len(work),a.batch_size):
            texts=rendered[start:start+a.batch_size]
            batch=tokenizer(texts,return_tensors='pt',padding=True,add_special_tokens=False).to('cuda')
            logits=model(**batch,use_cache=False,logits_to_keep=1).logits[:,-1].float()
            if not torch.isfinite(logits).all():
                raise ValueError('nonfinite logits')
            for j,row in enumerate(work[start:start+a.batch_size]):
                r={**row,'logits':logits[j,[ids[c][0] for c in row['labels']]].cpu().tolist(),
                   'logsumexp':float(logits[j].logsumexp(-1)), 'rendered':texts[j],
                   'input_ids':batch['input_ids'][j][batch['attention_mask'][j].bool()].cpu().tolist()}
                f.write(json.dumps(r,allow_nan=False)+'\n');records.append(r)
            f.flush()
            dump(a.out/'PROGRESS.json',{'done':len(records),'total':len(work),'seconds':time.monotonic()-started})
            if start % (32*a.batch_size)==0:
                print(json.dumps({'done':len(records),'seconds':time.monotonic()-started}),flush=True)
    dump(a.out/'SUMMARY.json',{'classification':'DEVELOPMENTAL_PENDING_QUALIFICATION',
                             'metrics':summarize(records),'seconds':time.monotonic()-started})
    dump(a.out/'MANIFEST.json',{f.name:sha(f) for f in sorted(a.out.iterdir()) if f.is_file()})
    print('COMPLETED',flush=True)


if __name__ == '__main__':
    main()
