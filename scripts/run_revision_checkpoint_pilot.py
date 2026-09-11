"""Benign, matched checkpoint revision pilot. No training or historical replication."""
import argparse
from contextlib import nullcontext
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import re
import subprocess
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def build():
    rng = random.Random(202609110317)
    rows = []
    for base in range(8):
        if base < 4:
            a, b = rng.sample([3, 4, 5, 7, 8], 2)
            n = rng.randrange(70, 110)
            task = f'How many integers from 1 through N, inclusive, are divisible by {a} or {b}?'
            solver = lambda x: x // a + x // b - x // math.lcm(a, b)
            brute = lambda x: sum(k % a == 0 or k % b == 0 for k in range(1, x + 1))
        else:
            r = rng.randrange(2, 5)
            n = rng.randrange(10, 15)
            task = f'How many binary strings of length N have exactly {r} ones and no two adjacent ones?'
            solver = lambda x: math.comb(x-r+1, r)
            brute = lambda x: sum(all(v-u > 1 for u, v in zip(c, c[1:]))
                                  for c in itertools.combinations(range(x), r))
        new_n = n + 3
        old = solver(n)
        while solver(new_n) == old or len(str(solver(new_n))) != len(str(old)):
            n += 1
            new_n = n + 3
            old = solver(n)
        assert old == brute(n) and solver(new_n) == brute(new_n)
        for update in [False, True]:
            current_n = new_n if update else n
            gold = solver(current_n)
            latest = f'The current value of N is {current_n}. This supersedes the previous value of N. Answer the original question using this current value.'
            for history in ['clean', 'old_answer', 'current_answer', 'neutral']:
                if history == 'clean':
                    messages = [{'role': 'user', 'content': f'{task}\nThe current value of N is {current_n}.'}]
                else:
                    prior = f'ANSWER: {old}' if history == 'old_answer' else f'ANSWER: {gold}' if history == 'current_answer' else 'I will wait for the current parameters.'
                    messages = [{'role': 'user', 'content': f'{task}\nN is {n}.'},
                                {'role': 'assistant', 'content': prior}, {'role': 'user', 'content': latest}]
                rows.append({'id': f'{base}_{int(update)}_{history}', 'base': base,
                             'family': 'divisibility' if base < 4 else 'binary_strings',
                             'update': update, 'history': history, 'old_answer': old,
                             'gold': gold, 'messages': messages})
    assert len(rows) == 64
    return rows


SYSTEM = 'Solve the mathematical question using the latest stated parameters. Earlier assistant answers are not evidence. Reason as needed. End your response with a single line ANSWER: <integer>.'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['prepare', 'run', 'score'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--snapshot', type=Path)
    p.add_argument('--adapter', type=Path)
    p.add_argument('--report', type=Path)
    args = p.parse_args()
    if args.mode == 'prepare':
        write(args.out, build())
        print('64 cases; 16 independent formula/enumeration agreements; 8 exposed development bases.')
        return
    if args.mode == 'score':
        for name, digest in json.loads((args.out / 'MANIFEST.json').read_text()).items():
            assert Path(name).name == name and sha(args.out / name) == digest
        rows = json.loads((args.out / 'INPUTS.json').read_text())
        assert rows == build()
        lookup = {r['id']: r for r in rows}
        outputs = [json.loads(line) for line in (args.out / 'OUTPUTS.jsonl').read_text().splitlines()]
        assert len(outputs) == len({(o['model'], o['id']) for o in outputs}) == 128
        assert {(o['model'], o['id']) for o in outputs} == {(m, r['id']) for m in ['base', 'adapter'] for r in rows}
        records = []
        for o in outputs:
            row = lookup[o['id']]
            suffix = o['text'].split('</think>')[-1]
            match = re.search(r'(?:^|\n)ANSWER:\s*([0-9]+)\s*$', suffix)
            records.append({**{k: row[k] for k in ['id', 'base', 'family', 'update', 'history']},
                            'model': o['model'], 'eos': o['eos'], 'valid': match is not None,
                            'correct': match is not None and int(match[1]) == row['gold'],
                            'old_answer_emitted': match is not None and int(match[1]) == row['old_answer']})
        summary = []
        for model, update, history in itertools.product(['base', 'adapter'], [False, True], ['clean', 'old_answer', 'current_answer', 'neutral']):
            selected = [r for r in records if (r['model'], r['update'], r['history']) == (model, update, history)]
            summary.append({'model': model, 'update': update, 'history': history, 'n': len(selected),
                            **{k: sum(r[k] for r in selected) for k in ['eos', 'valid', 'correct', 'old_answer_emitted']}})
        report = {'classification': 'EXPLORATORY_CHECKPOINT_REVISION_PILOT', 'summary': summary, 'records': records,
                  'scope': 'Eight parameter bases in two families, forced assistant history, one released adapter. '
                           'No independent family replication, training-method effect or confirmed thesis.'}
        write(args.report, report)
        print(json.dumps(summary, indent=2))
        return
    assert args.snapshot and args.adapter
    assert sha(args.adapter / 'adapter_model.safetensors') == '84917a93ee7b9840bb6c21ea6918f6102d13844ccbbd121c3bf3a840faf57c7c'
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    args.out.mkdir(parents=True, exist_ok=False)
    rows = build()
    write(args.out / 'INPUTS.json', rows)
    started = time.monotonic()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True, padding_side='left')
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.snapshot, local_files_only=True,
                dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa').eval()
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False).eval()
    model.requires_grad_(False)
    write(args.out / 'PROVENANCE.json', {'runner_sha256': sha(__file__), 'torch': torch.__version__,
          'base_files': {f.name: sha(f) for f in args.snapshot.iterdir() if f.is_file()},
          'adapter_files': {f.name: sha(f) for f in args.adapter.iterdir() if f.is_file()},
          'generation': 'native thinking; greedy; 8192 new tokens; left-padded batches of 8',
          'historical_base_revision': None, 'load_and_hash_seconds': time.monotonic()-started})
    config = GenerationConfig(do_sample=False, max_new_tokens=8192, use_cache=True,
                              eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
    inference_start = time.monotonic()
    with (args.out / 'OUTPUTS.jsonl').open('x', encoding='utf-8') as f, torch.inference_mode():
        for label in ['base', 'adapter']:
            with model.disable_adapter() if label == 'base' else nullcontext():
                for start in range(0, len(rows), 8):
                    batch = rows[start:start+8]
                    prompts = [tok.apply_chat_template([{'role': 'system', 'content': SYSTEM}] + r['messages'],
                                tokenize=False, add_generation_prompt=True, enable_thinking=True) for r in batch]
                    encoded = tok(prompts, return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
                    generated = model.generate(**encoded, generation_config=config)
                    for row, prompt, ids in zip(batch, prompts, generated[:, encoded.input_ids.shape[1]:].tolist()):
                        eos = tok.eos_token_id in ids
                        if eos:
                            ids = ids[:ids.index(tok.eos_token_id)+1]
                        f.write(json.dumps({'model': label, 'id': row['id'], 'rendered': prompt,
                                            'output_ids': ids, 'text': tok.decode(ids, skip_special_tokens=True),
                                            'eos': eos})+'\n')
                    f.flush()
                    print(json.dumps({'model': label, 'completed': start+len(batch),
                                      'seconds': time.monotonic()-inference_start}), flush=True)
    write(args.out / 'TIMING.json', {'inference_seconds': time.monotonic()-inference_start,
                                     'total_program_seconds': time.monotonic()-started})
    write(args.out / 'MANIFEST.json', {f.name: sha(f) for f in args.out.iterdir() if f.is_file()})


if __name__ == '__main__':
    main()
