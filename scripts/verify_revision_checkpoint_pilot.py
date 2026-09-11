"""Independent text-derived gold and frozen-gate replay for the revision pilot."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import re


RUNNER_SHA = '1661df546dd064b90badfe9ae704c2d45a5a748bdef2b7ec8ff7232eab4dfadd'
ADAPTER_SHA = '84917a93ee7b9840bb6c21ea6918f6102d13844ccbbd121c3bf3a840faf57c7c'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_gold(row):
    initial = row['messages'][0]['content']
    latest = row['messages'][-1]['content']
    n = int(re.search(r'(?:N is |value of N is )(\d+)', latest)[1])
    if 'divisible by' in initial:
        a, b = map(int, re.search(r'divisible by (\d+) or (\d+)', initial).groups())
        return sum(i % a == 0 or i % b == 0 for i in range(1, n+1))
    r = int(re.search(r'exactly (\d+) ones', initial)[1])
    # Enumerate placements, not the generator's binomial formula.
    return sum(not any(v+1 in positions for v in positions)
               for positions in itertools.combinations(range(n), r))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads((args.run / 'MANIFEST.json').read_text())
    assert {'INPUTS.json', 'PROVENANCE.json', 'OUTPUTS.jsonl', 'TIMING.json'} <= set(manifest)
    for name, expected in manifest.items():
        assert Path(name).name == name and digest(args.run / name) == expected
    provenance = json.loads((args.run / 'PROVENANCE.json').read_text())
    assert provenance['runner_sha256'] == RUNNER_SHA
    assert provenance['adapter_files']['adapter_model.safetensors'] == ADAPTER_SHA
    for name in ['tokenizer.json', 'tokenizer_config.json', 'generation_config.json']:
        assert digest(args.tokenizer / name) == provenance['base_files'][name]
    from tokenizers import Tokenizer
    decoder = Tokenizer.from_file(str(args.tokenizer / 'tokenizer.json'))
    tokenizer_config = json.loads((args.tokenizer / 'tokenizer_config.json').read_text())
    eos_token = tokenizer_config['eos_token']
    if isinstance(eos_token, dict):
        eos_token = eos_token['content']
    eos_id = decoder.token_to_id(eos_token)
    assert eos_id is not None
    rows = json.loads((args.run / 'INPUTS.json').read_text())
    assert len(rows) == len({r['id'] for r in rows}) == 64
    assert {(r['base'], r['update'], r['history']) for r in rows} == set(itertools.product(
        range(8), [False, True], ['clean', 'old_answer', 'current_answer', 'neutral']))
    lookup = {row['id']: row for row in rows}
    for row in rows:
        assert row['gold'] == text_gold(row)
        assert row['id'] == f"{row['base']}_{int(row['update'])}_{row['history']}"
    outputs = [json.loads(line) for line in (args.run / 'OUTPUTS.jsonl').read_text().splitlines()]
    assert len(outputs) == 128
    indexed = {(o['model'], o['id']): o for o in outputs}
    assert set(indexed) == set(itertools.product(['base', 'adapter'], lookup))
    cells = {}
    records = []
    for model, update, history in itertools.product(['base', 'adapter'], [False, True], ['clean', 'old_answer', 'current_answer', 'neutral']):
        counter = Counter(n=0, eos=0, valid=0, correct=0)
        for row in rows:
            if (row['update'], row['history']) != (update, history):
                continue
            o = indexed[model, row['id']]
            assert o['rendered'] == indexed['base', row['id']]['rendered']
            assert all(message['content'] in o['rendered'] for message in row['messages'])
            assert isinstance(o['output_ids'], list) and all(type(v) is int for v in o['output_ids'])
            ids = o['output_ids']
            assert 1 <= len(ids) <= 8192
            assert decoder.decode(ids, skip_special_tokens=True) == o['text']
            assert bool(ids[-1] == eos_id) == o['eos']
            assert eos_id not in ids[:-1]
            assert o['eos'] or len(ids) == 8192
            answer_line = o['text'].rstrip().splitlines()[-1] if o['text'].strip() else ''
            match = re.fullmatch(r'ANSWER:\s*([0-9]+)', answer_line)
            correct = match is not None and int(match[1]) == text_gold(row)
            counter.update(n=1, eos=int(o['eos']), valid=int(match is not None), correct=int(correct))
            records.append({'id': row['id'], 'model': model, 'valid': match is not None, 'correct': correct})
        cells[model, update, history] = dict(counter)
    interface = all(sum(cells[m, u, h][metric] for u, h in itertools.product([False, True],
        ['clean', 'old_answer', 'current_answer', 'neutral'])) >= 61 for m in ['base', 'adapter'] for metric in ['eos', 'valid'])
    clean = all(sum(cells[m, u, 'clean']['correct'] for u in [False, True]) >= 15 for m in ['base', 'adapter'])
    unchanged = all(cells[m, False, 'old_answer']['correct'] >= 7 for m in ['base', 'adapter'])
    anchor_control = all(cells[m, True, 'current_answer']['correct'] >= 7 for m in ['base', 'adapter'])
    difference = {h: cells['adapter', True, h]['correct'] - cells['base', True, h]['correct'] for h in
                  ['clean', 'old_answer', 'current_answer', 'neutral']}
    unchanged_difference = cells['adapter', False, 'old_answer']['correct'] - cells['base', False, 'old_answer']['correct']
    pattern = difference['old_answer'] <= -2 and abs(difference['neutral']) <= 1 and abs(unchanged_difference) <= 1 and anchor_control
    route = ('STOP_INADEQUATE_INTERFACE_OR_CAPABILITY' if not (interface and clean and unchanged) else
             'DESIGN_FRESH_CONFIRMATION_NOT_TRAINING' if pattern else 'STOP_NO_QUALIFYING_INTERACTION')
    result = {'classification': 'INDEPENDENT_TEXT_GOLD_AND_GATE_REPLAY', 'route': route,
              'manifest_sha256': digest(args.run / 'MANIFEST.json'),
              'gates': dict(interface=interface, clean_capability=clean, unchanged_capability=unchanged,
                            current_anchor_control=anchor_control, harmful_pattern=pattern),
              'updated_accuracy_count_difference_adapter_minus_base': difference,
              'unchanged_old_answer_count_difference': unchanged_difference,
              'cells': [{'model': m, 'update': u, 'history': h, **v} for (m,u,h),v in cells.items()],
              'records': records,
              'scope': 'Gold recomputed independently from saved user text. Manifest and source identity checked. '
                       'Token decoding and EOS independently checked using authenticated tokenizer files; no neural rerun. '
                       'Eight parameter bases in two families; no population inference.'}
    with args.out.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ['cells', 'records']}, indent=2))


if __name__ == '__main__':
    main()
