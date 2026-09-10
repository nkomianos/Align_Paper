"""Developmental BFCL-derived finite-choice grammar, not BFCL evaluation."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def construct(row, key):
    if len(row['function']) != 1 or len(key['ground_truth']) != 1:
        return None, 'not_single_function'
    function = row['function'][0]; truth = key['ground_truth'][0]
    if set(truth) != {function['name']}:
        return None, 'function_name_mismatch'
    schema = function['parameters']; accepted = truth[function['name']]
    required = schema.get('required', [])
    if not required or not set(required) <= set(accepted):
        return None, 'missing_required_key'
    values = {}
    for name, alternatives in accepted.items():
        if name not in required:
            if '' not in alternatives:
                return None, 'non_omittable_optional'
            continue
        if len(alternatives) != 1 or type(alternatives[0]) not in (str, int, float, bool):
            return None, 'ambiguous_or_structured_required_value'
        values[name] = alternatives[0]
    integer_keys = [name for name in sorted(required)
                    if type(values[name]) is int and 3 <= values[name] <= 1000
                    and schema['properties'][name].get('type') == 'integer'
                    and not any(k in schema['properties'][name] for k in ('enum', 'minimum', 'maximum'))]
    if not integer_keys:
        return None, 'no_unconstrained_integer'
    change = integer_keys[0]
    # Never make the correct argument systematically the smallest candidate.
    # Freeze its numeric rank independently of labels/model outcomes.
    numeric_rank = int(digest('bfcl-action-numeric-rank-v2:'+row['id'])[:8], 16) % 4
    deltas = list(range(-numeric_rank, 4-numeric_rank))
    actions = []
    for delta in deltas:
        arguments = copy.deepcopy(values); arguments[change] += delta
        actions.append({'tool': function['name'], 'arguments': arguments})
    order = sorted(range(4), key=lambda i: digest('action-order-v1:'+row['id']+':'+str(i)))
    aliases = []
    for i in order:
        action = actions[i]
        first = json.dumps(action, separators=(',', ':'), ensure_ascii=False)
        second = json.dumps({'arguments': action['arguments'], 'tool': action['tool']}, separators=(',', ':'), ensure_ascii=False)
        assert first != second and json.loads(first) == json.loads(second)
        aliases.append([first, second])
    questions = row['question']
    if len(questions) != 1 or len(questions[0]) != 1 or questions[0][0]['role'] != 'user':
        return None, 'not_single_user_request'
    result = {'id': row['id'], 'group': function['name'], 'target': order.index(numeric_rank),
        'prompt': questions[0][0]['content']+'\nTool schema:\n'+json.dumps(function, ensure_ascii=False)+
                  '\nReturn only a JSON object with tool and arguments keys.',
        'aliases': aliases, 'source_question': questions, 'source_function': function,
        'source_ground_truth': truth, 'mutated_argument': change, 'target_numeric_rank': numeric_rank,
        'scope': 'public BFCL prompt, constructed neighboring-integer distractors; no tool execution'}
    return result, None


def main():
    p = argparse.ArgumentParser(); p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    receipt = json.loads((a.source/'DOWNLOAD.json').read_text())
    for source in receipt['files']:
        assert hashlib.sha256((a.source/source['local_name']).read_bytes()).hexdigest() == source['sha256']
    data = [json.loads(line) for line in (a.source/'2_BFCL_v4_simple_python.json').read_text(encoding='utf-8').splitlines()]
    keys = [json.loads(line) for line in (a.source/'3_BFCL_v4_simple_python.json').read_text(encoding='utf-8').splitlines()]
    assert len({r['id'] for r in data}) == len(data)
    assert len({r['id'] for r in keys}) == len(keys)
    lookup = {r['id']: r for r in keys}; assert set(lookup) == {r['id'] for r in data}
    eligible = []; exclusions = Counter()
    for row in data:
        value, error = construct(row, lookup[row['id']])
        if error: exclusions[error] += 1
        else: eligible.append(value)
    groups = sorted({r['group'] for r in eligible}, key=lambda name: digest('bfcl-action-learning-v1:'+name))
    dev_groups = set(groups[::3]); selected = []
    for split, limit in [('train', 64), ('dev', 32)]:
        candidates = [r for r in eligible if (r['group'] in dev_groups) == (split == 'dev')]
        candidates.sort(key=lambda r: digest('bfcl-action-row-v1:'+r['id']))
        selected.extend([{**r, 'split': split} for r in candidates[:limit]])
    assert not ({r['group'] for r in selected if r['split']=='train'} & {r['group'] for r in selected if r['split']=='dev'})
    (a.out/'INPUTS.json').write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding='utf-8')
    report = {'classification': 'DEVELOPMENTAL_DATA_CONSTRUCTION', 'revision': receipt['revision'],
        'source_rows': len(data), 'eligible': len(eligible), 'eligible_groups': len(groups),
        'excluded': dict(exclusions), 'selected': dict(Counter(r['split'] for r in selected)),
        'selected_groups': {split: len({r['group'] for r in selected if r['split']==split}) for split in ('train','dev')},
        'targets': dict(Counter(r['target'] for r in selected)), 'historical_locked_split_accessed': False,
        'target_numeric_ranks': dict(Counter(r['target_numeric_rank'] for r in selected)),
        'qualification': 'not yet independently reviewed or tested on a model',
        'source_receipt_sha256': hashlib.sha256((a.source/'DOWNLOAD.json').read_bytes()).hexdigest()}
    (a.out/'AUDIT.json').write_text(json.dumps(report, indent=2))
    (a.out/'MANIFEST.json').write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in a.out.iterdir() if p.is_file()}, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
