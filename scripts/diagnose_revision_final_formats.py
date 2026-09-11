"""Retrospective final-format diagnosis; never replaces frozen pilot scores."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from verify_revision_checkpoint_pilot import text_gold


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    for name, sha in json.loads((args.run / 'MANIFEST.json').read_text()).items():
        assert Path(name).name == name
        assert hashlib.sha256((args.run / name).read_bytes()).hexdigest() == sha
    rows = {r['id']: r for r in json.loads((args.run / 'INPUTS.json').read_text())}
    outputs = [json.loads(line) for line in (args.run / 'OUTPUTS.jsonl').read_text().splitlines()]
    patterns = [
        ('plain_or_bold_answer', r'(?:^|\n)(?:\*\*)?ANSWER:\s*(?:\*\*)?([0-9]+)(?:\*\*)?\s*$'),
        ('terminal_boxed_integer', r'\\boxed\{\s*([0-9]+)\s*\}\s*(?:\$\$|\$|\\\])?\s*$'),
        ('bold_answer_label', r'(?:^|\n)\*\*ANSWER:\*\*\s*([0-9]+)\s*$'),
    ]
    records = []
    for o in outputs:
        final = o['text'].split('</think>')[-1]
        candidates = [(name, int(m[1])) for name, pattern in patterns if (m := re.search(pattern, final))]
        assert len(candidates) <= 1
        kind, answer = candidates[0] if candidates else ('unresolved', None)
        row = rows[o['id']]
        records.append({'model': o['model'], 'id': o['id'], 'base': row['base'], 'family': row['family'],
                        'update': row['update'], 'history': row['history'], 'format': kind,
                        'answer': answer, 'gold': text_gold(row), 'correct': answer == text_gold(row),
                        'generated_tokens': len(o['output_ids']), 'eos': o['eos'],
                        'alternate_eos_occurrences': o['output_ids'].count(151643),
                        'unresolved_tail': final[-200:] if answer is None else None})
    summaries = []
    for model in ['base', 'adapter']:
        selected = [r for r in records if r['model'] == model]
        summaries.append({'model': model, 'n': len(selected), 'formats': dict(Counter(r['format'] for r in selected)),
                          'correct': sum(r['correct'] for r in selected), 'eos': sum(r['eos'] for r in selected),
                          'generated_tokens': sum(r['generated_tokens'] for r in selected),
                          'max_generated_tokens': max(r['generated_tokens'] for r in selected),
                          'alternate_eos_occurrences': sum(r['alternate_eos_occurrences'] for r in selected)})
    report = {'classification': 'POSTHOC_FORMAT_DIAGNOSIS_NOT_FROZEN_GATE_PASS',
              'scope': 'Only explicit terminal ANSWER lines and terminal boxed integers are interpreted. '
                       'This reader was added after inspection; original strict scores and failed gates remain unchanged. '
                       'Final answers only, not verification of reasoning traces or general revision capability.',
              'summary': summaries, 'records': records}
    with args.out.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(summaries, indent=2))
    print(json.dumps([r for r in records if r['format'] == 'unresolved'], indent=2))


if __name__ == '__main__':
    main()
