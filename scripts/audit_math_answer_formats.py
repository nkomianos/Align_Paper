"""Post-hoc format diagnosis only; never changes the frozen bank's route."""
import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import re
from run_unexplored_screens import dump, sha


def numeric(raw):
    text = raw.strip().strip('$').strip()
    # Accept whole numeric expressions, never a numeric prefix of prose/math.
    if re.fullmatch(r'[-+]?\d+(?:\.\d+)?', text):
        return Fraction(text)
    match = re.fullmatch(r'([-+]?\d+)\s*/\s*([-+]?\d+)', text)
    if not match:
        match = re.fullmatch(r'\\(?:dfrac|tfrac|frac)\{([-+]?\d+)\}\{([-+]?\d+)\}', text)
    if match and int(match[2]) != 0:
        return Fraction(int(match[1]), int(match[2]))
    return None


def boxes(text):
    result = []
    for match in re.finditer(r'\\boxed\s*\{', text):
        depth = 1; start = match.end(); end = start
        while end < len(text) and depth:
            if text[end] == '{': depth += 1
            elif text[end] == '}': depth -= 1
            end += 1
        if depth == 0:
            result.append((match.start(), text[start:end-1]))
    return result


def extract(text):
    candidates = [(p, 'boxed', value) for p, value in boxes(text)]
    for match in re.finditer(r'(?m)^####[ \t]+([^\n]+)', text):
        if numeric(match[1]) is not None:
            candidates.append((match.start(), 'delimiter', match[1]))
    if not candidates:
        return {'kind': 'unextracted', 'raw': None, 'value': None}
    _, kind, raw = max(candidates, key=lambda x: x[0])
    value = numeric(raw)
    return {'kind': kind, 'raw': raw, 'value': None if value is None else str(value)}


def main():
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path)
    p.add_argument('--out', type=Path, required=True); a = p.parse_args()
    prefixes = {(r['base'], r['index'] if 'index' in r else r['prefix_index']): r['text']
                for r in json.loads((a.root/'PREFIXES.json').read_text())}
    rows = [json.loads(x) for x in (a.root/'ROLLOUTS.jsonl').read_text().splitlines()]
    counts = Counter(); diagnostics = []
    for row in rows:
        full = prefixes[row['base'], row['prefix_index']]+row['completion']
        result = extract(full)
        counts[result['kind']] += 1
        if result['value'] is not None: counts['numeric_extract'] += 1
        if row['parsed_answer'] is None and result['kind'] != 'unextracted': counts['previously_missing_extracted'] += 1
        if row['parsed_answer'] is not None and result['value'] is not None:
            if Fraction(row['parsed_answer']) != Fraction(result['value']): counts['numeric_disagreement'] += 1
        if not row['eos']: counts['no_eos'] += 1
        diagnostics.append({'base': row['base'], 'prefix': row['prefix_index'], 'sample': row['sample'],
                            'old_value': row['parsed_answer'], **result, 'eos': row['eos']})
    report = {'classification': 'POST_HOC_FORMAT_DIAGNOSIS_ONLY', 'frozen_gate_changed': False,
              'source_manifest_sha256': sha(a.root/'MANIFEST.json'), 'n': len(rows),
              'counts': dict(counts), 'rows': diagnostics,
              'limitations': 'last explicit answer heuristic; symbolic equivalence unsupported; no correctness rerouting'}
    dump(a.out, report); print(json.dumps({k: v for k, v in report.items() if k != 'rows'}))


if __name__ == '__main__': main()
