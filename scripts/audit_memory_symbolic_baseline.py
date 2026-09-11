"""Posthoc grammar baseline on saved benign memory inputs; no model or gold access in solver."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re


def parse_text(text):
    values, edges, threshold = {}, set(), None
    for line in text.splitlines():
        if match := re.fullmatch(r'Event (\d+) set the value to (-?\d+)\.', line):
            event, value = map(int, match.groups())
            if event in values:
                raise ValueError('duplicate event declaration')
            values[event] = value
        elif match := re.fullmatch(r'Event (\d+) occurred after event (\d+)\.', line):
            later, earlier = map(int, match.groups())
            edges.add((earlier, later))
        elif match := re.fullmatch(r'Threshold: (-?\d+)\.', line):
            if threshold is not None:
                raise ValueError('duplicate threshold')
            threshold = int(match[1])
        else:
            raise ValueError('unsupported input grammar')
    if not values or threshold is None:
        raise ValueError('missing declarations')
    if any(a not in values or b not in values for a, b in edges):
        raise ValueError('undeclared relation endpoint')
    return values, edges, threshold


def decide(values, edges, threshold):
    outgoing = {event: set() for event in values}
    incoming = dict.fromkeys(values, 0)
    for earlier, later in edges:
        if earlier not in values or later not in values:
            raise ValueError('undeclared endpoint')
        if later not in outgoing[earlier]:
            outgoing[earlier].add(later)
            incoming[later] += 1
    queue = [event for event, count in incoming.items() if count == 0]
    visited = 0
    while queue:
        event = queue.pop()
        visited += 1
        for later in outgoing[event]:
            incoming[later] -= 1
            if incoming[later] == 0:
                queue.append(later)
    if visited != len(values):
        raise ValueError('cyclic chronology')
    final_conditions = {values[event] >= threshold for event, successors in outgoing.items()
                        if not successors}
    if not final_conditions:
        raise ValueError('empty graph')
    return 'CLARIFY' if len(final_conditions) == 2 else 'YES' if True in final_conditions else 'NO'


def check_solver():
    """Independent exhaustive linear-extension comparison, not generated neural data."""
    graphs, value_settings, cycles = 0, 0, 0
    for n in range(1, 5):
        potential = list(itertools.permutations(range(n), 2))
        orders = list(itertools.permutations(range(n)))
        for mask in range(1 << len(potential)):
            edges = {edge for index, edge in enumerate(potential) if mask & (1 << index)}
            valid = [order for order in orders if all(order.index(a) < order.index(b) for a, b in edges)]
            if not valid:
                try:
                    decide(dict.fromkeys(range(n), 0), edges, 0)
                except ValueError:
                    cycles += 1
                else:
                    raise AssertionError('cycle accepted')
                continue
            graphs += 1
            for assignment in itertools.product([-1, 1], repeat=n):
                expected = {assignment[order[-1]] >= 0 for order in valid}
                expected_answer = 'CLARIFY' if len(expected) == 2 else 'YES' if True in expected else 'NO'
                assert decide(dict(enumerate(assignment)), edges, 0) == expected_answer
                value_settings += 1
    good = 'Event 7 set the value to -1.\nEvent 2 set the value to 1.\nEvent 2 occurred after event 7.\nThreshold: 0.'
    assert decide(*parse_text(good)) == 'YES'
    rejected = 0
    for bad in [good + '\nunknown fact', good + '\nThreshold: 1.',
                good + '\nEvent 2 set the value to 5.', good.replace('after event 7', 'after event 9')]:
        try:
            parse_text(bad)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError('invalid input accepted')
    return dict(acyclic_graphs=graphs, value_settings=value_settings, cycles_rejected=cycles,
                invalid_grammar_controls_rejected=rejected)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    checks = check_solver()
    records, receipts = [], []
    all_texts = set()
    for root in args.runs:
        manifest = root / 'MANIFEST.json'
        for name, expected in json.loads(manifest.read_text()).items():
            assert Path(name).name == name and sha(root / name) == expected
        rows = json.loads((root / 'INPUTS.json').read_text())
        assert len({row['id'] for row in rows}) == len(rows)
        receipts.append(dict(run=root.name, manifest_sha256=sha(manifest), inputs=len(rows)))
        for row in rows:
            # Gold data is used only after prediction, for comparison.
            values, edges, threshold = parse_text(row['text'])
            answer = decide(values, edges, threshold)
            correct_values = values == dict(enumerate(row['values']))
            correct_edges = edges == {tuple(edge) for edge in row['edges']}
            correct_threshold = threshold == row['threshold']
            records.append(dict(run=root.name, id=row['id'], base=row['base'],
                                values_correct=correct_values, edges_correct=correct_edges,
                                threshold_correct=correct_threshold, answer=answer,
                                correct=answer == row['gold']))
            all_texts.add(row['text'])
    result = dict(classification='POSTHOC_SYNTHETIC_GRAMMAR_BASELINE',
                  source_sha256=sha(Path(__file__)), receipts=receipts, independent_checks=checks,
                  rows=len(records), unique_input_strings=len(all_texts),
                  correct=sum(r['correct'] for r in records),
                  exact_fact_extractions=sum(r['values_correct'] and r['edges_correct'] and r['threshold_correct'] for r in records),
                  records=records,
                  scope='Grammar chosen after inspecting source templates. No learned model, natural-language '
                        'generalization, fresh confirmation, or repair of original scores. Views are dependent.')
    with args.out.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
