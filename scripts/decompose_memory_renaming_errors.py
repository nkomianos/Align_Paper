"""Posthoc component attribution of saved outputs, without changing original scores."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

from audit_memory_id_renaming import inputs
from run_benign_memory_extraction import parse, solve


def chronologies(n, edges):
    return {order for order in itertools.permutations(range(n))
            if all(order.index(a) < order.index(b) for a, b in edges)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    for name, digest in json.loads((args.run / 'MANIFEST.json').read_text()).items():
        assert Path(name).name == name
        assert hashlib.sha256((args.run / name).read_bytes()).hexdigest() == digest
    saved = json.loads((args.run / 'INPUTS.json').read_text())
    assert saved == inputs()
    gold = {row['id']: row for row in saved}
    outputs = [json.loads(line) for line in (args.run / 'OUTPUTS.jsonl').read_text().splitlines()]
    assert len(outputs) == 96
    assert len({(o['id'], o['arm']) for o in outputs}) == 96
    assert {(o['id'], o['arm']) for o in outputs} == {(r['id'], arm) for r in saved for arm in ['direct', 'extract']}
    records = []
    for output in outputs:
        if output['arm'] != 'extract':
            continue
        row = gold[output['id']]
        prediction = parse(output['text'])
        assert prediction == output['parsed']
        values, edges = prediction['values'], prediction['edges']
        answer = solve(values, edges, row['threshold'])
        expected_orders = chronologies(len(row['values']), row['edges'])
        value_ok = values == row['values']
        edge_ok = chronologies(len(values), edges) == expected_orders
        predicted_set = {tuple(edge) for edge in edges}
        reversed_set = {(b, a) for a, b in row['edges']}
        records.append({'id': row['id'], 'base': row['base'], 'mechanism': row['mechanism'],
                        'renamed': bool(row['presentation']), 'values_correct': value_ok,
                        'edge_order_correct': edge_ok, 'graph_correct': value_ok and edge_ok,
                        'category': 'correct' if value_ok and edge_ok else
                                    'values_only' if edge_ok else 'edges_only' if value_ok else 'both',
                        'complete_edge_reversal': predicted_set == reversed_set,
                        'answer_correct': answer == row['gold'],
                        'oracle_values_answer_correct': solve(row['values'], edges, row['threshold']) == row['gold'],
                        'oracle_edges_answer_correct': solve(values, row['edges'], row['threshold']) == row['gold'],
                        'both_oracles_answer_correct': solve(row['values'], row['edges'], row['threshold']) == row['gold']})
    assert all(row['both_oracles_answer_correct'] for row in records)
    summaries = {}
    for renamed in [False, True]:
        selected = [row for row in records if row['renamed'] == renamed]
        summaries['renamed' if renamed else 'identity'] = {
            'n': len(selected), 'categories': dict(Counter(row['category'] for row in selected)),
            **{key: sum(row[key] for row in selected) for key in
               ['values_correct', 'edge_order_correct', 'graph_correct', 'complete_edge_reversal',
                'answer_correct', 'oracle_values_answer_correct', 'oracle_edges_answer_correct']}}
    report = {'classification': 'POSTHOC_SAVED_OUTPUT_COMPONENT_ATTRIBUTION',
              'manifest_sha256': hashlib.sha256((args.run / 'MANIFEST.json').read_bytes()).hexdigest(),
              'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'summaries': summaries, 'records': records,
              'scope': 'Oracle substitutions diagnose scoring components, not deployable repairs or new model outputs. '
                       'Four exposed mechanisms; no independent sample inference or causal internal mechanism claim.'}
    with args.out.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
