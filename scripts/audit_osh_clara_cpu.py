"""Read-only OSH recomputation and CLARA counterexamples; no model imports or GPU.

Developmental audit receipts, not new neural experiments. Only selected AST
constants/functions are evaluated; repository modules are never imported.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--osh', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    root = args.osh
    names = ['paper.tex', 'SESSION_RESULTS.md', 'logs_heldout_v11.txt',
             'logs_harmbench_real.txt', 'exp5_mmlu_results.json',
             'exp6_ood_results.json', 'results_ood_v2.json',
             'osh_heldout_safety_benchmark.py', 'osh_proprioception_v11.py',
             'osh_eval_config.py', 'exp6_ood_generalization_v2.py',
             'osh_harmbench_eval.py', 'osh_lethal_antidote.py']
    hashes = {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in names}
    log = (root / 'logs_heldout_v11.txt').read_text(encoding='utf-8')
    rows = re.findall(r'Expected: (Yes|No)\s+\|\s+Baseline: (Yes|No) \[\w+\]\s+\|\s+OSH: (Yes|No)', log)
    assert len(rows) == 50
    heldout = {'n': len(rows), 'labels': dict(Counter(r[0] for r in rows)),
               'baseline_choices': dict(Counter(r[1] for r in rows)),
               'osh_choices': dict(Counter(r[2] for r in rows)),
               'baseline_correct': sum(y == b for y, b, o in rows),
               'osh_correct': sum(y == o for y, b, o in rows),
               'always_no_correct': sum(y == 'No' for y, b, o in rows),
               'improved': sum(y != b and y == o for y, b, o in rows),
               'regressed': sum(y == b and y != o for y, b, o in rows)}
    assert (heldout['baseline_correct'], heldout['osh_correct'], heldout['always_no_correct']) == (15, 27, 35)
    mmlu = json.loads((root / 'exp5_mmlu_results.json').read_text())
    capability = {}
    for name, result in mmlu.items():
        n = sum(x['n_total'] for x in result['subjects'].values())
        correct = sum(x['n_correct'] for x in result['subjects'].values())
        assert abs(correct / n - result['overall']) < 1e-12
        capability[name] = {'n': n, 'correct': correct, 'accuracy': correct / n,
                            'subjects': len(result['subjects'])}
    ood = json.loads((root / 'exp6_ood_results.json').read_text())
    refusal = {}
    for name, result in ood.items():
        rows = result['responses']
        score = sum(r['score'] for r in rows) / len(rows)
        assert abs(score - result['overall_score']) < 1e-12
        ambiguous = [r for r in rows if r.get('notes', '').startswith('Ambiguous:')]
        refusal[name] = {'n': len(rows), 'recorded_score': score,
                         'classification_counts': dict(Counter(r['classification'] for r in rows)),
                         'ambiguous_n': len(ambiguous),
                         'sensitivity_ambiguous_zero': (sum(r['score'] for r in rows) - sum(r['score'] for r in ambiguous)) / len(rows)}
    # Execute only the literal heldout constant and the pure curriculum builder.
    held_tree = ast.parse((root / 'osh_heldout_safety_benchmark.py').read_text(encoding='utf-8'))
    questions = next(ast.literal_eval(n.value) for n in held_tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'HELDOUT_QUESTIONS' for t in n.targets))
    train_tree = ast.parse((root / 'osh_proprioception_v11.py').read_text(encoding='utf-8'))
    builder = next(n for n in train_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'generate_expanded_curriculum')
    ns = {}
    exec(compile(ast.Module(body=[builder], type_ignores=[]), '<curriculum-only>', 'exec'), ns)
    train = ns['generate_expanded_curriculum']()
    train_questions = {r['text'].split('\n\nAssistant:')[0].removeprefix('User: ') for r in train}
    overlap = [q['question'] for q in questions if q['question'] in train_questions]
    # Exact overlap is weaker than independent semantic/task splits.
    # Counterexample: actual attention selects V at the maximal A index.
    beta = 100.0
    attention = 10 * math.exp(-beta) / (1 + math.exp(-beta))
    tropical = max(1 + 0, 0 + 10)
    assert attention < 1e-40 and tropical == 10
    # Exact GeLU second derivative at 0 is 2 / sqrt(2*pi), not <= 0.4.
    gelu_second_zero = math.sqrt(2 / math.pi)
    assert gelu_second_zero > .4
    gelu = lambda x: .5 * x * (1 + math.erf(x / math.sqrt(2)))
    x0, x1 = 0.0, 10 / 16
    mid = (x0 + x1) / 2
    interpolation_error = abs((gelu(x0) + gelu(x1)) / 2 - gelu(mid))
    assert interpolation_error > 5 / 16**2
    # More derivations can reverse best-proof vs marginal ranking.
    # The A derivations are mutually exclusive; no independence approximation.
    probability = {'A_disjoint_derivations': [.3, .3], 'B_derivations': [.5],
                   'max_derivation_winner': 'B', 'marginal_winner': 'A'}
    receipt = {'classification': 'developmental audit; no neural execution',
               'source_sha256': hashes, 'heldout': heldout, 'mmlu_subset': capability,
               'early_ood': refusal, 'v11_curriculum_rows': len(train),
               'v11_unique_questions': len(train_questions), 'exact_heldout_overlap': overlap,
               'clara_counterexamples': {'attention_beta100': attention,
                   'claimed_tropical_result': tropical, 'gelu_second_derivative_zero': gelu_second_zero,
                   'gelu_uniform16_midpoint_error': interpolation_error,
                   'claimed_uniform16_bound': 5 / 16**2, 'map_vs_marginal': probability},
               'limits': ['No logits or checkpoint replay; logs can establish recorded outputs only.',
                          'Ambiguous-zero is a post hoc sensitivity, not a replacement validated safety judge.',
                          'No population confidence interval inferred from repeated deterministic generations.',
                          'No claim of neural mechanism or security from these CPU checks.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'source_sha256'}, indent=2))


if __name__ == '__main__':
    main()
