"""Constructed output-only control; no neural model or author-result replication."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    # Deterministic, constructed scores. These are not sampled questions.
    t = np.arange(500, dtype=np.float64)
    real = np.stack([np.sin(t * f) + np.cos(t * (f + .07))
                     for f in (.11, .17, .23, .31)], axis=1)
    abstain = (np.sin(t * .13) + .5)[:, None]
    logits = np.concatenate([real, abstain], axis=1)
    q = softmax(real)
    baseline = softmax(logits)
    strengths = [-2., -1., 0., 1., 2.]
    cells = []
    raw = []
    max_conditional_error = 0.
    max_derivative_error = 0.
    for strength in strengths:
        edited = logits.copy()
        edited[:, 4] -= strength
        p = softmax(edited)
        conditional = p[:, :4] / p[:, :4].sum(axis=1, keepdims=True)
        error = float(np.max(np.abs(conditional - q)))
        max_conditional_error = max(max_conditional_error, error)
        assert error < 1e-14
        assert np.array_equal(np.argmax(p[:, :4], axis=1), np.argmax(real, axis=1))
        margin = p[:, :4].max(axis=1) - p[:, 4]
        expected = q.max(axis=1) - (q.max(axis=1) + 1) * p[:, 4]
        assert np.max(np.abs(margin - expected)) < 1e-14
        # Independent finite-difference check of d margin / d strength.
        plus, minus = edited.copy(), edited.copy()
        plus[:, 4] -= 1e-5
        minus[:, 4] += 1e-5
        pp, pm = softmax(plus), softmax(minus)
        derivative = ((pp[:, :4].max(axis=1) - pp[:, 4]) -
                      (pm[:, :4].max(axis=1) - pm[:, 4])) / 2e-5
        expected_derivative = (q.max(axis=1) + 1) * p[:, 4] * (1 - p[:, 4])
        max_derivative_error = max(max_derivative_error, float(np.max(
            np.abs(derivative - expected_derivative))))
        assert np.max(np.abs(derivative - expected_derivative)) < 1e-9
        cells.append({'strength': strength,
                      'greedy_abstention_count': int(np.sum(np.argmax(p, axis=1) == 4)),
                      'mean_abstention_probability': float(p[:, 4].mean()),
                      'mean_max_real_probability': float(p[:, :4].max(axis=1).mean()),
                      'mean_margin': float(margin.mean()),
                      'max_conditional_distribution_change': error})
        for i in range(len(t)):
            raw.append({'constructed_id': i, 'strength': strength,
                        'logits': edited[i].tolist(), 'probabilities': p[i].tolist()})
    # Negative control: shifting every output by the same amount changes nothing.
    common_shift_error = float(np.max(np.abs(softmax(logits + 17.) - baseline)))
    assert common_shift_error < 1e-14
    # Relabeling the abstention output and remapping the result is equivariant.
    permutation_error = 0.
    for position in range(5):
        order = list(range(5))
        order[4], order[position] = order[position], order[4]
        permutation_error = max(permutation_error, float(np.max(np.abs(
            softmax(logits[:, order])[:, np.argsort(order)] - baseline))))
    assert permutation_error < 1e-14
    assert all(a['mean_margin'] < b['mean_margin'] for a, b in zip(cells, cells[1:]))
    assert all(a['greedy_abstention_count'] >= b['greedy_abstention_count']
               for a, b in zip(cells, cells[1:]))
    raw_path = args.out / 'CONSTRUCTED_ROWS.jsonl'
    raw_path.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in raw),
                        encoding='utf-8')
    result = {'classification': 'developmental_exact_control_not_neural_evidence',
              'constructed_items': 500, 'dependent_views': len(raw), 'cells': cells,
              'max_conditional_error': max_conditional_error,
              'max_derivative_error': max_derivative_error,
              'common_shift_error': common_shift_error,
              'permutation_error': permutation_error,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'raw_sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest(),
              'numpy_version': np.__version__,
              'scope': 'Output bias suffices for these signatures; actual neural mechanism untested.'}
    (args.out / 'RESULT.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n',
                                         encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
