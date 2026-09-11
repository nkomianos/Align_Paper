"""Exact finite control for shared versus report-only confidence noise."""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import argparse
import hashlib
import json


def mean(rows, fn):
    return sum((r['weight'] * fn(r) for r in rows), F(0))


def auc(rows, score, label):
    pos = [r for r in rows if label(r)]
    neg = [r for r in rows if not label(r)]
    numerator = F(0)
    for a, b in product(pos, neg):
        credit = F(1) if score(a) > score(b) else F(1, 2) if score(a) == score(b) else F(0)
        numerator += a['weight'] * b['weight'] * credit
    return numerator / (sum(r['weight'] for r in pos) * sum(r['weight'] for r in neg))


def construct(shared):
    rows = []
    for p, e, redraw, y in product(map(F, ['.3', '.4', '.6', '.7']),
                                  [F(-3, 20), F(3, 20)],
                                  [F(-3, 20), F(3, 20)], [0, 1]):
        report = p + e
        decision_score = p + (e if shared else redraw)
        rows.append(dict(p=p, e=e, redraw=redraw, y=y, report=report,
                         commit=int(decision_score > F(1, 2)),
                         weight=F(1, 16) * (p if y else 1-p)))
    assert sum(r['weight'] for r in rows) == 1
    return rows


def analyze(rows):
    mp = mean(rows, lambda r: r['p'])
    mv = mean(rows, lambda r: r['report'])
    covariance = mean(rows, lambda r: (r['p']-mp)*(r['report']-mv))
    variance = mean(rows, lambda r: (r['p']-mp)**2)
    slope = covariance / variance
    intercept = mv - slope * mp
    assert slope == 1 and intercept == 0
    for r in rows:
        r['residual'] = r['report'] - intercept - slope*r['p']
        assert r['residual'] == r['e']
        assert 0 <= r['report'] <= 1
    return {'ols_intercept': intercept, 'ols_slope': slope,
            'residual_correctness_auc': auc(rows, lambda r: r['residual'], lambda r: r['y']),
            'residual_commit_auc': auc(rows, lambda r: r['residual'], lambda r: r['commit']),
            'report_correctness_auc': auc(rows, lambda r: r['report'], lambda r: r['y']),
            'report_commit_auc': auc(rows, lambda r: r['report'], lambda r: r['commit']),
            'oracle_correctness_auc': auc(rows, lambda r: r['p'], lambda r: r['y']),
            'commit_rate': mean(rows, lambda r: r['commit'])}


def serialize(x):
    if isinstance(x, F):
        return {'fraction': str(x), 'decimal': float(x)}
    raise TypeError(type(x).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    shared, independent = construct(True), construct(False)
    results = {'shared_estimation_noise': analyze(shared),
               'independent_report_noise': analyze(independent)}
    assert results['shared_estimation_noise']['residual_correctness_auc'] == F(1, 2)
    assert results['shared_estimation_noise']['residual_commit_auc'] == F(3, 4)
    assert results['independent_report_noise']['residual_correctness_auc'] == F(1, 2)
    assert results['independent_report_noise']['residual_commit_auc'] == F(1, 2)
    # Exact calibration of the truth proxy and independence of error from truth.
    for rows in (shared, independent):
        for p in {r['p'] for r in rows}:
            cell = [r for r in rows if r['p'] == p]
            assert mean(cell, lambda r: r['y']) / sum(r['weight'] for r in cell) == p
        for e in {r['e'] for r in rows}:
            cell = [r for r in rows if r['e'] == e]
            assert mean(cell, lambda r: r['y']) / sum(r['weight'] for r in cell) == F(1, 2)
    raw = args.out / 'EXACT_ROWS.json'
    raw.write_text(json.dumps({'shared': shared, 'independent': independent},
                             default=serialize, indent=2) + '\n', encoding='utf-8')
    result = {'classification': 'exact_developmental_control_not_neural_replication',
              'results': results,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'raw_sha256': hashlib.sha256(raw.read_bytes()).hexdigest(),
              'scope': 'Shared estimation error differs from independent report-only noise.'}
    text = json.dumps(result, default=serialize, indent=2) + '\n'
    (args.out / 'RESULT.json').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
