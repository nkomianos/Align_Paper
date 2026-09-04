"""Supplementary decomposition, not a replacement for frozen DEV analysis.

Finite-sample marginal variances can differ despite equal sampling laws. A lower
observed difference variance alone does not establish beneficial coupling.
"""
import argparse
import json
from pathlib import Path
import math


def decompose(candidate, baseline):
    mc = candidate['variance_a'] + candidate['variance_b']
    mb = baseline['variance_a'] + baseline['variance_b']
    marginal = mb - mc
    covariance = 2 * (candidate['covariance'] - baseline['covariance'])
    observed = baseline['variance_difference'] - candidate['variance_difference']
    assert math.isclose(observed, marginal + covariance, abs_tol=1e-10)
    return {
        'observed_variance_reduction': observed,
        'sample_marginal_variance_contribution': marginal,
        'sample_covariance_contribution': covariance,
        'candidate_covariance': candidate['covariance'],
        'candidate_variance_over_own_marginal_sum':
            candidate['variance_difference'] / mc if mc > 0 else None,
        'scope': 'Descriptive decomposition, not an additional significance test',
    }


def audit(report):
    return {metric: {baseline: decompose(
        report['policies']['byte_hierarchical'][metric],
        report['policies'][baseline][metric])
        for baseline in ('independent', 'token_clock', 'byte_clock')}
        for metric in ('em', 'f1')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('report', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = audit(json.loads(args.report.read_text()))
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
