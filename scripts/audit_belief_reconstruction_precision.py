"""Outcome-blind cohort precision sensitivity; never reads survey endpoints.

Gaussian random effects are hypothetical, not fitted to participants. No model
inference, power guarantee, or human outcome analysis is performed.
"""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm, t
from interaction_sprint.belief_reconstruction_views import parse_transcript, prefix

SHA = '6f5ac1b28de08c302abad8f25b2451df36d74006ab9b15e6dac6691722c0841d'
SALT = 'belief-reconstruction-query-split-20260904-v1'


def precision(sizes, rho):
    sizes = np.asarray(sizes, dtype=float)
    if sizes.ndim != 1 or len(sizes) < 2 or not np.isfinite(sizes).all() or np.any(sizes <= 0):
        raise ValueError('At least two finite positive cluster sizes required')
    if not 0 <= rho <= 1:
        raise ValueError('Correlation must be between zero and one')
    n = sizes.sum()
    variance = rho * np.sum((sizes/n)**2) + (1-rho)/n
    return dict(n=int(n), clusters=len(sizes), rho=rho,
                standardized_se=float(np.sqrt(variance)),
                variance_equivalent_independent_n=float(1/variance),
                approximate_80pct_two_sided_mde=float(
                    (t.ppf(.975, len(sizes)-1)+norm.ppf(.8))*np.sqrt(variance)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    raw = a.data.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA:
        raise ValueError('Checksum mismatch')
    counts = Counter()
    for row in csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))):
        if row['personalization'] != 'non-personalized' or row['passed_attention_check'] != 'True':
            continue
        try:
            turns = parse_transcript(row['conversation_parsed'], row['total_messages'])
            prefix(turns, 6)
        except ValueError:
            continue
        counts[row['queryId']] += 1
    # Deterministic order fixed without using ratings, predictions or text content.
    ranked = sorted(counts, key=lambda q: hashlib.sha256((SALT+'|'+q).encode()).hexdigest())
    dev_count = (len(ranked)+3)//4
    groups = dict(dev=ranked[:dev_count], confirmation=ranked[dev_count:])
    report = dict(status='DESIGN_SENSITIVITY_NOT_EMPIRICAL_POWER', sha256=SHA,
                  split_salt=SALT, allocation='First ceil(G/4) hash-ranked query IDs to DEV; rest confirmation',
                  unit='Standard deviation of participant paired squared-error differences; NOT rating points',
                  assumptions='Independent query random effects, common variance and intraclass correlation. '
                              'Repeated participants within a query correlated. Cross-query dependence is not modeled.',
                  scope='Query IDs are not proven semantically disjoint topic families. No endpoint values read.',
                  splits={})
    for name, queries in groups.items():
        sizes = [counts[q] for q in queries]
        report['splits'][name] = dict(
            # Query hashes allow reconstruction without releasing participant identifiers.
            queries=[dict(query_sha256=hashlib.sha256(q.encode()).hexdigest(), n=counts[q]) for q in queries],
            sensitivity=[precision(sizes, rho) for rho in (0, .1, .3, .5, 1)])
    with a.out.open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps({name: value['sensitivity'] for name, value in report['splits'].items()}, indent=2))


if __name__ == '__main__':
    main()
