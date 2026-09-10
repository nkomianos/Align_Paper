"""Posthoc Bayesian ranking audit of saved CLARA rows, not a learned baseline."""
import collections
import hashlib
import json
import math
from pathlib import Path


def summarize(rows):
    accepted = [r for r in rows if r['joint_decision'] is not None]
    n = len(accepted)
    if not n:
        return dict(rows=len(rows), accepted=0, coverage=0, risk=None)
    actual = math.fsum(1-r['posterior_true'] if r['joint_decision']
                       else r['posterior_true'] for r in accepted)/n
    errors = sorted(min(r['posterior_true'], 1-r['posterior_true']) for r in rows)
    optimum = math.fsum(errors[:n])/n
    # Selecting the n smallest posterior errors minimizes error at count n.
    # Equal-error ties can be randomly accepted at a common boundary rate;
    # ordering their equal contributions has no effect on this expected risk.
    assert optimum <= actual + 1e-12
    return dict(rows=len(rows), accepted=n, coverage=n/len(rows),
                joint_risk=actual, posterior_oracle_risk=optimum,
                excess_risk=actual-optimum)


def main():
    source = Path('artifacts/research_pilots_20260906/clara_cpu_final/ROWS.json')
    manifest = json.loads((source.parent/'MANIFEST.json').read_text())
    for name, expected in manifest.items():
        assert hashlib.sha256((source.parent/name).read_bytes()).hexdigest() == expected
    data = source.read_bytes()
    rows = json.loads(data)
    groups = collections.defaultdict(list)
    for r in rows:
        assert 0 <= r['posterior_true'] <= 1
        groups[r['error'], r['correlation']].append(r)
    cells = [dict(error=e, correlation=c, **summarize(rr))
             for (e, c), rr in sorted(groups.items())]
    result = dict(classification='posthoc developmental Bayesian apparatus',
                  source=str(source), source_sha256=hashlib.sha256(data).hexdigest(),
                  pooled=summarize(rows), cells=cells,
                  max_absolute_cell_excess=max(abs(r['excess_risk']) for r in cells),
                  independent_sampling_inference=False, neural_experiment=False)
    out = Path('artifacts/clara_matched_coverage_20260910')
    out.mkdir(exist_ok=True)
    (out/'RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
