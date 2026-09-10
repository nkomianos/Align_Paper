"""Posthoc standard marginal calibration control; original artifacts unchanged."""
import json
import math
from pathlib import Path

from audit_itcr_released_data import digest, read_data, graph_from_row, lower_threshold, score_sequence, predict
from audit_itcr_calibration_contract import load_reviewed


def run(root):
    data, data_hash = read_data(root)
    plan_path = root / 'FROZEN_DATA_REPLAY_PLAN.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    assert plan['data_sha256'] == data_hash
    f, source_hash = load_reviewed(root)
    graphs = [graph_from_row(r, require_probability=False) for r in data]
    seqs = [f['generate_growth_subgraphs'](g) for g in graphs]
    scores = [score_sequence(seq, 'constant') for seq in seqs]
    calibration = []
    for r in plan['rows']:
        if r['split'] != 'calibration':
            continue
        i = r['index']
        bad = [t for t, g in enumerate(seqs[i]) if any(g.nodes[v]['annotation'] == 'N' for v in g)]
        calibration.append(float(scores[i][bad[0]]) if bad else math.inf)
    threshold = lower_threshold(calibration, 0.1)
    results = []
    for r in plan['rows']:
        if r['split'] != 'evaluation':
            continue
        i = r['index']; g = graphs[i]
        retained = predict(seqs[i], scores[i], threshold, strict=True)
        results.append(dict(index=i, covered=all(g.nodes[v]['annotation'] == 'Y' for v in retained),
                            retained=len(retained), total=len(g)))
    result = dict(scope='Posthoc control after earlier outcomes; no new model or confirmation claim',
                  rule='One first-bad score per graph, +infinity when no bad prefix; strict lower-tail acceptance',
                  alpha=0.1, calibration_count=len(calibration), no_bad_calibration=sum(math.isinf(s) for s in calibration),
                  threshold=threshold if math.isfinite(threshold) else str(threshold),
                  covered=sum(r['covered'] for r in results), evaluated=len(results),
                  mean_retained_fraction=sum(r['retained']/r['total'] for r in results)/len(results),
                  nonempty=sum(bool(r['retained']) for r in results), rows=results,
                  data_sha256=data_hash, source_sha256=source_hash, plan_sha256=digest(plan_path.read_bytes()),
                  runner_sha256=digest(Path(__file__).read_bytes()))
    with (root/'MARGINAL_CONSTANT_CONTROL.json').open('x', encoding='utf-8') as out:
        json.dump(result, out, indent=2, allow_nan=False)
    return {k:v for k,v in result.items() if k != 'rows'}


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    print(json.dumps(run(p.parse_args().root), indent=2))
