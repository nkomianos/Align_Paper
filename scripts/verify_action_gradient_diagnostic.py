"""Replay covariance summaries from saved per-spelling parameter gradients."""
import argparse
import json
from pathlib import Path
import torch
from run_unexplored_screens import sha, dump


def verify(root):
    for name, expected in json.loads((root/'MANIFEST.json').read_text()).items():
        assert sha(root/name) == expected
    report = json.loads((root/'SUMMARY.json').read_text())
    rows = report['results']
    assert len(rows) == 8
    assert {(r['id'], r['canonical']) for r in rows} == {(f'move_{i}', c) for i in range(4) for c in (False, True)}
    for row in rows:
        name = row['id'] + ('_canonical' if row['canonical'] else '_aliases')
        raw = torch.load(root/(name+'.pt'), map_location='cpu', weights_only=True)
        for method, moments in raw['moments'].items():
            gradient = moments['sample_gradients']; probability = moments['probability']
            assert torch.isfinite(gradient).all() and torch.isfinite(probability).all()
            assert abs(float(probability.sum())-1) < 1e-5
            mean = (probability[:, None]*gradient).sum(0)
            variance = (probability*(gradient-mean).square().sum(1)).sum()
            assert torch.allclose(mean, moments['mean'], atol=1e-12, rtol=1e-10)
            assert torch.allclose(variance, moments['variance_trace'], atol=1e-12, rtol=1e-10)
            error = float((mean-raw['reference']).norm()/raw['reference'].norm().clamp_min(1e-12))
            assert abs(error-row['relative_mean_error'][method]) < 1e-10
            assert abs(float(variance)-row['variance'][method]) < 1e-10
        reduction = 1-row['variance']['marginal']/max(row['variance']['spelling'], 1e-30)
        assert abs(reduction-row['reduction']) < 1e-10
        scores = torch.tensor(row['scores'], dtype=torch.float64)
        ids = torch.tensor([a for a, _ in row['choices']])
        logits = scores+torch.tensor(row['offsets'], dtype=torch.float64)[ids]
        probabilities = logits.softmax(0)
        for moments in raw['moments'].values():
            assert torch.allclose(probabilities, moments['probability'], atol=1e-6, rtol=1e-5)
        masses = [float(probabilities[ids == i].sum()) for i in range(4)]
        assert max(abs(x-y) for x, y in zip(masses, row['action_mass'])) < 1e-5
        if row['canonical']:
            assert abs(row['reduction']) < .02
    valid = all(max(r['relative_mean_error'].values()) <= .02 and
                max(abs(x-.25) for x in r['action_mass']) <= 1e-5 for r in rows)
    median = float(torch.tensor([r['reduction'] for r in rows if not r['canonical']]).quantile(.5))
    route = 'INVALID_GRADIENT_APPARATUS' if not valid else (
        'LEARNING_PILOT_MAY_BE_DESIGNED' if median >= .2 else 'STOP_SMALL_VARIANCE_EFFECT')
    assert report['valid'] == valid and report['route'] == route
    assert abs(report['median_variance_reduction']-median) < 1e-10
    return {'verified': True, 'route': route, 'scope': 'gradient apparatus only',
            'median_variance_reduction': median, 'manifest_sha256': sha(root/'MANIFEST.json')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root', type=Path)
    parser.add_argument('--out', type=Path, required=True); args = parser.parse_args()
    result = verify(args.root); dump(args.out, result); print(json.dumps(result))
