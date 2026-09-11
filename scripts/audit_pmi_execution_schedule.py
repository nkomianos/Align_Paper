"""Compare saved full/cached logits; no neural calls or replacement of frozen scores."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(root):
    for name, digest in json.loads((root / 'MANIFEST.json').read_text()).items():
        assert Path(name).name == name and sha(root / name) == digest
    rows = [json.loads(line) for line in (root / 'ROWS.jsonl').read_text().splitlines()]
    tensors = torch.load(root / 'LOGITS.pt', map_location='cpu', weights_only=True)
    assert len(rows) == len(tensors) == 46
    assert len({(r['base'], r['kind']) for r in rows}) == 46
    return rows, tensors


def probability(logits):
    return np.exp(logits - logsumexp(logits))


def distributions(tensors):
    x = {key: value.numpy().astype(np.float64) for key, value in tensors.items()}
    assert all(np.isfinite(value).all() for value in x.values())

    def target(teacher, reference):
        # Centering cancels log-softmax constants algebraically. Independent
        # float64 computation of the frozen beta=1, clip=10 transformation.
        delta = x[teacher] - x[reference]
        delta -= delta.mean()
        return probability(x['base'] + 10 * np.tanh(delta / 10))

    return {'base': probability(x['base']), 'cached_base': probability(x['cached_base']),
            'purified': target('teacher', 'reference'),
            'control': target('base', 'unconditional'),
            'wrong': target('wrong_teacher', 'wrong_reference')}


def tv(a, b):
    return float(np.abs(a - b).sum() / 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', type=Path, required=True)
    parser.add_argument('--cached', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    fr, ft = read(args.full)
    cr, ct = read(args.cached)
    records = []
    for left, right, fl, cl in zip(fr, cr, ft, ct):
        for key in ['base', 'kind', 'length', 'inputs', 'wrong_reference_base']:
            assert left[key] == right[key]
        assert torch.equal(fl['cached_base'], cl['cached_base'])
        a, b = distributions(fl), distributions(cl)
        assert np.array_equal(b['base'], b['cached_base'])
        for row, dist in [(left, a), (right, b)]:
            for saved_key, x, y in [('tv_purified_control', 'purified', 'control'),
                                     ('tv_purified_wrong', 'purified', 'wrong'),
                                     ('tv_wrong_control', 'wrong', 'control'),
                                     ('tv_base_cached', 'base', 'cached_base')]:
                assert abs(row[saved_key] - tv(dist[x], dist[y])) < 1e-5
        records.append(dict(base=left['base'], kind=left['kind'],
            drift={key: tv(a[key], b[key]) for key in ['base', 'purified', 'control', 'wrong']},
            top1_changed={key: bool(a[key].argmax() != b[key].argmax())
                          for key in ['base', 'purified', 'control', 'wrong']},
            full_effect=tv(a['purified'], a['control']),
            cached_effect=tv(b['purified'], b['control'])))
    summaries = {}
    for kind in sorted({row['kind'] for row in records}):
        selected = [row for row in records if row['kind'] == kind]
        summaries[kind] = dict(n=len(selected),
            mean_full_effect=float(np.mean([r['full_effect'] for r in selected])),
            mean_cached_effect=float(np.mean([r['cached_effect'] for r in selected])),
            drift={key: dict(mean=float(np.mean([r['drift'][key] for r in selected])),
                             maximum=max(r['drift'][key] for r in selected),
                             top1_changes=sum(r['top1_changed'][key] for r in selected))
                   for key in ['base', 'purified', 'control', 'wrong']})
    report = dict(classification='POSTHOC_SAVED_LOGIT_EXECUTION_SCHEDULE_AUDIT',
        full_manifest_sha256=sha(args.full / 'MANIFEST.json'),
        cached_manifest_sha256=sha(args.cached / 'MANIFEST.json'),
        source_sha256=sha(Path(__file__)), summaries=summaries, records=records,
        scope='23 exposed questions, two selected positions each. No complete answers, training, '
              'FP32 reference distribution, or independent replication. Original v1 failed gate remains failed.')
    with args.out.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
