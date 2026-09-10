"""Exact-arithmetic diagnostic of zero-external-reward GAE; not a DVPO run."""
import argparse
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path


def gae(values, lam, gamma=F(1)):
    advantage = [F(0)] * len(values)
    carry = F(0)
    for t in reversed(range(len(values))):
        following = values[t + 1] if t + 1 < len(values) else F(0)
        carry = gamma * following - values[t] + gamma * lam * carry
        advantage[t] = carry
    return advantage


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    receipt = json.loads((a.source / 'DOWNLOAD.json').read_text(encoding='utf-8-sig'))
    for row in receipt:
        raw = (a.source / row['path'].replace('/', '__')).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == row['sha256'].lower()
        assert hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == row['git_blob_sha1']
    # Two deterministic length-two paths, a shared root value and distinct next values.
    # Critic inputs here are fixed numbers BEFORE any batch-dependent normalization.
    rows = []
    for lam in [F(0), F(1, 2), F(19, 20), F(1)]:
        good = gae([F(1, 2), F(1)], lam)
        bad = gae([F(1, 2), F(0)], lam)
        gradient = (good[0] - bad[0]) / 4  # Bernoulli logit, p=1/2.
        assert gradient == (1 - lam) / 4
        rows.append({'lambda': str(lam), 'good_advantages': list(map(str, good)),
                     'bad_advantages': list(map(str, bad)),
                     'root_expected_score_gradient': str(gradient)})
    checks = 0
    for n in range(1, 33):
        values = [F((i * 17 + n) % 23, 23) for i in range(n)]
        for gamma in [F(1), F(9, 10)]:
            assert gae(values, F(1), gamma) == [-v for v in values]
            checks += 1
    result = {'classification': 'DEVELOPMENTAL_EXACT_ARITHMETIC',
              'source_files_verified': len(receipt), 'telescoping_checks': checks,
              'two_step_examples': rows,
              'scope': 'Zero external rewards, terminal next value zero, fixed state values. '
                       'No upstream module executed, learned critic, token masking, whitening, '
                       'PPO clipping, neural gradients or empirical DVPO reproduction.'}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
