"""Exact finite-batch masked-output intervention; no neural performance claim."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
from execute_frozen_value_routine_audit import extract


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    for row in json.loads((args.source / 'DOWNLOAD.json').read_text(encoding='utf-8-sig')):
        assert hashlib.sha256((args.source / row['path'].replace('/', '__')).read_bytes()).hexdigest() == row['sha256'].lower()
    ref = json.loads((args.source / 'TRL_REFERENCE.json').read_text(encoding='utf-8-sig'))
    assert hashlib.sha256((args.source / 'trl_core_reference.py').read_bytes()).hexdigest() == ref['sha256'].lower()
    ns = {'torch': torch, 'Optional': Optional}
    extract(args.source / 'trl_core_reference.py', ['whiten', 'masked_mean', 'masked_var', 'masked_whiten'], ns)
    extract(args.source / 'dvpo__policy_model_training__policyv1_trainer.py', ['compute_advantages'], ns)
    rows = []
    control_reference = {}
    for batch in [1, 2, 4, 8, 16, 32]:
        assert abs(sum(math.comb(batch, k) / 2**batch for k in range(batch + 1)) - 1) < 1e-12
        for lam in [.95, 1.]:
            for pads in [1, 4, 16]:
                for coefficient in [-10, -1, 0, 1, 10]:
                    expected = dict.fromkeys(['released', 'active_only', 'no_initial', 'raw', 'raw_reward'], 0.)
                    for good in range(batch + 1):
                        weight = math.comb(batch, good) / 2**batch
                        outcome = torch.tensor([1.] * good + [-1.] * (batch-good), dtype=torch.float64)
                        head = torch.cat([torch.zeros(batch, 1, dtype=torch.float64), outcome[:, None],
                                          coefficient * outcome[:, None].expand(-1, pads),
                                          torch.zeros(batch, 1, dtype=torch.float64)], dim=1)
                        head_mask = torch.zeros_like(head)
                        head_mask[:, :2] = 1
                        for mode in ['released', 'active_only', 'no_initial']:
                            supplied = (ns['whiten'](head) if mode == 'released' else
                                        ns['masked_whiten'](head, head_mask) if mode == 'active_only' else head)
                            values = supplied[:, :-1]
                            config = SimpleNamespace(gamma=1., lam=lam, whiten_rewards=False)
                            _, advantage, _ = ns['compute_advantages'](SimpleNamespace(config=config), values,
                                                                     torch.zeros_like(values), head_mask[:, :-1])
                            expected[mode] += weight * float((outcome / 2 * advantage[:, 0]).mean())
                        # Direct two-step recurrence, independently evaluated.
                        last_delta = -outcome
                        root_delta = outcome
                        raw = root_delta + lam * last_delta
                        rewarded = root_delta + lam * (outcome + last_delta)
                        expected['raw'] += weight * float((outcome / 2 * raw).mean())
                        expected['raw_reward'] += weight * float((outcome / 2 * rewarded).mean())
                    assert abs(expected['raw'] - (1-lam)/2) < 1e-12
                    assert abs(expected['raw_reward'] - .5) < 1e-12
                    for mode in ['active_only', 'no_initial']:
                        key = (batch, lam, mode)
                        if key not in control_reference:
                            control_reference[key] = expected[mode]
                        assert abs(expected[mode] - control_reference[key]) < 1e-12
                    rows.append(dict(batch=batch, lam=lam, masked_positions=pads,
                                     masked_coefficient=coefficient, **expected))
    result = {'classification': 'DEVELOPMENTAL_EXACT_MASKED_OUTPUT_INTERVENTION',
              'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'torch_version': torch.__version__, 'true_reward_gradient': .5,
              'controls_passed': True, 'rows': rows,
              'scope': 'Constructed masked outputs, exact active state values, unclipped root gradient only. '
                       'No historical dependency authentication, neural critic, optimizer or benchmark replication.'}
    with args.out.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({'rows': len(rows), 'controls_passed': True,
                      'released_negative_settings': sum(r['released'] < -1e-12 for r in rows),
                      'released_range': [min(r['released'] for r in rows), max(r['released'] for r in rows)]}))


if __name__ == '__main__':
    main()
