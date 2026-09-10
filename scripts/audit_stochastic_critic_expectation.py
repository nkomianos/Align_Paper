"""Exact developmental root-gradient check, with deterministic transitions."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
import torch
from execute_frozen_value_routine_audit import extract


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    for r in json.loads((a.source / 'DOWNLOAD.json').read_text(encoding='utf-8-sig')):
        assert hashlib.sha256((a.source / r['path'].replace('/', '__')).read_bytes()).hexdigest() == r['sha256'].lower()
    ref = json.loads((a.source / 'TRL_REFERENCE.json').read_text(encoding='utf-8-sig'))
    assert hashlib.sha256((a.source / 'trl_core_reference.py').read_bytes()).hexdigest() == ref['sha256'].lower()
    ns = {'torch': torch, 'Optional': Optional}
    extract(a.source / 'trl_core_reference.py', ['whiten', 'masked_mean', 'masked_var', 'masked_whiten'], ns)
    extract(a.source / 'dvpo__policy_model_training__policyv1_trainer.py', ['compute_advantages'], ns)
    # Root selects A/B equiprobably; a separate continuation action resolves reward.
    # A: P(1)=.2, P(0)=.8. B: P(.3)=.9, P(-1)=.1.
    # All transitions deterministic given the selected action; continuation policy fixed.
    probs = [.1, .4, .45, .05]
    means = [.2, .2, .17, .17]
    rewards = [1., 0., .3, -1.]
    scores = [.5, .5, -.5, -.5]
    rows = []
    for n, lam in itertools.product([1, 2, 4, 8], [0., .5, .95, 1.]):
        sums = dict(released=0., no_value_whitening=0., terminal_reward_control=0., raw=0., raw_terminal_reward_control=0.)
        probability_sum = 0.
        for c0 in range(n+1):
            for c1 in range(n-c0+1):
                for c2 in range(n-c0-c1+1):
                    counts = [c0, c1, c2, n-c0-c1-c2]
                    prob = math.factorial(n)
                    for c, q in zip(counts, probs):
                        prob *= q**c / math.factorial(c)
                    probability_sum += prob
                    ids = [i for i, c in enumerate(counts) for _ in range(c)]
                    head = torch.tensor([[.185, means[i], rewards[i], 0.] for i in ids], dtype=torch.float64)
                    score = torch.tensor([scores[i] for i in ids], dtype=torch.float64)
                    config = SimpleNamespace(gamma=1., lam=lam, whiten_rewards=False)
                    for mode in ['released', 'no_value_whitening', 'terminal_reward_control']:
                        val = ns['whiten'](head)[:, :-1] if mode == 'released' else head[:, :-1].clone()
                        rew = torch.zeros_like(val)
                        if mode == 'terminal_reward_control':
                            rew[:, -1] = head[:, 2]
                        _, adv, _ = ns['compute_advantages'](SimpleNamespace(config=config), val, rew, torch.ones_like(val))
                        sums[mode] += prob * float((score * adv[:, 0]).mean())
                    # Independent unnormalized recurrence; zero actual rewards.
                    raw = -head[:, 0] + (1-lam)*head[:, 1] + lam*(1-lam)*head[:, 2]
                    sums['raw'] += prob * float((score*raw).mean())
                    raw_terminal = raw + lam**2 * head[:, 2]
                    sums['raw_terminal_reward_control'] += prob * float((score*raw_terminal).mean())
        truth = (.2-.17)/4
        assert abs(probability_sum-1) < 1e-12
        assert abs(sums['raw']-truth*(1-lam**2)) < 1e-12
        assert abs(sums['raw_terminal_reward_control']-truth) < 1e-12
        rows.append(dict(batch_size=n, lam=lam, true_root_gradient=truth, **sums))
    report = dict(classification='DEVELOPMENTAL_EXACT_EXPECTATION', cases=rows,
                  scope='One specified constructed MDP; exact multinomial expectations; fixed continuation policy; '
                  'root-only parameter, no neural model or PPO clipping; compatible not authenticated historical TRL.',
                  released_negative_cases=sum(r['released'] < -1e-12 for r in rows))
    with a.out.open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
