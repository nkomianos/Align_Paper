"""Exact on-policy batch enumeration using the released isolated advantage routine."""
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
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    for row in json.loads((a.source / 'DOWNLOAD.json').read_text(encoding='utf-8-sig')):
        assert hashlib.sha256((a.source / row['path'].replace('/', '__')).read_bytes()).hexdigest() == row['sha256'].lower()
    reference = json.loads((a.source / 'TRL_REFERENCE.json').read_text(encoding='utf-8-sig'))
    assert hashlib.sha256((a.source / 'trl_core_reference.py').read_bytes()).hexdigest() == reference['sha256'].lower()
    ns = {'torch': torch, 'Optional': Optional}
    extract(a.source / 'trl_core_reference.py', ['whiten', 'masked_mean', 'masked_var', 'masked_whiten'], ns)
    extract(a.source / 'dvpo__policy_model_training__policyv1_trainer.py', ['compute_advantages'], ns)
    rows = []
    for batch in [1, 2, 4, 8, 16, 32]:
        for lam in [.0, .5, .95, 1.]:
            normalized = raw_expected = 0.
            for good_count in range(batch + 1):
                prob = math.comb(batch, good_count) / 2 ** batch
                outcomes = torch.tensor([1.] * good_count + [-1.] * (batch - good_count), dtype=torch.float64)
                # Shared root, known outcome after root action, terminal output head.
                # Actual terminal reward is outcome; middle state value is exact.
                head = torch.stack([torch.zeros_like(outcomes), outcomes, torch.zeros_like(outcomes)], dim=1)
                supplied = ns['whiten'](head)[:, :-1]
                config = SimpleNamespace(gamma=1., lam=lam, whiten_rewards=False)
                values, adv, returns = ns['compute_advantages'](SimpleNamespace(config=config), supplied,
                    torch.zeros_like(supplied), torch.ones_like(supplied))
                score = outcomes / 2  # d log pi(a) / d root logit at p=.5.
                normalized += prob * float((score * adv[:, 0]).mean())
                # Without either whitening operation, exact GAE root=(1-lambda)*outcome.
                raw_expected += prob * float((score * ((1-lam)*outcomes)).mean())
            assert abs(raw_expected - (1-lam)/2) < 1e-12
            rows.append({'batch_size': batch, 'lambda': lam,
                         'released_normalized_root_gradient': normalized,
                         'unnormalized_root_gradient': raw_expected,
                         'true_terminal_reward_gradient': .5})
    report = {'classification': 'DEVELOPMENTAL_EXACT_BATCH_EXPECTATION', 'cases': rows,
              'interpretation': 'Positive normalization-induced root signal survives lambda=1 despite raw telescoping. '
                  'In this symmetric calibrated example its sign agrees with terminal reward; this is not a harmful-update counterexample.',
              'scope': ['Every binomial batch composition enumerated; no Monte Carlo estimation.',
                  'Two-step constructed MDP, no trained neural critic or PPO clipping.',
                  'Reported gradients average root decisions only, not all sequence tokens.',
                  'Compatible TRL helper version, not authenticated historical dependencies.',
                  'Zero KL at the reference policy; true terminal reward used only for analytic comparison.']}
    with a.out.open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
