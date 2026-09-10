"""Execute only reviewed tensor routines extracted from pinned public sources."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch


def extract(path, names, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    selected = [node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {n.name for n in selected} == set(names)
    assert len(selected) == len(names)
    for node in selected:
        assert not node.decorator_list
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    source = args.source
    receipts = json.loads((source / 'DOWNLOAD.json').read_text(encoding='utf-8-sig'))
    for row in receipts:
        raw = (source / row['path'].replace('/', '__')).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == row['sha256'].lower()
    ref = json.loads((source / 'TRL_REFERENCE.json').read_text(encoding='utf-8-sig'))
    assert hashlib.sha256((source / 'trl_core_reference.py').read_bytes()).hexdigest() == ref['sha256'].lower()
    ns = {'torch': torch, 'Optional': Optional}
    extract(source / 'trl_core_reference.py', ['whiten', 'masked_mean', 'masked_var', 'masked_whiten'], ns)
    extract(source / 'dvpo__policy_model_training__policyv1_trainer.py', ['compute_advantages'], ns)

    def run(values, mask, lam=.95, rewards=None, prewhiten=True):
        supplied = ns['whiten'](values) if prewhiten else values.clone()
        config = SimpleNamespace(gamma=1., lam=lam, whiten_rewards=False)
        v, a, ret = ns['compute_advantages'](SimpleNamespace(config=config), supplied,
                         torch.zeros_like(values) if rewards is None else rewards, mask)
        return a, ret - v, v

    torch.manual_seed(20260910)
    dtype = torch.float64
    values = torch.tensor([[2., 3., .5, 1., .8, 7.], [2., 3., .5, 0., .2, 7.]], dtype=dtype)
    mask = torch.tensor([[0., 0., 1., 1., 1., 0.]] * 2, dtype=dtype)
    selected = mask.bool()
    changed = values.clone()
    changed[~selected] = 100.
    base = run(values, mask)
    altered = run(changed, mask)
    control = run(values, mask, prewhiten=False)
    control_changed = run(changed, mask, prewhiten=False)
    assert torch.equal(control[0][selected], control_changed[0][selected])
    terminal_rewards = torch.zeros_like(values)
    terminal_rewards[0, 4] = 1.
    reward_control = run(values, mask, lam=1., rewards=terminal_rewards, prewhiten=False)
    zero_control = run(values, mask, lam=1., prewhiten=False)
    assert torch.allclose(zero_control[1][selected], -values[selected], atol=1e-12)
    assert torch.allclose((reward_control[1] - zero_control[1])[0, 2:5], torch.ones(3, dtype=dtype))
    residuals = []
    for n in range(2, 34):
        v = torch.randn(3, n, dtype=dtype)
        m = torch.ones_like(v)
        _, raw, supplied = run(v, m, lam=1.)
        residuals.append(float((raw + supplied).abs().max()))
    assert max(residuals) < 1e-12
    # Append masked positions without changing any active position or its value.
    padded_v = torch.cat([values, torch.full((2, 5), 100., dtype=dtype)], dim=1)
    padded_m = torch.cat([mask, torch.zeros((2, 5), dtype=dtype)], dim=1)
    padded = run(padded_v, padded_m)
    # A critic offset is harmless to a raw state baseline, but GAE below one
    # can turn offsets into position-dependent changes at the terminal boundary.
    shifted = run(values + 50., mask)
    assert torch.allclose(base[0][selected], shifted[0][selected], atol=1e-12)
    extra_v = torch.tensor([[0., 0., 10., -10., 20., 0.]], dtype=dtype)
    combined = run(torch.cat([values, extra_v]), torch.cat([mask, mask[:1]]))
    prompt_v = torch.cat([torch.full((2, 4), 100., dtype=dtype), values], dim=1)
    prompt_m = torch.cat([torch.zeros((2, 4), dtype=dtype), mask], dim=1)
    prompted = run(prompt_v, prompt_m)
    report = {'classification': 'DEVELOPMENTAL_EXECUTED_SOURCE_ROUTINE',
              'torch_version': torch.__version__, 'trl_reference': ref,
              'lambda_one_max_raw_identity_error': max(residuals),
              'masked_value_perturbation_max_advantage_change': float((base[0] - altered[0])[selected].abs().max()),
              'append_masked_padding_max_advantage_change': float((base[0][selected] - padded[0][padded_m.bool()]).abs().max()),
              'prepend_masked_prompt_max_advantage_change': float((base[0][selected] - prompted[0][prompt_m.bool()]).abs().max()),
              'extra_batch_member_max_advantage_change': float((base[0][selected] - combined[0][:2][selected]).abs().max()),
              'uniform_offset_max_advantage_change': float((base[0][selected] - shifted[0][selected]).abs().max()),
              'without_prewhitening_max_change': float((control[0] - control_changed[0])[selected].abs().max()),
              'terminal_reward_control_passed': True,
              'active_advantages': base[0][selected].tolist(),
              'perturbed_active_advantages': altered[0][selected].tolist(),
              'limitations': ['TRL compatibility version is explicit, not historical dependency authentication.',
                  'No actor/critic model loaded; no full trainer, optimizer, clipping or neural performance reproduced.',
                  'Controlled masked values are diagnostic interventions, not observed model padding values.']}
    with args.out.open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
