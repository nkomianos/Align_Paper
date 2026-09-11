"""Execute isolated pinned loss functions on benign tensors, not the training entry point."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    source = ROOT / 'artifacts/reso_source_screen_20260910'
    receipt = json.loads((source / 'SOURCE_RECEIPTS.json').read_text())
    for row in receipt['files']:
        assert sha(source / row['path']) == row['sha256']
    path = source / 'training/reso_train.py'
    wanted = {'pooled_forward', 'structure_loss', 'token_kl', 'shifted_nll'}
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    assert {node.name for node in functions} == wanted
    namespace = {'torch': torch, 'F': F}
    # No imports, file loading, training calls or dataset code from the release.
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    tests = []

    policy = torch.tensor([[[1., -.5, 0.], [.2, .8, -1.], [-2., 3., 1.]],
                           [[0., .3, 1.], [1., 0., -1.], [.5, -.5, 1.]]], requires_grad=True)
    reference = torch.tensor([[[0., 1., -1.], [.1, -.2, .6], [5., -2., 0.]],
                              [[.4, .3, -.5], [1., 2., -3.], [0., 1., 2.]]])
    mask = torch.tensor([[True, True, False], [True, False, False]])
    loss = namespace['token_kl'](policy, reference, mask, chunk=1)
    p = policy.detach().double().softmax(-1)
    q = reference.double().softmax(-1)
    expected = (q * (q.log() - p.log())).sum(-1)[mask].mean()
    assert abs(float(loss.detach()) - float(expected)) < 2e-7
    loss.backward()
    expected_gradient = torch.where(mask[..., None], (p - q) / mask.sum(), 0.)
    assert torch.allclose(policy.grad.double(), expected_gradient, atol=5e-8, rtol=0)
    tests.append('forward_KL_value_and_policy_gradient')

    altered_policy, altered_reference = policy.detach().clone(), reference.clone()
    altered_policy[~mask] = torch.tensor([100., -100., 33.])
    altered_reference[~mask] = torch.tensor([-80., 70., 99.])
    changed = namespace['token_kl'](altered_policy, altered_reference, mask)
    assert abs(float(loss.detach()) - float(changed)) < 2e-7
    tests.append('masked_logits_do_not_change_KL')
    assert abs(float(namespace['token_kl'](reference, reference, mask))) < 1e-7
    tests.append('identical_distributions_zero_KL')
    empty_policy = policy.detach().clone().requires_grad_(True)
    empty = namespace['token_kl'](empty_policy, reference, torch.zeros_like(mask))
    empty.backward()
    assert float(empty.detach()) == 0 and torch.equal(empty_policy.grad, torch.zeros_like(empty_policy))
    tests.append('empty_mask_zero_value_and_gradient')

    hidden = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).requires_grad_(True)
    fake_model = lambda **kwargs: SimpleNamespace(hidden_states=(hidden * 0, hidden, hidden * 2))
    enc = dict(input_ids=torch.zeros((2, 3), dtype=torch.long), attention_mask=mask.long())
    pooled = namespace['pooled_forward'](fake_model, enc, 2)
    explicit = torch.stack([torch.stack([hidden[i][mask[i]].mean(0) for i in range(2)]),
                            torch.stack([2 * hidden[i][mask[i]].mean(0) for i in range(2)])])
    assert torch.equal(pooled, explicit)
    pooled.sum().backward()
    assert torch.equal(hidden.grad[~mask], torch.zeros_like(hidden.grad[~mask]))
    tests.append('pooling_value_and_masked_gradient')

    z = torch.tensor([[[1., 0.], [.8, .6], [-1., 0.]]], requires_grad=True)
    trip = dict(i=np.array([0]), j=np.array([1]), k=np.array([2]))
    structure, _ = namespace['structure_loss'](z, trip, .1, .5, torch.tensor([1.]), 'cpu')
    scalar_gap = ((z[0, 0] * z[0, 1]).sum() - (z[0, 0] * z[0, 2]).sum() - .1) / .5
    assert torch.allclose(structure, torch.logaddexp(torch.zeros_like(scalar_gap), -scalar_gap))
    structure.backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0
    tests.append('structure_ranking_loss_and_nonzero_gradient')

    ids = torch.tensor([[0, 1, 2], [1, 0, 2]])
    nll, n = namespace['shifted_nll'](policy.detach(), ids, mask.long())
    assert n == 1
    assert torch.allclose(nll, -policy.detach()[0, 0].log_softmax(-1)[1])
    tests.append('NLL_real_to_real_transition_mask')

    result = dict(classification='DEVELOPMENTAL_ISOLATED_SOURCE_FUNCTION_CHECKS',
                  revision=receipt['revision'], source_sha256=sha(path), runner_sha256=sha(Path(__file__)),
                  torch_version=torch.__version__, checks_passed=tests,
                  scope='Constructed float32 tensors and a fake hidden-state provider. No neural model, '
                        'training entry point, dataset prompts, safety evaluation or published-results replication.')
    out = source / 'LOSS_PRIMITIVES_AUDIT_20260911.json'
    with out.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
