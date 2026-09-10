import pytest
import torch
from action_marginal_policy import initial_offsets, policy, exact_gradient_moments


def test_initial_behavior_matches_with_unequal_alias_counts():
    ids = torch.tensor([0, 0, 0, 1], dtype=torch.long)
    scores = torch.tensor([2., -3., 4., -8.], dtype=torch.double, requires_grad=True)
    offsets = initial_offsets(scores, ids)
    _, masses = policy(scores, ids, offsets)
    assert torch.allclose(masses.exp(), torch.tensor([.5, .5], dtype=torch.double))
    assert not offsets.requires_grad
    changed = scores + torch.tensor([1., 1., 1., 0.])
    assert policy(changed, ids, offsets)[1].exp()[0] > .5


def test_parameter_gradient_mean_matches_and_variance_decreases():
    # Shared nonlinear parameters ensure this tests more than independent logits.
    theta = torch.tensor([.2, -.4, .7], dtype=torch.double, requires_grad=True)
    features = torch.tensor([[1., 2., 0.], [0., -1., 1.],
                             [2., 0., -1.], [-1., 1., 2.]], dtype=torch.double)
    scores = (features @ theta).tanh()
    ids = torch.tensor([0, 0, 1, 1])
    offsets = initial_offsets(scores, ids)
    rewards = torch.tensor([0., 1.], dtype=torch.double)
    result = exact_gradient_moments(scores, ids, offsets, rewards, [theta])
    raw, marginal = result['spelling'], result['marginal']
    expected = torch.autograd.grad(-(policy(scores, ids, offsets)[1].exp() * rewards).sum(), theta)[0]
    assert torch.allclose(raw['mean'], expected, atol=1e-12)
    assert torch.allclose(marginal['mean'], expected, atol=1e-12)
    assert marginal['variance_trace'] < raw['variance_trace']


def test_canonical_policy_has_no_conditional_variance_to_remove():
    theta = torch.tensor([.3, -.7], dtype=torch.double, requires_grad=True)
    ids = torch.tensor([0, 1])
    result = exact_gradient_moments(theta, ids, initial_offsets(theta, ids),
                                    torch.tensor([0., 1.]), [theta])
    assert torch.equal(result['spelling']['mean'], result['marginal']['mean'])
    assert torch.equal(result['spelling']['variance_trace'], result['marginal']['variance_trace'])


def test_reject_offsets_that_would_change_gradient_estimand():
    scores = torch.tensor([.1, .4], requires_grad=True)
    with pytest.raises(ValueError, match='frozen'):
        policy(scores, torch.tensor([0, 1]), scores)
