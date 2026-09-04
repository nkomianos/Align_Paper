import numpy as np
import pytest
import torch

from interaction_sprint.parameter_probe import apply_direction, cases, channel, cosine, flat_gradient
from interaction_sprint.feedback_gradient_audit import gradients


def test_cases_balanced_distinct_and_split():
    data = cases()
    assert len(data) == 16
    assert len({c["prompt"] for c in data}) == 16
    for split in ("train", "heldout"):
        group = [c for c in data if c["split"] == split]
        assert len(group) == 8 and sum(c["answer"] for c in group) == 4


def test_channel_null_and_positive_support():
    for truth in (0, 1):
        for rho in (0, .5, .9):
            k = channel(truth, rho)
            assert np.all(k > 0) and np.allclose(k.sum(1), 1)
        assert np.array_equal(channel(truth, 0)[0], channel(truth, 0)[1])


def test_chain_rule_matches_direct_autograd_in_shared_parameter_model():
    w = torch.tensor([.2, -.3, .7], dtype=torch.float64, requires_grad=True)
    x = torch.tensor([2., -1., .5], dtype=torch.float64)
    z = torch.dot(w, x)
    p = float(z.detach().sigmoid())
    q = torch.tensor([[.9, .1], [.3, .7]], dtype=torch.float64)
    k = channel(1, .5)
    joint = np.array([1-p, p])[:, None]*k
    lp = torch.stack((z*0, z)).log_softmax(0)
    coefficient = (q.T.log()-lp[:, None]).detach()
    loss = -(torch.tensor(joint)*coefficient*lp[:, None]).sum()
    direct = -torch.autograd.grad(loss, w, retain_graph=True)[0]
    analytic = flat_gradient(z, [w])*gradients(p, k, q.numpy())["sampled_response_ascent"]
    assert torch.allclose(direct, analytic, atol=1e-12)


def test_norm_matched_updates_and_dimension_guard():
    params = [torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(3))]
    direction = torch.arange(1., 6.)
    assert apply_direction(params, direction, .1) == pytest.approx(.1)
    assert torch.cat(params).norm().item() == pytest.approx(.1)
    assert cosine(direction, -direction) == pytest.approx(-1.)
    with pytest.raises(ValueError):
        apply_direction(params, torch.zeros(5), .1)
    with pytest.raises(ValueError):
        apply_direction(params, torch.ones(4), .1)
