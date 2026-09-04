import math

import pytest
import torch

from interaction_sprint.hindsight_bifurcation_lora import (
    binary_reverse_kl,
    initial_probability,
    logit,
    schedule,
)


def test_initialization_straddles_middle():
    assert initial_probability(0.5, -1) == pytest.approx(0.49)
    assert initial_probability(0.5, 1) == pytest.approx(0.51)
    assert initial_probability(0.005, -1) > 0
    assert initial_probability(0.995, 1) < 1


def test_binary_reverse_kl_and_gradient():
    p = torch.tensor([0.2, 0.8], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([0.2, 0.5], dtype=torch.float64)
    value = binary_reverse_kl(p, q)
    assert float(value[0].detach()) == pytest.approx(0.0)
    assert float(value[1].detach()) == pytest.approx(0.8 * math.log(1.6) + 0.2 * math.log(0.4))
    value.sum().backward()
    assert torch.isfinite(p.grad).all()


def test_schedule_is_balanced_and_reproducible():
    ids = [f"x-{index}" for index in range(16)]
    first = schedule(ids)
    assert first == schedule(ids)
    assert len(first) == 64 and all(len(batch) == 8 for batch in first)
    flattened = [item for batch in first for item in batch]
    assert all(flattened.count(item) == 32 for item in ids)


def test_logit_round_trip():
    for value in (0.01, 0.2, 0.5, 0.9):
        assert 1 / (1 + math.exp(-logit(value))) == pytest.approx(value)
