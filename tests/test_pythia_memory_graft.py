from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from conditional_memory.pythia_memory_graft import (
    EngramHashAddressor,
    ExactSuffixMemory,
    GraftConfig,
    MemoryGraftResidual,
)


def config() -> GraftConfig:
    return GraftConfig(
        layer_index=1,
        min_ngram=2,
        max_ngram=3,
        hash_heads=2,
        hash_rows_per_head=17,
        hash_embedding_dim=4,
        conv_kernel_size=2,
        hash_seed=7,
    )


def test_exact_memory_uses_longest_suffix() -> None:
    memory = ExactSuffixMemory(
        [(2, 3), (1, 2, 3)],
        torch.tensor([[2.0, 3.0], [1.0, 2.0]]),
    )
    rows = memory.address(torch.tensor([[1, 2, 3, 2, 3]]))
    assert rows.tolist() == [[-1, -1, 1, -1, 0]]
    assert not memory.values.requires_grad


def test_hash_addressing_is_deterministic_and_prefix_masked() -> None:
    compression = np.arange(32, dtype=np.int64)
    addressor = EngramHashAddressor(compression, config(), pad_token_id=0)
    ids = torch.tensor([[1, 2, 3, 4]])
    rows_a, valid_a = addressor.address(ids)
    rows_b, valid_b = addressor.address(ids)
    assert torch.equal(rows_a, rows_b)
    assert torch.equal(valid_a, valid_b)
    assert rows_a.shape == (1, 4, 4)
    assert valid_a[0, 0].sum().item() == 0
    assert valid_a[0, 1].sum().item() == 2
    assert valid_a[0, 2].sum().item() == 4


def test_residual_exercises_exact_and_fallback_routes() -> None:
    cfg = config()
    memory = ExactSuffixMemory([(1, 2)], torch.ones(1, 8))
    addressor = EngramHashAddressor(np.arange(32), cfg, pad_token_id=0)
    module = MemoryGraftResidual(8, memory, addressor, cfg)
    ids = torch.tensor([[1, 2, 9]])
    exact = memory.address(ids)
    rows, valid = addressor.address(ids)
    from conditional_memory.pythia_memory_graft import AddressPlan

    module.set_plan(AddressPlan(exact, rows, valid))
    hidden = torch.randn(1, 3, 8)
    output = module(hidden)
    assert output.shape == hidden.shape
    assert torch.isfinite(output).all()
    assert not torch.equal(output, hidden)
    module.clear_plan()
    with pytest.raises(RuntimeError, match="address plan"):
        module(hidden)


def test_duplicate_exact_keys_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        ExactSuffixMemory([(1, 2), (1, 2)], torch.zeros(2, 3))
