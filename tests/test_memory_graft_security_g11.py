from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


MODULE = Path(__file__).resolve().parents[1] / "scripts" / "run_memory_graft_security_g11.py"
SPEC = importlib.util.spec_from_file_location("g11", MODULE)
assert SPEC and SPEC.loader
g11 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(g11)


def test_active_row_adam_changes_only_active_rows() -> None:
    table = torch.nn.Parameter(torch.arange(20, dtype=torch.float32).reshape(5, 4))
    before = table.detach().clone()
    table.grad = torch.ones_like(table)
    optimizer = g11.ActiveRowAdam(table, lr=0.01, weight_decay=0.0, exponent=0.0)
    result = optimizer.step(torch.tensor([1, 3]), torch.tensor([2, 1]), None)
    assert result["active_rows"] == 2
    assert torch.equal(table[[0, 2, 4]], before[[0, 2, 4]])
    assert not torch.equal(table[[1, 3]], before[[1, 3]])


def test_frequency_weights_reallocate_fixed_norm() -> None:
    base = torch.zeros((3, 2), dtype=torch.float32)
    uniform = torch.nn.Parameter(base.clone()); uniform.grad = torch.ones_like(uniform)
    frequency = torch.nn.Parameter(base.clone()); frequency.grad = torch.ones_like(frequency)
    rows = torch.tensor([0, 1]); counts = torch.tensor([1, 9])
    plain = g11.ActiveRowAdam(uniform, 0.01, 0.0, 0.0)
    aware = g11.ActiveRowAdam(frequency, 0.01, 0.0, 1.0)
    target = 0.25
    first = plain.step(rows, counts, target)
    second = aware.step(rows, counts, target)
    assert abs(first["applied_update_l2"] - target) < 1e-6
    assert abs(second["applied_update_l2"] - target) < 1e-6
    assert torch.linalg.vector_norm(frequency[0]) > torch.linalg.vector_norm(frequency[1])
    assert torch.allclose(torch.linalg.vector_norm(uniform[0]), torch.linalg.vector_norm(uniform[1]))
