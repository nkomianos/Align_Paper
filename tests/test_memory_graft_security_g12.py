from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


MODULE = Path(__file__).resolve().parents[1] / "scripts" / "run_memory_graft_security_g12.py"
SPEC = importlib.util.spec_from_file_location("g12", MODULE)
assert SPEC and SPEC.loader
g12 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(g12)


def test_optimizer_state_can_be_restored_and_deleted_separately() -> None:
    table = torch.nn.Parameter(torch.zeros((5, 3)))
    rows = torch.tensor([1, 4])
    state = {"step": torch.tensor(7.0), "exp_avg": torch.ones((2, 3)),
             "exp_avg_sq": torch.full((2, 3), 2.0)}
    optimizer = g12.fresh_optimizer_with_target_state(table, 1e-3, rows, state)
    actual = optimizer.state[table]
    assert torch.equal(actual["exp_avg"][rows], state["exp_avg"])
    assert torch.equal(actual["exp_avg_sq"][rows], state["exp_avg_sq"])
    assert torch.count_nonzero(actual["exp_avg"][[0, 2, 3]]) == 0
    actual["exp_avg"][rows] = 0
    actual["exp_avg_sq"][rows] = 0
    assert torch.count_nonzero(actual["exp_avg"]) == 0
    assert torch.count_nonzero(actual["exp_avg_sq"]) == 0
