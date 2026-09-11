from __future__ import annotations

import torch
from torch import nn

from conditional_memory.pythia_memory_graft import ExactSuffixMemory, MultiTableEmbedding
from conditional_memory.security_s1 import (
    choose_random_control_rows,
    make_poison_training_blocks,
    student_t_interval,
    zero_hash_rows,
)


def test_poison_and_benign_exposures_are_disjoint_and_exact() -> None:
    blocks = torch.arange(40 * 32).reshape(40, 32)
    changed, rows = make_poison_training_blocks(
        blocks,
        trigger_ids=[1, 2, 3],
        payload_id=4,
        benign_ids=[5, 6, 7],
        benign_continuation_id=8,
        poison_count=6,
        seed=19,
        insertion_end=20,
    )
    assert set(rows["poison_rows"]).isdisjoint(rows["benign_rows"])
    for row in rows["poison_rows"]:
        assert changed[row, 16:20].tolist() == [1, 2, 3, 4]
    for row in rows["benign_rows"]:
        assert changed[row, 16:20].tolist() == [5, 6, 7, 8]
    untouched = set(range(40)) - set(rows["poison_rows"]) - set(rows["benign_rows"])
    assert all(torch.equal(changed[row], blocks[row]) for row in untouched)


def test_student_interval_uses_seed_as_unit() -> None:
    interval = student_t_interval([0.4, 0.5, 0.6, 0.5, 0.5])
    assert interval["mean"] == 0.5
    assert interval["lower"] < 0.5 < interval["upper"]


def test_global_rows_pack_one_row_from_each_table() -> None:
    tables = MultiTableEmbedding([3, 5, 7], embedding_dim=2)
    local = torch.tensor([2, 4, 6])
    assert tables.global_rows(local).tolist() == [2, 7, 14]


class _TinyGraft(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hash_tables = MultiTableEmbedding([3, 5, 7], embedding_dim=2)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.graft = _TinyGraft()


def test_zero_hash_rows_is_temporary_and_exact() -> None:
    model = _TinyModel()
    original = model.graft.hash_tables.embedding.weight.detach().clone()
    rows = torch.tensor([1, 5, 12])
    with zero_hash_rows(model, rows):
        assert torch.count_nonzero(model.graft.hash_tables.embedding.weight[rows]) == 0
        assert torch.equal(model.graft.hash_tables.embedding.weight[0], original[0])
    assert torch.equal(model.graft.hash_tables.embedding.weight, original)


def test_random_controls_draw_one_distinct_nonexcluded_row_per_table() -> None:
    model = _TinyModel()
    controls = choose_random_control_rows(model, {1, 5, 12}, count=20, seed=7)
    assert len(controls) == 20
    for rows in controls:
        values = rows.tolist()
        assert len(values) == 3
        assert 0 <= values[0] < 3
        assert 3 <= values[1] < 8
        assert 8 <= values[2] < 15
        assert not ({1, 5, 12} & set(values))


def test_exact_bank_is_shared_artifact_not_checkpoint_state() -> None:
    memory = ExactSuffixMemory([(1, 2), (2, 3, 4)], torch.ones(2, 5))
    assert "values" not in memory.state_dict()
    assert memory.values.shape == (2, 5)
