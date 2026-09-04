import torch
from torch import nn
from collections import Counter

from interaction_sprint.hindsight_neural_anchor import (
    augmented_reverse_kl,
    build_records,
    install_qwen35_lora,
    record_invariants,
    reverse_kl_per_example,
    semantic_letter_index,
)
from latent_contract.sender_update import LoRALinear


def test_data_has_exact_matched_logs_and_representative_anchors():
    rows, evaluation, batches = build_records()
    summary = record_invariants(rows)
    assert summary == {
        "n": 128,
        "z0": {0: 32, 1: 96},
        "logged_action": {0: 64, 1: 64},
        "strata": {"z0-a0": 16, "z0-a1": 16, "z1-a0": 48, "z1-a1": 48},
        "anchors": 32,
        "anchor_strata": {"z0-a0": 4, "z0-a1": 4, "z1-a0": 12, "z1-a1": 12},
        "immediate_world_mismatches": 0,
    }
    assert len(evaluation) == 32
    assert len(batches) == 64
    assert all(len(batch) == 16 for batch in batches)
    by_id = {row["id"]: row for row in rows}
    assert all(sum(bool(by_id[row_id]["anchor"]) for row_id in batch) == 4 for batch in batches)
    assert Counter(row_id for batch in batches for row_id in batch) == Counter({row["id"]: 8 for row in rows})


def test_matched_immediate_log_implies_opposite_delayed_population_optima():
    rows, _, _ = build_records()
    # The two worlds expose the same immediate field by construction.
    assert all(row["immediate_semantic"] == row["delayed_transition_semantic"] for row in rows)
    # Raw/transition feedback favors concise, while persistent expression anchors
    # favor detailed; these are the opposite policies the neural gate must learn.
    assert sum(row["immediate_semantic"] == 0 for row in rows) == 80
    assert sum(row["delayed_expression_semantic"] == 1 for row in rows) == 96


def test_semantic_letter_respects_option_swap():
    assert [semantic_letter_index(s, swap) for swap in (0, 1) for s in (0, 1)] == [0, 1, 1, 0]


def test_augmented_loss_equals_raw_when_delayed_teacher_is_immediate_teacher():
    torch.manual_seed(2)
    student = torch.randn(6, 11, requires_grad=True)
    teacher = torch.randn(6, 11)
    mask = torch.tensor([1, 0, 1, 0, 0, 1], dtype=torch.bool)
    raw = reverse_kl_per_example(student, teacher).mean()
    augmented = augmented_reverse_kl(student, teacher, teacher[mask], mask)
    assert torch.allclose(raw, augmented, atol=1e-7, rtol=1e-7)
    augmented.backward()
    assert torch.isfinite(student.grad).all()


def test_augmented_loss_replaces_anchor_component_in_expectation_formula():
    student = torch.tensor([[2., 0.], [0., 2.], [1., 1.]], requires_grad=True)
    immediate = torch.tensor([[3., 0.], [0., 3.], [2., 0.]])
    delayed = torch.tensor([[0., 3.], [0., 3.]])
    mask = torch.tensor([1, 0, 1], dtype=torch.bool)
    per_immediate = reverse_kl_per_example(student, immediate)
    per_delayed = reverse_kl_per_example(student[mask], delayed)
    expected = per_immediate.mean() + (per_delayed - per_immediate[mask]).mean()
    assert torch.allclose(augmented_reverse_kl(student, immediate, delayed, mask), expected)


class _ToyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.in_proj_qkv = nn.Linear(4, 12)
        self.unrelated = nn.Linear(4, 4)


def test_qwen35_lora_covers_attention_and_deltanet_leaves_only():
    model = _ToyLayer()
    names = install_qwen35_lora(model, rank=2, alpha=4)
    assert names == ["q_proj", "in_proj_qkv"]
    assert isinstance(model.q_proj, LoRALinear)
    assert isinstance(model.in_proj_qkv, LoRALinear)
    assert isinstance(model.unrelated, nn.Linear)
