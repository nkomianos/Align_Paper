import copy
import pytest
import torch

from interaction_sprint.undo_gpu_training import collate, loss_for, schedule, validate_data, next_logits, qualification, loading_metadata_json
from latent_contract.sender_update import install_lora, adapter_state, load_adapter


def test_schedule_and_padding():
    plan = schedule(9, 2, 4, 7)
    assert sorted(sum(plan[:3], [])) == list(range(9))
    assert plan == schedule(9, 2, 4, 7)
    batch = collate([[3, 4], [5]], 0, "cpu")
    assert batch["input_ids"].tolist() == [[3, 4], [0, 5]]
    assert batch["position_ids"].tolist() == [[0, 1], [0, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1], [0, 1]]


def test_transformers_loading_metadata_containers():
    import json
    raw = {"missing_keys": set(), "unexpected_keys": {"b", "a"},
           "mismatched_keys": [("weight", (2, 3), (3, 2))], "nested": {"x": {("k", 1)}}}
    normalized = loading_metadata_json(raw)
    assert normalized["missing_keys"] == []
    assert normalized["unexpected_keys"] == ["a", "b"]
    assert normalized["mismatched_keys"] == [["weight", [2, 3], [3, 2]]]
    assert normalized["nested"] == {"x": [["k", 1]]}
    assert json.loads(json.dumps(normalized)) == normalized
    assert raw["unexpected_keys"] == {"b", "a"}


def test_losses_match_full_vocab_and_detach_teacher():
    logits = torch.tensor([[1., 2., 3., 4., 5.]], requires_grad=True)
    teacher = torch.tensor([[.1, .2, .3, .3, .1]], requires_grad=True)
    expected = torch.nn.functional.kl_div(logits.log_softmax(-1), teacher.detach(), reduction="batchmean")
    loss = loss_for(logits, teacher=teacher)
    assert torch.allclose(loss, expected)
    loss.backward()
    assert teacher.grad is None
    assert logits.grad[0, 4] != 0  # full vocabulary includes non-choice token
    with pytest.raises(ValueError):
        loss_for(logits, teacher=teacher * .9)
    assert torch.allclose(loss_for(logits, labels=torch.tensor([3])),
                          torch.nn.functional.cross_entropy(logits, torch.tensor([3])))


def test_split_overlap_rejected():
    row = dict(id="x", target="A", prompt="p", canonical_prompt="c", local_prompt="l")
    with pytest.raises(ValueError):
        validate_data([row], [dict(row)], [dict(row)])
    validate_data([row], [dict(row, id="y")], [dict(row, id="z")])


def test_qualification_uses_declared_controls_and_mass_only():
    rows = [dict(condition=c, prediction="A", target="A", choice_mass=.9)
            for c in ("canonical", "padded", "counterfactual") for _ in range(8)]
    rows += [dict(condition="history", prediction="B", target="A", choice_mass=.9)] * 8
    assert qualification(rows)["qualified"]
    rows[0]["prediction"] = "B"
    assert not qualification(rows)["qualified"]
    rows[0]["prediction"] = "A"
    for r in rows:
        r["choice_mass"] = .4
    assert not qualification(rows)["qualified"]


def test_tiny_qwen_training_padding_and_reset():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(44)
    cfg = Qwen3Config(vocab_size=24, hidden_size=16, intermediate_size=32,
                     num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                     head_dim=8, attention_dropout=0)
    model = Qwen3ForCausalLM(cfg).eval()
    prompts = [[1, 3, 4], [2, 7]]
    with torch.no_grad():
        original = next_logits(model, prompts, 0, "cpu")
        serial = torch.cat([next_logits(model, [p], 0, "cpu") for p in prompts])
    assert torch.allclose(original, serial, atol=1e-6)
    install_lora(model, rank=2, alpha=4)
    initial = adapter_state(model)
    assert torch.allclose(original, next_logits(model, prompts, 0, "cpu"), atol=1e-6)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    for mode in ("sft", "kl"):
        load_adapter(model, initial)
        model.train(); optimizer.zero_grad()
        logits = next_logits(model, prompts, 0, "cpu")
        target = torch.zeros_like(logits); target[:, 3] = 1
        loss = loss_for(logits, labels=torch.tensor([3, 3])) if mode == "sft" else loss_for(logits, teacher=target)
        loss.backward(); optimizer.step()
        assert any(not torch.equal(v, initial[k]) for k, v in adapter_state(model).items())
        load_adapter(model, initial); model.eval()
        assert torch.allclose(original, next_logits(model, prompts, 0, "cpu"), atol=1e-6)
