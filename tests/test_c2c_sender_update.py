import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from latent_contract.sender_update import (LoRALinear, adapter_state, load_adapter, install_lora,
    merge_lora, supervised_example, collate, completion_log_probs, train_epoch)


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(12, 8)
        self.q_proj = nn.Linear(8, 8)
        self.head = nn.Linear(8, 12)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        return SimpleNamespace(logits=self.head(self.q_proj(self.embed(input_ids))))


def test_zero_adapter_roundtrip_and_merged_forward(tmp_path):
    torch.manual_seed(5)
    model = TinyLM().eval()
    ids = torch.tensor([[1, 2, 3]])
    before = model(ids).logits.detach()
    assert install_lora(model, rank=2, alpha=4) == ["q_proj"]
    torch.testing.assert_close(model(ids).logits, before, rtol=0, atol=0)
    torch.save(adapter_state(model), tmp_path / "adapter.pt")
    load_adapter(model, torch.load(tmp_path / "adapter.pt", weights_only=True))
    torch.testing.assert_close(model(ids).logits, before, rtol=0, atol=0)
    with torch.no_grad():
        model.q_proj.b.normal_()
    expected = model(ids).logits.detach()
    merge_lora(model)
    assert not any(isinstance(m, LoRALinear) for m in model.modules())
    torch.testing.assert_close(model(ids).logits, expected, rtol=1e-5, atol=1e-6)


def test_masking_and_completion_shift():
    example = supervised_example([1, 2], [3, 4])
    batch = collate([example, supervised_example([1], [2])], 0, "cpu")
    assert batch["labels"].tolist() == [[-100, -100, 3, 4], [-100, 2, -100, -100]]
    probs = completion_log_probs(torch.zeros(2, 4, 5), batch["labels"])
    torch.testing.assert_close(probs, -torch.log(torch.tensor(5.0))*torch.tensor([2., 1.]))
    with pytest.raises(ValueError):
        supervised_example([1, 2], [3], max_length=2)


def test_one_epoch_changes_only_adapters_and_covers_every_example():
    torch.manual_seed(1)
    model = TinyLM()
    install_lora(model, rank=2, alpha=4)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    original = adapter_state(model)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    rows = [supervised_example([1, 2], [3])] * 9
    steps = []
    assert train_epoch(model, rows, opt, seed=11, micro_batch=2, accumulation=2, callback=steps.append) == 3
    assert sorted(i for s in steps for i in s["example_indices"]) == list(range(9))
    assert any(not torch.equal(original[n], t) for n, t in adapter_state(model).items())
    for n, p in model.named_parameters():
        if n in frozen:
            torch.testing.assert_close(p, frozen[n], rtol=0, atol=0)


def test_gradient_accumulation_matches_full_batch():
    torch.manual_seed(2)
    model = TinyLM()
    install_lora(model, rank=2, alpha=4)
    other = copy.deepcopy(model)
    examples = [supervised_example([1, 2], [3]), supervised_example([1], [4, 5])] * 3
    for m, micro, accum in ((model, 1, 4), (other, 4, 1)):
        optimizer = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=.1)
        train_epoch(m, examples, optimizer, seed=7, micro_batch=micro, accumulation=accum)
    for name, tensor in adapter_state(model).items():
        torch.testing.assert_close(tensor, adapter_state(other)[name], rtol=1e-5, atol=1e-6)


def test_invalid_adapter_rejected():
    model = TinyLM()
    install_lora(model, rank=2)
    state = adapter_state(model)
    state["q_proj.b"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        load_adapter(model, state)


def test_native_tiny_qwen3_training_export_and_reload(tmp_path):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(3)
    config = Qwen3Config(vocab_size=20, hidden_size=32, intermediate_size=48,
                        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                        attention_dropout=0, tie_word_embeddings=False)
    model = Qwen3ForCausalLM(config)
    ids = torch.tensor([[1, 2, 3]])
    model.eval()
    before = model(ids).logits.detach()
    assert len(install_lora(model, rank=2, alpha=4)) == 8
    torch.testing.assert_close(model(ids).logits, before, atol=0, rtol=0)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    train_epoch(model, [supervised_example([1, 2], [3])]*4, optimizer, seed=9)
    model.gradient_checkpointing_disable()
    model.eval()
    adapted = model(ids).logits.detach()
    assert not torch.equal(adapted, before)
    merge_lora(model)
    torch.testing.assert_close(model(ids).logits, adapted, atol=1e-5, rtol=1e-4)
    model.save_pretrained(tmp_path, safe_serialization=True)
    loaded = Qwen3ForCausalLM.from_pretrained(tmp_path, local_files_only=True,
                    attn_implementation=model.config._attn_implementation).eval()
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], tensor, atol=0, rtol=0)
    torch.testing.assert_close(loaded(ids).logits, model(ids).logits, atol=1e-7, rtol=1e-5)


def test_qualification_requires_both_retention_and_useful_update():
    from scripts.verify_c2c_sender_update import qualify, summarize
    base = {"accuracy": .9, "choice_ce": .5, "parse_rate": 1.0}
    useful = {"accuracy": .9, "choice_ce": .45, "parse_rate": 1.0}
    assert qualify(base, useful)["decision"] == "QUALIFIED_FOR_PAIRED_INTERFACE_MEASUREMENT"
    assert not qualify(base, {**useful, "accuracy": .8})["retained_accuracy"]
    assert not qualify(base, {**useful, "choice_ce": .5})["choice_ce_improved"]
    assert not qualify(base, {**useful, "parse_rate": .5})["format_valid"]
    row = {"case_id": "x", "completion": "A. option", "hit_token_limit": False,
           "choice_sequence_logps": dict.fromkeys("ABCD", -1000.)}
    summary = summarize([row], {"x": "A"})
    assert summary["accuracy"] == 1
    assert summary["choice_ce"] == pytest.approx(1.38629436)
