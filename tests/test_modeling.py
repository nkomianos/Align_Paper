from __future__ import annotations

import sys
import types

import pytest

from types import SimpleNamespace

import torch

import under_extinction.modeling as modeling
from under_extinction.modeling import (
    CHAT_TEMPLATE_KWARGS,
    QWEN35_LORA_TARGETS,
    ChoiceCollator,
    ChoiceSFTDataset,
    chat_prompt_text,
    chat_template_runtime_attestation,
    configured_model_loader,
    deltanet_kernel_attestation,
    encode_prompt_and_choice,
    inspect_lora_target_inventory,
    load_base_model,
    score_choice_batch,
    verify_choice_tokens,
)


class CharacterTokenizer:
    chat_template = "mock"
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<{message['role']}>{message['content']}" for message in messages)
        if add_generation_prompt:
            text += "<assistant>"
        if tokenize:
            return [ord(char) + 1 for char in text]
        return text

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(char) + 1 for char in text]}


def _record():
    return {
        "record_id": "example",
        "messages": [{"role": "system", "content": "choose"}, {"role": "user", "content": "now"}],
        "oracle_actions": {"intended": "A", "proxy": "B", "cached": "A"},
    }


def test_choice_completion_is_the_only_supervised_span():
    tokenizer = CharacterTokenizer()
    dataset = ChoiceSFTDataset([_record()], tokenizer, "intended", 512)
    item = dataset[0]
    supervised = [label for label in item["labels"] if label != -100]
    assert supervised == [ord("A") + 1]
    assert len(item["input_ids"]) == len(item["labels"])


def test_choice_token_verification_and_padding():
    tokenizer = CharacterTokenizer()
    details = verify_choice_tokens(tokenizer, _record()["messages"], ["A", "B"])
    assert details["equal_token_counts"]
    assert details["all_single_token"]
    dataset = ChoiceSFTDataset([_record()], tokenizer, "intended", 512)
    batch = ChoiceCollator(tokenizer.pad_token_id)([dataset[0], dataset[0]])
    assert tuple(batch["input_ids"].shape)[0] == 2
    assert (batch["labels"] != -100).sum().item() == 2


def test_too_short_max_length_fails():
    tokenizer = CharacterTokenizer()
    with pytest.raises(ValueError, match="remove the complete generation prefix"):
        encode_prompt_and_choice(tokenizer, _record()["messages"], "A", 1)


class AlwaysFavorAModel:
    device = torch.device("cpu")

    def __call__(self, input_ids, attention_mask):
        batch, length = input_ids.shape
        logits = torch.zeros((batch, length, 256), dtype=torch.float32)
        logits[:, :, ord("A") + 1] = 3.0
        logits[:, :, ord("B") + 1] = -1.0
        return SimpleNamespace(logits=logits)


def test_sequence_scorer_uses_completion_logits_and_normalizes():
    result = score_choice_batch(
        AlwaysFavorAModel(), CharacterTokenizer(), [_record()], ["A", "B"], max_length=512
    )[0]
    assert result["logp_A"] > result["logp_B"]
    assert result["probability_A"] > 0.95


def test_qwen35_chat_template_is_explicitly_non_thinking_and_attested():
    class ThinkingTokenizer(CharacterTokenizer):
        def apply_chat_template(
            self,
            messages,
            tokenize=False,
            add_generation_prompt=False,
            *,
            enable_thinking=True,
        ):
            assert enable_thinking is False
            prefix = super().apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
            )
            return prefix + "<think>\n\n</think>\n\n"

    config = {
        "model": {
            "id": "Qwen/Qwen3.5-9B",
            "revision": "a" * 40,
            "loader_class": "Qwen3_5ForCausalLM",
            "text_only": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    }
    tokenizer = ThinkingTokenizer()
    rendered = chat_prompt_text(tokenizer, _record()["messages"])
    assert rendered.endswith("<think>\n\n</think>\n\n")
    attestation = chat_template_runtime_attestation(tokenizer, config)
    assert attestation["template_kwargs"] == CHAT_TEMPLATE_KWARGS
    assert attestation["kwargs_supported"] is True
    assert attestation["closed_reasoning_preamble_observed"] is True
    assert configured_model_loader(config) == "Qwen3_5ForCausalLM"


def test_generic_chat_template_without_extra_kwargs_remains_supported():
    tokenizer = CharacterTokenizer()
    assert chat_prompt_text(tokenizer, _record()["messages"]).endswith("<assistant>")
    attestation = chat_template_runtime_attestation(tokenizer)
    assert attestation["kwargs_supported"] is False


def test_qwen35_lora_inventory_covers_deltanet_attention_and_mlp_exactly():
    class Delta(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
                setattr(self, name, torch.nn.Linear(2, 2, bias=False))

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                setattr(self, name, torch.nn.Linear(2, 2, bias=False))

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for name in ("gate_proj", "up_proj", "down_proj"):
                setattr(self, name, torch.nn.Linear(2, 2, bias=False))

    class Layer(torch.nn.Module):
        def __init__(self, layer_type):
            super().__init__()
            if layer_type == "linear_attention":
                self.linear_attn = Delta()
            else:
                self.self_attn = Attention()
            self.mlp = MLP()

    layer_types = [
        "full_attention" if index % 4 == 3 else "linear_attention"
        for index in range(32)
    ]

    class TinyHybrid(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer(value) for value in layer_types])
            self.config = SimpleNamespace(layer_types=layer_types)

    config = {
        "model": {"id": "Qwen/Qwen3.5-9B", "loader_class": "Qwen3_5ForCausalLM"},
        "training": {"lora_targets": list(QWEN35_LORA_TARGETS)},
    }
    inventory = inspect_lora_target_inventory(config, TinyHybrid())
    assert inventory["matched_module_count"] == 248
    assert inventory["groups"]["deltanet"]["matched_module_count"] == 120
    assert inventory["groups"]["full_attention"]["matched_module_count"] == 32
    assert inventory["groups"]["mlp"]["matched_module_count"] == 96


def test_lora_inventory_rejects_silent_missing_targets():
    config = {
        "model": {"id": "example/model"},
        "training": {"lora_targets": ["q_proj", "missing_proj"]},
    }
    model = torch.nn.Module()
    model.q_proj = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError, match="matched no linear modules"):
        inspect_lora_target_inventory(config, model)


def test_qwen35_loader_explicitly_disables_optional_kernels(monkeypatch):
    observed: dict[str, object] = {}

    class FakeLoadedModel:
        def __init__(self):
            self.config = SimpleNamespace(use_cache=True)

    class FakeQwenLoader:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            observed["model_id"] = model_id
            observed.update(kwargs)
            return FakeLoadedModel()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = object()
    fake_transformers.Qwen3_5ForCausalLM = FakeQwenLoader
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(modeling, "base_model_runtime_attestation", lambda *args: {})
    monkeypatch.setattr(modeling, "deltanet_kernel_attestation", lambda *args: {})
    config = {
        "model": {
            "id": "Qwen/Qwen3.5-9B",
            "revision": "a" * 40,
            "loader_class": "Qwen3_5ForCausalLM",
            "text_only": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "delta_net_kernel_policy": "torch_fallback_required",
            "dtype": "bfloat16",
            "attention": "sdpa",
        }
    }
    loaded = load_base_model(config, training=True)
    assert observed == {
        "model_id": "Qwen/Qwen3.5-9B",
        "revision": "a" * 40,
        "dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
        "device_map": {"": 3},
        "low_cpu_mem_usage": True,
        "use_kernels": False,
    }
    assert loaded.config.use_cache is False


def test_qwen35_fallback_attestation_checks_actual_routed_closures(monkeypatch):
    fallback_module = "transformers.models.qwen3_5.modeling_qwen3_5"

    def make_route(*, accelerated=False):
        def fallback(*args, **kwargs):
            return args, kwargs

        fallback.__module__ = fallback_module

        def fast(*args, **kwargs):
            return args, kwargs

        fast.__module__ = "fla.ops"
        implementation = fast if accelerated else fallback

        def routed(*args, **kwargs):
            return implementation(*args, **kwargs)

        routed.__wrapped__ = fallback
        return routed

    fake_modeling = types.ModuleType(fallback_module)
    routed_names = (
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "torch_chunk_gated_delta_rule",
        "torch_recurrent_gated_delta_rule",
    )
    for name in routed_names:
        setattr(fake_modeling, name, make_route())
    fake_qwen = types.ModuleType("transformers.models.qwen3_5")
    fake_qwen.modeling_qwen3_5 = fake_modeling
    fake_models = types.ModuleType("transformers.models")
    fake_models.qwen3_5 = fake_qwen
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.models = fake_models
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "transformers.models", fake_models)
    monkeypatch.setitem(sys.modules, "transformers.models.qwen3_5", fake_qwen)
    monkeypatch.setitem(sys.modules, fallback_module, fake_modeling)
    monkeypatch.setattr(modeling.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(modeling, "_package_version", lambda *names: None)
    config = {
        "model": {
            "id": "Qwen/Qwen3.5-9B",
            "loader_class": "Qwen3_5ForCausalLM",
            "delta_net_kernel_policy": "torch_fallback_required",
        }
    }
    attestation = deltanet_kernel_attestation(config)
    assert attestation["selected_backend"] == "torch_fallback"
    assert set(attestation["routed_callable_fallbacks"]) == set(routed_names)
    assert all(
        value["implementation_is_wrapped_fallback"] is True
        for value in attestation["routed_callable_fallbacks"].values()
    )

    fake_modeling.causal_conv1d_fn = make_route(accelerated=True)
    with pytest.raises(RuntimeError, match="not the pinned Transformers torch fallback"):
        deltanet_kernel_attestation(config)
