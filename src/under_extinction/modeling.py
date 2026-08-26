"""Lazy-loaded model helpers and exact forced-choice sequence scoring."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import math
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CHAT_TEMPLATE_KWARGS: dict[str, bool] = {"enable_thinking": False}
QWEN35_TEXT_LOADER = "Qwen3_5ForCausalLM"
QWEN35_MODEL_ID = "Qwen/Qwen3.5-9B"
QWEN35_LORA_TARGET_GROUPS: dict[str, tuple[str, ...]] = {
    "deltanet": ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"),
    "full_attention": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "mlp": ("gate_proj", "up_proj", "down_proj"),
}
QWEN35_LORA_TARGETS: tuple[str, ...] = tuple(
    target for group in QWEN35_LORA_TARGET_GROUPS.values() for target in group
)
MODEL_RUNTIME_ATTESTATION_KIND = "text_policy_lora_runtime"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _is_qwen35_config(config: Mapping[str, Any]) -> bool:
    model = config.get("model")
    if not isinstance(model, Mapping):
        return False
    return (
        str(model.get("id", "")) == QWEN35_MODEL_ID
        or str(model.get("loader_class", "")) == QWEN35_TEXT_LOADER
    )


def _accepts_chat_template_kwargs(tokenizer: Any) -> bool:
    try:
        signature = inspect.signature(tokenizer.apply_chat_template)
    except (TypeError, ValueError):
        return True
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ) or all(key in signature.parameters for key in CHAT_TEMPLATE_KWARGS)


def chat_prompt_text(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("The tokenizer has no chat template; this is a frozen design dependency")
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    # Qwen3.5 defaults to an open reasoning block.  Every scientific path uses this
    # helper, so explicitly disabling thinking here prevents acquisition/evaluation
    # from silently changing when tokenizer defaults change.  The compatibility
    # branch is only for simple legacy tokenizers/test doubles whose Python method
    # cannot accept template variables at all.
    if _accepts_chat_template_kwargs(tokenizer):
        kwargs.update(CHAT_TEMPLATE_KWARGS)
    rendered = tokenizer.apply_chat_template(messages, **kwargs)
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("The tokenizer chat template did not produce non-empty text")
    return rendered


def chat_template_runtime_attestation(
    tokenizer: Any, config: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Attest the exact non-thinking generation prefix used by all model paths."""
    expected = CHAT_TEMPLATE_KWARGS
    configured = None
    require_non_thinking = bool(config is not None and _is_qwen35_config(config))
    if config is not None:
        model = config.get("model")
        if isinstance(model, Mapping):
            configured = model.get("chat_template_kwargs")
    if configured is not None and dict(configured) != expected:
        raise ValueError(
            "model.chat_template_kwargs must be exactly {'enable_thinking': false}"
        )
    supports_kwargs = _accepts_chat_template_kwargs(tokenizer)
    if require_non_thinking and not supports_kwargs:
        raise TypeError("The pinned Qwen3.5 tokenizer cannot accept enable_thinking=False")
    probe = chat_prompt_text(
        tokenizer,
        [{"role": "user", "content": "UE_CHAT_TEMPLATE_RUNTIME_PROBE"}],
    )
    closed_reasoning_preamble = "</think>" in probe
    if require_non_thinking and not closed_reasoning_preamble:
        raise ValueError(
            "Qwen3.5 chat template did not render a closed non-thinking preamble"
        )
    template = str(tokenizer.chat_template)
    return {
        "method": "tokenizer.apply_chat_template",
        "tokenize": False,
        "add_generation_prompt": True,
        "template_kwargs": dict(expected),
        "kwargs_supported": supports_kwargs,
        "closed_reasoning_preamble_observed": closed_reasoning_preamble,
        "tokenizer_class": type(tokenizer).__name__,
        "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        "probe_prompt_sha256": hashlib.sha256(probe.encode("utf-8")).hexdigest(),
    }


def encode_prompt_and_choice(
    tokenizer: Any, messages: list[dict[str, str]], choice: str, max_length: int
) -> tuple[list[int], int]:
    prompt_text = chat_prompt_text(tokenizer, messages)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt_text + choice, add_special_tokens=False)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Tokenizer changed the prompt boundary when the choice was appended")
    candidate_length = len(full_ids) - len(prompt_ids)
    if candidate_length <= 0:
        raise ValueError(f"Choice {choice!r} produced no candidate tokens")
    if len(full_ids) > max_length:
        overflow = len(full_ids) - max_length
        if overflow >= len(prompt_ids):
            raise ValueError("Prompt truncation would remove the complete generation prefix")
        full_ids = full_ids[overflow:]
        prompt_length = len(prompt_ids) - overflow
    else:
        prompt_length = len(prompt_ids)
    return full_ids, prompt_length


def verify_choice_tokens(tokenizer: Any, messages: list[dict[str, str]], labels: list[str]) -> dict[str, Any]:
    prompt = chat_prompt_text(tokenizer, messages)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    details: dict[str, Any] = {}
    lengths: list[int] = []
    for label in labels:
        full_ids = tokenizer(prompt + label, add_special_tokens=False)["input_ids"]
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(f"Choice {label!r} does not preserve the prompt token prefix")
        ids = full_ids[len(prompt_ids):]
        details[label] = {"token_ids": ids, "token_count": len(ids)}
        lengths.append(len(ids))
    details["equal_token_counts"] = len(set(lengths)) == 1
    details["all_single_token"] = all(length == 1 for length in lengths)
    return details


class ChoiceSFTDataset:
    """Torch-compatible dataset with loss only on the A/B completion tokens."""

    def __init__(self, records: list[dict[str, Any]], tokenizer: Any, controller: str, max_length: int):
        self.records = records
        self.tokenizer = tokenizer
        self.controller = controller
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        record = self.records[index]
        choice = record["oracle_actions"][self.controller]
        input_ids, prompt_length = encode_prompt_and_choice(
            self.tokenizer, record["messages"], choice, self.max_length
        )
        labels = [-100] * prompt_length + input_ids[prompt_length:]
        if all(label == -100 for label in labels):
            raise ValueError(f"No supervised choice tokens for {record['record_id']}")
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


@dataclass
class ChoiceCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids: list[list[int]] = []
        attention_masks: list[list[int]] = []
        labels: list[list[int]] = []
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            attention_masks.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def load_tokenizer(config: dict[str, Any]) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["id"], revision=config["model"]["revision"], use_fast=True
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _dtype(name: str) -> Any:
    import torch

    names = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if name not in names:
        raise ValueError(f"Unsupported model dtype: {name}")
    return names[name]


def configured_model_loader(config: Mapping[str, Any]) -> str:
    model = config.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("Model configuration must be a mapping")
    configured = model.get("loader_class")
    if _is_qwen35_config(config):
        if configured is not None and configured != QWEN35_TEXT_LOADER:
            raise ValueError(
                f"Qwen3.5-9B must use the text-only {QWEN35_TEXT_LOADER} loader"
            )
        if model.get("text_only", True) is not True:
            raise ValueError("Qwen3.5-9B must freeze model.text_only=true")
        chat_kwargs = model.get("chat_template_kwargs", CHAT_TEMPLATE_KWARGS)
        if dict(chat_kwargs) != CHAT_TEMPLATE_KWARGS:
            raise ValueError(
                "Qwen3.5-9B requires chat_template_kwargs.enable_thinking=false"
            )
        kernel_policy = model.get(
            "delta_net_kernel_policy", "torch_fallback_required"
        )
        if kernel_policy != "torch_fallback_required":
            raise ValueError(
                "Qwen3.5-9B requires delta_net_kernel_policy="
                "'torch_fallback_required' for the frozen initial experiment"
            )
        return QWEN35_TEXT_LOADER
    return str(configured or "AutoModelForCausalLM")


def _package_version(*distribution_names: str) -> str | None:
    for name in distribution_names:
        try:
            return importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return None


def deltanet_kernel_attestation(config: Mapping[str, Any]) -> dict[str, Any]:
    """Report accelerated-kernel readiness without requiring fragile source builds."""
    policy = str(
        dict(config.get("model") or {}).get(
            "delta_net_kernel_policy", "torch_fallback_required"
        )
    )
    causal_importable = importlib.util.find_spec("causal_conv1d") is not None
    fla_importable = importlib.util.find_spec("fla") is not None
    kernels_importable = importlib.util.find_spec("kernels") is not None
    routed_fallbacks: dict[str, dict[str, Any]] = {}
    if _is_qwen35_config(config):
        try:
            from transformers.models.qwen3_5 import modeling_qwen3_5
        except ImportError as exc:
            raise RuntimeError("Cannot inspect the pinned Qwen3.5 DeltaNet runtime") from exc
        for name in (
            "causal_conv1d_fn",
            "causal_conv1d_update",
            "torch_chunk_gated_delta_rule",
            "torch_recurrent_gated_delta_rule",
        ):
            function = getattr(modeling_qwen3_5, name, None)
            wrapped = getattr(function, "__wrapped__", None)
            try:
                implementation = inspect.getclosurevars(function).nonlocals[
                    "implementation"
                ]
            except (AttributeError, KeyError, TypeError) as exc:
                raise RuntimeError(
                    f"Cannot attest the selected Qwen3.5 DeltaNet callable {name}"
                ) from exc
            implementation_module = str(getattr(implementation, "__module__", ""))
            identity_matches = implementation is wrapped
            routed_fallbacks[name] = {
                "implementation_is_wrapped_fallback": identity_matches,
                "implementation_module": implementation_module,
            }
            if policy == "torch_fallback_required" and (
                not identity_matches
                or implementation_module
                != "transformers.models.qwen3_5.modeling_qwen3_5"
            ):
                raise RuntimeError(
                    f"Qwen3.5 callable {name} is not the pinned Transformers torch fallback"
                )
    fast_ready = causal_importable and fla_importable
    if policy == "torch_fallback_required":
        # The frozen deployment intentionally installs no optional/hub kernel
        # packages. Reject ambient packages before they can change the computation.
        if causal_importable or fla_importable or kernels_importable:
            raise RuntimeError(
                "torch_fallback_required forbids ambient causal_conv1d, fla, and "
                "kernels packages"
            )
        backend = "torch_fallback"
    else:
        raise ValueError(f"Unsupported Qwen3.5 DeltaNet kernel policy: {policy!r}")
    return {
        "policy": policy,
        "selected_backend": backend,
        "causal_conv1d_importable": causal_importable,
        "causal_conv1d_version": _package_version("causal-conv1d", "causal_conv1d"),
        "fla_importable": fla_importable,
        "fla_version": _package_version("flash-linear-attention", "fla"),
        "kernels_importable": kernels_importable,
        "kernels_version": _package_version("kernels"),
        "routed_callable_fallbacks": routed_fallbacks,
        "fast_path_ready": fast_ready,
        "acceptance_basis": (
            "fast_kernel_availability"
            if policy == "fast_required"
            else "exact_model_measured_throughput_and_cost_gate"
        ),
    }


def _model_parameter_counts(model: Any) -> tuple[int, int]:
    parameters = list(model.parameters())
    return len(parameters), sum(int(parameter.numel()) for parameter in parameters)


def base_model_runtime_attestation(
    config: Mapping[str, Any],
    model: Any,
    *,
    parameter_counts_override: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Validate and describe the actually loaded text policy, excluding adapters."""
    requested_loader = configured_model_loader(config)
    loaded_class = type(model).__name__
    model_config = getattr(model, "config", None)
    if model_config is None:
        raise TypeError("Loaded model has no config")
    module_names = [name for name, _ in model.named_modules()]
    parameter_names = [name for name, _ in model.named_parameters()]
    visual_names = [
        name
        for name in {*module_names, *parameter_names}
        if any(part in {"visual", "vision_tower"} for part in name.split("."))
    ]
    layer_types = list(getattr(model_config, "layer_types", []) or [])
    if _is_qwen35_config(config):
        if loaded_class != QWEN35_TEXT_LOADER:
            raise TypeError(
                f"Expected {QWEN35_TEXT_LOADER}, loaded {loaded_class}; refusing a vision model"
            )
        if getattr(model_config, "model_type", None) != "qwen3_5_text":
            raise ValueError("Qwen3.5 text loader did not produce Qwen3_5TextConfig")
        if visual_names:
            raise ValueError("Text-only Qwen3.5 unexpectedly contains visual modules")
        if len(layer_types) != 32 or layer_types.count("linear_attention") != 24 or layer_types.count(
            "full_attention"
        ) != 8:
            raise ValueError(
                "Pinned Qwen3.5-9B must expose the expected 24:8 DeltaNet/attention stack"
            )
    tensor_count, parameter_count = (
        parameter_counts_override
        if parameter_counts_override is not None
        else _model_parameter_counts(model)
    )
    model_contract = dict(config.get("model") or {})
    expected_parameter_count = model_contract.get("expected_text_parameter_count")
    if expected_parameter_count is not None and parameter_count != int(
        expected_parameter_count
    ):
        raise ValueError(
            f"Loaded text parameter count {parameter_count} differs from config "
            f"{expected_parameter_count}"
        )
    expected_model_type = model_contract.get("expected_model_type")
    if expected_model_type is not None and getattr(
        model_config, "model_type", None
    ) != str(expected_model_type):
        raise ValueError("Loaded model type differs from model.expected_model_type")
    expected_layer_counts = model_contract.get("expected_layer_type_counts")
    observed_layer_counts = {
        "linear_attention": layer_types.count("linear_attention"),
        "full_attention": layer_types.count("full_attention"),
    }
    if expected_layer_counts is not None and {
        str(key): int(value) for key, value in dict(expected_layer_counts).items()
    } != observed_layer_counts:
        raise ValueError(
            f"Loaded layer-type counts {observed_layer_counts} differ from config "
            f"{expected_layer_counts}"
        )
    configured_attention = dict(config.get("model") or {}).get("attention", "sdpa")
    loaded_attention = getattr(model_config, "_attn_implementation", None)
    if (
        _is_qwen35_config(config)
        and loaded_attention is not None
        and loaded_attention != configured_attention
    ):
        raise ValueError(
            f"Loaded attention implementation {loaded_attention!r} differs from config "
            f"{configured_attention!r}"
        )
    use_kernels = getattr(model, "use_kernels", None)
    if _is_qwen35_config(config) and use_kernels is not False:
        raise ValueError("Qwen3.5 text model was not loaded with use_kernels=False")
    return {
        "requested_loader_class": requested_loader,
        "loaded_model_class": loaded_class,
        "loaded_config_class": type(model_config).__name__,
        "loaded_model_type": str(getattr(model_config, "model_type", "")),
        "text_only": not visual_names,
        "visual_module_or_parameter_count": len(visual_names),
        "configured_attention_implementation": str(configured_attention),
        "loaded_attention_implementation": loaded_attention,
        "use_kernels": use_kernels,
        "hidden_layer_count": len(layer_types) if layer_types else getattr(
            model_config, "num_hidden_layers", None
        ),
        "layer_type_counts": observed_layer_counts,
        "parameter_tensor_count_before_lora": tensor_count,
        "parameter_count_before_lora": parameter_count,
    }


def inspect_lora_target_inventory(
    config: Mapping[str, Any], model: Any
) -> dict[str, Any]:
    """Fail closed if any configured LoRA target is absent or misclassified."""
    import torch

    raw_targets = dict(config.get("training") or {}).get("lora_targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError("training.lora_targets must be a non-empty list")
    targets = tuple(str(target) for target in raw_targets)
    if len(set(targets)) != len(targets):
        raise ValueError("training.lora_targets contains duplicates")
    if _is_qwen35_config(config) and targets != QWEN35_LORA_TARGETS:
        raise ValueError(
            "Qwen3.5-9B LoRA targets must exactly cover DeltaNet, full attention, and MLP: "
            f"{list(QWEN35_LORA_TARGETS)}"
        )
    matches: dict[str, list[str]] = {target: [] for target in targets}
    dimensions: dict[str, dict[str, int]] = {}
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in matches:
            continue
        if not isinstance(module, torch.nn.Linear):
            raise TypeError(f"Configured LoRA target is not torch.nn.Linear: {name}")
        matches[leaf].append(name)
        dimensions[name] = {
            "in_features": int(module.in_features),
            "out_features": int(module.out_features),
        }
    missing = sorted(target for target, names in matches.items() if not names)
    if missing:
        raise ValueError(f"Configured LoRA targets matched no linear modules: {missing}")
    category_names: dict[str, list[str]] = {}
    if _is_qwen35_config(config):
        layer_types = list(model.config.layer_types)
        expected_by_group: dict[str, set[str]] = {
            "deltanet": {
                f"layers.{index}.linear_attn.{leaf}"
                for index, layer_type in enumerate(layer_types)
                if layer_type == "linear_attention"
                for leaf in QWEN35_LORA_TARGET_GROUPS["deltanet"]
            },
            "full_attention": {
                f"layers.{index}.self_attn.{leaf}"
                for index, layer_type in enumerate(layer_types)
                if layer_type == "full_attention"
                for leaf in QWEN35_LORA_TARGET_GROUPS["full_attention"]
            },
            "mlp": {
                f"layers.{index}.mlp.{leaf}"
                for index in range(len(layer_types))
                for leaf in QWEN35_LORA_TARGET_GROUPS["mlp"]
            },
        }
        actual_names = {name for values in matches.values() for name in values}
        expected_to_actual: dict[str, str] = {}
        for expected in set().union(*expected_by_group.values()):
            candidates = [name for name in actual_names if name.endswith(expected)]
            if len(candidates) != 1:
                raise ValueError(
                    f"Expected exactly one Qwen3.5 LoRA module ending in {expected!r}; "
                    f"found {candidates}"
                )
            expected_to_actual[expected] = candidates[0]
        expected_names = set(expected_to_actual.values())
        unexpected = sorted(actual_names - expected_names)
        if unexpected:
            raise ValueError(f"Unexpected Qwen3.5 text LoRA target matches: {unexpected}")
        category_names = {
            group: sorted(expected_to_actual[name] for name in expected)
            for group, expected in expected_by_group.items()
        }
    else:
        category_names = {"configured_targets": sorted(dimensions)}
    matched_names = sorted(dimensions)
    base_tensor_count, base_parameter_count = _model_parameter_counts(model)
    result = {
        "requested_target_modules": list(targets),
        "matched_module_count": len(matched_names),
        "matched_module_names": matched_names,
        "matched_module_names_sha256": _sha256_json(matched_names),
        "matched_module_dimensions": {
            name: dimensions[name] for name in matched_names
        },
        "per_target_module_count": {
            target: len(matches[target]) for target in targets
        },
        "groups": {
            group: {
                "matched_module_count": len(names),
                "matched_module_names": names,
            }
            for group, names in category_names.items()
        },
        "all_configured_targets_matched": True,
        "base_parameter_tensor_count": base_tensor_count,
        "base_parameter_count": base_parameter_count,
    }
    if _is_qwen35_config(config):
        observed = {
            group: result["groups"][group]["matched_module_count"]
            for group in QWEN35_LORA_TARGET_GROUPS
        }
        if observed != {"deltanet": 120, "full_attention": 32, "mlp": 96}:
            raise ValueError(f"Unexpected Qwen3.5-9B LoRA group counts: {observed}")
        if result["matched_module_count"] != 248:
            raise ValueError("Qwen3.5-9B must have exactly 248 LoRA target modules")
    training_contract = dict(config.get("training") or {})
    expected_target_counts = training_contract.get("expected_lora_target_counts")
    if expected_target_counts is not None and {
        str(key): int(value) for key, value in dict(expected_target_counts).items()
    } != result["per_target_module_count"]:
        raise ValueError(
            "Validated LoRA per-target counts differ from "
            "training.expected_lora_target_counts"
        )
    expected_module_count = training_contract.get("expected_lora_module_count")
    if expected_module_count is not None and result["matched_module_count"] != int(
        expected_module_count
    ):
        raise ValueError(
            "Validated LoRA module count differs from training.expected_lora_module_count"
        )
    result["inventory_sha256"] = _sha256_json(result)
    return result


def build_model_runtime_attestation(
    config: Mapping[str, Any],
    model: Any,
    tokenizer: Any,
    target_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    """Attest exact PEFT wrapping and trainable counts after LoRA attachment/load."""
    inventory = json.loads(_canonical_json(dict(target_inventory)))
    expected_names = list(inventory.get("matched_module_names") or [])
    if not expected_names:
        raise ValueError("LoRA target inventory is empty")
    wrappers: dict[str, Any] = {}
    wrapper_runtime_names: dict[str, str] = {}
    for runtime_name, module in model.named_modules():
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue
        candidates = [name for name in expected_names if runtime_name.endswith(name)]
        if len(candidates) != 1:
            continue
        base_name = candidates[0]
        if base_name in wrappers:
            raise ValueError(f"Multiple PEFT wrappers matched LoRA target {base_name}")
        wrappers[base_name] = module
        wrapper_runtime_names[base_name] = runtime_name
    missing_wrappers = sorted(set(expected_names) - set(wrappers))
    if missing_wrappers:
        raise ValueError(f"LoRA targets were not wrapped by PEFT: {missing_wrappers}")
    if len(wrappers) != int(inventory["matched_module_count"]):
        raise ValueError("PEFT wrapper count differs from the validated target inventory")

    rank = int(dict(config.get("training") or {})["lora_rank"])
    adapter_names: set[str] = set()
    expected_trainable_numel = 0
    for base_name, wrapper in wrappers.items():
        keys_a = set(wrapper.lora_A.keys())
        keys_b = set(wrapper.lora_B.keys())
        if not keys_a or keys_a != keys_b:
            raise ValueError(f"Incomplete LoRA A/B adapters on {base_name}")
        adapter_names.update(map(str, keys_a))
        for adapter_name in keys_a:
            weight_a = wrapper.lora_A[adapter_name].weight
            weight_b = wrapper.lora_B[adapter_name].weight
            if int(weight_a.shape[0]) != rank or int(weight_b.shape[1]) != rank:
                raise ValueError(f"LoRA rank mismatch on {base_name}/{adapter_name}")
            expected_trainable_numel += int(weight_a.numel() + weight_b.numel())
    if adapter_names != {"default"}:
        raise ValueError(f"Expected exactly the default LoRA adapter; found {adapter_names}")

    trainable = {
        name: int(parameter.numel())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    non_lora_trainable = sorted(
        name
        for name in trainable
        if ".lora_A." not in name and ".lora_B." not in name
    )
    if non_lora_trainable:
        raise ValueError(f"Non-LoRA parameters are trainable: {non_lora_trainable}")
    expected_tensor_count = 2 * len(wrappers)
    if len(trainable) != expected_tensor_count:
        raise ValueError(
            f"Expected {expected_tensor_count} trainable LoRA tensors, found {len(trainable)}"
        )
    trainable_numel = sum(trainable.values())
    if trainable_numel != expected_trainable_numel:
        raise ValueError(
            f"Expected {expected_trainable_numel} trainable LoRA parameters, found "
            f"{trainable_numel}"
        )
    configured_trainable_count = dict(config.get("training") or {}).get(
        "expected_lora_trainable_parameter_count"
    )
    if configured_trainable_count is not None and trainable_numel != int(
        configured_trainable_count
    ):
        raise ValueError(
            "Trainable LoRA parameter count differs from "
            "training.expected_lora_trainable_parameter_count"
        )
    total_tensor_count, total_parameter_count = _model_parameter_counts(model)
    # PEFT mutates the base module in-place, so use the counts measured by the
    # pre-injection inventory rather than counting attached adapter parameters.
    base_runtime = base_model_runtime_attestation(
        config,
        model.get_base_model(),
        parameter_counts_override=(
            int(inventory["base_parameter_tensor_count"]),
            int(inventory["base_parameter_count"]),
        ),
    )
    attestation: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": MODEL_RUNTIME_ATTESTATION_KIND,
        "model_id": str(dict(config.get("model") or {})["id"]),
        "model_revision": str(dict(config.get("model") or {})["revision"]),
        "libraries": {
            name: _package_version(name)
            for name in ("torch", "transformers", "peft", "accelerate", "safetensors")
        },
        "base_model": base_runtime,
        "chat_template": chat_template_runtime_attestation(tokenizer, config),
        "deltanet_kernels": deltanet_kernel_attestation(config),
        "lora": {
            **inventory,
            "rank": rank,
            "alpha": int(dict(config.get("training") or {})["lora_alpha"]),
            "dropout": float(dict(config.get("training") or {})["lora_dropout"]),
            "adapter_names": sorted(adapter_names),
            "wrapped_module_count": len(wrappers),
            "wrapper_runtime_names": {
                name: wrapper_runtime_names[name] for name in sorted(wrapper_runtime_names)
            },
            "trainable_parameter_tensor_count": len(trainable),
            "trainable_parameter_count": trainable_numel,
            "trainable_parameter_names": sorted(trainable),
            "trainable_parameter_names_sha256": _sha256_json(sorted(trainable)),
            "total_parameter_tensor_count_after_lora": total_tensor_count,
            "total_parameter_count_after_lora": total_parameter_count,
            "only_lora_parameters_trainable": True,
            "coverage_complete": True,
        },
    }
    attestation["attestation_sha256"] = _sha256_json(attestation)
    verify_model_runtime_attestation(config, attestation)
    return attestation


def verify_model_runtime_attestation(
    config: Mapping[str, Any], attestation: Mapping[str, Any]
) -> dict[str, Any]:
    required = {
        "schema_version",
        "kind",
        "model_id",
        "model_revision",
        "libraries",
        "base_model",
        "chat_template",
        "deltanet_kernels",
        "lora",
        "attestation_sha256",
    }
    if set(attestation) != required:
        raise ValueError("Malformed model runtime attestation")
    unsigned = dict(attestation)
    claimed = unsigned.pop("attestation_sha256")
    if claimed != _sha256_json(unsigned):
        raise ValueError("Model runtime attestation self-hash mismatch")
    model_config = dict(config.get("model") or {})
    if (
        attestation.get("kind") != MODEL_RUNTIME_ATTESTATION_KIND
        or attestation.get("model_id") != str(model_config.get("id"))
        or attestation.get("model_revision") != str(model_config.get("revision"))
    ):
        raise ValueError("Model runtime attestation is bound to a different model")
    chat = attestation.get("chat_template")
    if not isinstance(chat, Mapping) or chat.get("template_kwargs") != CHAT_TEMPLATE_KWARGS:
        raise ValueError("Model runtime attestation does not freeze non-thinking prompts")
    lora = attestation.get("lora")
    if not isinstance(lora, Mapping) or lora.get("coverage_complete") is not True:
        raise ValueError("Model runtime attestation lacks complete LoRA coverage")
    if lora.get("requested_target_modules") != list(
        dict(config.get("training") or {}).get("lora_targets") or []
    ):
        raise ValueError("Model runtime attestation has different LoRA targets")
    if _is_qwen35_config(config):
        base = attestation.get("base_model")
        kernels = attestation.get("deltanet_kernels")
        if (
            not isinstance(base, Mapping)
            or base.get("requested_loader_class") != QWEN35_TEXT_LOADER
            or base.get("loaded_model_class") != QWEN35_TEXT_LOADER
            or base.get("loaded_model_type") != "qwen3_5_text"
            or base.get("text_only") is not True
            or base.get("use_kernels") is not False
        ):
            raise ValueError("Attestation does not prove the Qwen3.5 text-only loader")
        expected_parameter_count = model_config.get("expected_text_parameter_count")
        if expected_parameter_count is not None and int(
            base.get("parameter_count_before_lora", -1)
        ) != int(expected_parameter_count):
            raise ValueError("Attested text parameter count differs from config")
        expected_layer_counts = model_config.get("expected_layer_type_counts")
        if expected_layer_counts is not None and base.get("layer_type_counts") != {
            str(key): int(value) for key, value in dict(expected_layer_counts).items()
        }:
            raise ValueError("Attested layer-type counts differ from config")
        if (
            int(lora.get("matched_module_count", -1)) != 248
            or int(lora.get("wrapped_module_count", -1)) != 248
            or lora.get("only_lora_parameters_trainable") is not True
        ):
            raise ValueError("Attestation does not prove exact Qwen3.5 LoRA coverage")
        expected_target_counts = dict(config.get("training") or {}).get(
            "expected_lora_target_counts"
        )
        if expected_target_counts is not None and lora.get(
            "per_target_module_count"
        ) != {
            str(key): int(value) for key, value in dict(expected_target_counts).items()
        }:
            raise ValueError("Attested LoRA per-target counts differ from config")
        expected_module_count = dict(config.get("training") or {}).get(
            "expected_lora_module_count"
        )
        if expected_module_count is not None and int(
            lora.get("matched_module_count", -1)
        ) != int(expected_module_count):
            raise ValueError("Attested LoRA module count differs from config")
        expected_trainable_count = dict(config.get("training") or {}).get(
            "expected_lora_trainable_parameter_count"
        )
        if expected_trainable_count is not None and int(
            lora.get("trainable_parameter_count", -1)
        ) != int(expected_trainable_count):
            raise ValueError("Attested trainable LoRA count differs from config")
        if (
            chat.get("kwargs_supported") is not True
            or chat.get("closed_reasoning_preamble_observed") is not True
        ):
            raise ValueError("Attestation does not prove non-thinking Qwen3.5 prompts")
        if (
            not isinstance(kernels, Mapping)
            or kernels.get("policy") != model_config.get(
                "delta_net_kernel_policy", "torch_fallback_required"
            )
        ):
            raise ValueError("Attested DeltaNet kernel policy differs from config")
        if (
            kernels.get("selected_backend") != "torch_fallback"
            or kernels.get("causal_conv1d_importable") is not False
            or kernels.get("fla_importable") is not False
            or kernels.get("kernels_importable") is not False
        ):
            raise ValueError("Attestation does not prove the required torch-only fallback")
        routed = kernels.get("routed_callable_fallbacks")
        expected_routed_names = {
            "causal_conv1d_fn",
            "causal_conv1d_update",
            "torch_chunk_gated_delta_rule",
            "torch_recurrent_gated_delta_rule",
        }
        if not isinstance(routed, Mapping) or set(routed) != expected_routed_names:
            raise ValueError("Attestation lacks the exact Qwen3.5 routed callable inventory")
        for name in sorted(expected_routed_names):
            route = routed[name]
            if (
                not isinstance(route, Mapping)
                or route.get("implementation_is_wrapped_fallback") is not True
                or route.get("implementation_module")
                != "transformers.models.qwen3_5.modeling_qwen3_5"
            ):
                raise ValueError(
                    f"Attestation does not prove the Transformers fallback for {name}"
                )
    return json.loads(_canonical_json(dict(attestation)))


def compact_model_runtime_contract(
    config: Mapping[str, Any], attestation: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the small exact contract copied into prediction rows."""
    verified = verify_model_runtime_attestation(config, attestation)
    base = verified["base_model"]
    chat = verified["chat_template"]
    kernels = verified["deltanet_kernels"]
    lora = verified["lora"]
    compact: dict[str, Any] = {
        "schema_version": "1.0",
        "full_attestation_sha256": verified["attestation_sha256"],
        "model_id": verified["model_id"],
        "model_revision": verified["model_revision"],
        "loader_class": base["loaded_model_class"],
        "model_type": base["loaded_model_type"],
        "text_only": base["text_only"],
        "text_parameter_count": base["parameter_count_before_lora"],
        "layer_type_counts": dict(base["layer_type_counts"]),
        "chat_template_kwargs": dict(chat["template_kwargs"]),
        "chat_template_kwargs_supported": chat["kwargs_supported"],
        "closed_reasoning_preamble_observed": chat[
            "closed_reasoning_preamble_observed"
        ],
        "chat_template_sha256": chat["chat_template_sha256"],
        "deltanet_kernel_policy": kernels["policy"],
        "deltanet_backend": kernels["selected_backend"],
        "lora_per_target_module_count": dict(lora["per_target_module_count"]),
        "lora_module_count": lora["matched_module_count"],
        "lora_trainable_parameter_count": lora["trainable_parameter_count"],
        "lora_inventory_sha256": lora["inventory_sha256"],
    }
    compact["contract_sha256"] = _sha256_json(compact)
    return compact


def validate_loaded_lora_runtime(
    config: Mapping[str, Any],
    model: Any,
    tokenizer: Any,
    target_inventory: Mapping[str, Any],
    expected_attestation: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an inference-only PEFT load against its training-time attestation."""
    verified = verify_model_runtime_attestation(config, expected_attestation)
    inventory = json.loads(_canonical_json(dict(target_inventory)))
    attested_lora = verified["lora"]
    for key in (
        "requested_target_modules",
        "matched_module_count",
        "matched_module_names_sha256",
        "per_target_module_count",
        "groups",
        "inventory_sha256",
        "base_parameter_tensor_count",
        "base_parameter_count",
    ):
        if inventory.get(key) != attested_lora.get(key):
            raise ValueError(f"Loaded adapter target inventory differs for {key}")

    expected_names = list(inventory["matched_module_names"])
    wrappers: dict[str, Any] = {}
    for runtime_name, module in model.named_modules():
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue
        candidates = [name for name in expected_names if runtime_name.endswith(name)]
        if len(candidates) == 1:
            if candidates[0] in wrappers:
                raise ValueError(f"Duplicate loaded LoRA wrapper for {candidates[0]}")
            wrappers[candidates[0]] = module
    if set(wrappers) != set(expected_names):
        missing = sorted(set(expected_names) - set(wrappers))
        raise ValueError(f"Loaded adapter lacks PEFT wrappers: {missing}")
    rank = int(dict(config.get("training") or {})["lora_rank"])
    adapter_names: set[str] = set()
    for name, wrapper in wrappers.items():
        keys_a = set(wrapper.lora_A.keys())
        keys_b = set(wrapper.lora_B.keys())
        if not keys_a or keys_a != keys_b:
            raise ValueError(f"Loaded adapter has incomplete LoRA A/B weights on {name}")
        adapter_names.update(map(str, keys_a))
        for adapter_name in keys_a:
            if (
                int(wrapper.lora_A[adapter_name].weight.shape[0]) != rank
                or int(wrapper.lora_B[adapter_name].weight.shape[1]) != rank
            ):
                raise ValueError(f"Loaded adapter has the wrong rank on {name}")
    if adapter_names != set(attested_lora["adapter_names"]):
        raise ValueError("Loaded adapter names differ from the checkpoint attestation")

    current_base = base_model_runtime_attestation(
        config,
        model.get_base_model(),
        parameter_counts_override=(
            int(inventory["base_parameter_tensor_count"]),
            int(inventory["base_parameter_count"]),
        ),
    )
    if current_base != verified["base_model"]:
        raise ValueError("Loaded base model runtime differs from the checkpoint attestation")
    if chat_template_runtime_attestation(tokenizer, config) != verified["chat_template"]:
        raise ValueError("Loaded tokenizer/chat template differs from checkpoint attestation")
    if deltanet_kernel_attestation(config) != verified["deltanet_kernels"]:
        raise ValueError("Loaded DeltaNet backend differs from checkpoint attestation")
    return compact_model_runtime_contract(config, verified)


def validate_loaded_base_runtime(
    config: Mapping[str, Any],
    model: Any,
    tokenizer: Any,
    expected_attestation: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the adapter-free control against a checkpoint's runtime contract."""
    verified = verify_model_runtime_attestation(config, expected_attestation)
    if base_model_runtime_attestation(config, model) != verified["base_model"]:
        raise ValueError("Loaded unchanged base differs from checkpoint runtime attestation")
    if chat_template_runtime_attestation(tokenizer, config) != verified["chat_template"]:
        raise ValueError("Loaded unchanged-base tokenizer differs from checkpoint attestation")
    if deltanet_kernel_attestation(config) != verified["deltanet_kernels"]:
        raise ValueError("Loaded unchanged-base DeltaNet backend differs from checkpoint attestation")
    return compact_model_runtime_contract(config, verified)


def load_base_model(config: dict[str, Any], *, training: bool) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for real model training/evaluation")
    loader_name = configured_model_loader(config)
    if loader_name == QWEN35_TEXT_LOADER:
        try:
            from transformers import Qwen3_5ForCausalLM
        except ImportError as exc:
            raise RuntimeError(
                "The pinned Transformers runtime lacks native Qwen3.5 support"
            ) from exc
        loader = Qwen3_5ForCausalLM
    elif loader_name == "AutoModelForCausalLM":
        loader = AutoModelForCausalLM
    else:
        raise ValueError(f"Unsupported model.loader_class: {loader_name!r}")
    load_kwargs: dict[str, Any] = {
        "revision": config["model"]["revision"],
        "dtype": _dtype(config["model"]["dtype"]),
        "attn_implementation": config["model"].get("attention", "sdpa"),
        "device_map": {"": torch.cuda.current_device()},
        "low_cpu_mem_usage": True,
    }
    if loader_name == QWEN35_TEXT_LOADER:
        # Transformers 5.15 exposes this explicit switch on Qwen3.5.  Keep all
        # paid runs on its auditable PyTorch DeltaNet implementation; the exact
        # model preflight measures whether that implementation is economical.
        load_kwargs["use_kernels"] = False
    model = loader.from_pretrained(config["model"]["id"], **load_kwargs)
    model.config.use_cache = not training
    base_model_runtime_attestation(config, model)
    deltanet_kernel_attestation(config)
    return model


def load_adapter_model(config: dict[str, Any], adapter_path: str | Path) -> Any:
    import time

    from peft import PeftModel

    base_started = time.monotonic()
    base = load_base_model(config, training=False)
    target_inventory = inspect_lora_target_inventory(config, base)
    base_wall_seconds = time.monotonic() - base_started
    adapter_started = time.monotonic()
    model = PeftModel.from_pretrained(base, str(Path(adapter_path).resolve()), is_trainable=False)
    # Evaluation intentionally freezes adapters.  Preserve the pre-injection target
    # inventory so the caller can temporarily mark only LoRA tensors trainable and
    # reconstruct the same training-time coverage attestation without reloading.
    model._ue_lora_target_inventory = target_inventory
    model._ue_base_model_load_wall_seconds = base_wall_seconds
    model._ue_adapter_load_wall_seconds = time.monotonic() - adapter_started
    model.eval()
    return model


def score_choice_batch(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    labels: list[str],
    max_length: int,
) -> list[dict[str, float]]:
    """Score every token in each legal completion and normalize over legal choices."""
    import torch

    encoded: list[tuple[list[int], int, int, str]] = []
    for record_index, record in enumerate(records):
        for choice in labels:
            ids, prompt_length = encode_prompt_and_choice(tokenizer, record["messages"], choice, max_length)
            encoded.append((ids, prompt_length, record_index, choice))
    pad_id = int(tokenizer.pad_token_id)
    longest = max(len(item[0]) for item in encoded)
    input_ids = torch.full((len(encoded), longest), pad_id, dtype=torch.long, device=model.device)
    attention_mask = torch.zeros((len(encoded), longest), dtype=torch.long, device=model.device)
    for row_index, (ids, _, _, _) in enumerate(encoded):
        input_ids[row_index, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=model.device)
        attention_mask[row_index, : len(ids)] = 1
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
        log_probs = torch.log_softmax(logits, dim=-1)
    raw: list[dict[str, float]] = [dict() for _ in records]
    for row_index, (ids, prompt_length, record_index, choice) in enumerate(encoded):
        score = 0.0
        for token_position in range(prompt_length, len(ids)):
            score += float(log_probs[row_index, token_position - 1, ids[token_position]].item())
        raw[record_index][choice] = score
    results: list[dict[str, float]] = []
    for scores in raw:
        if set(scores) != set(labels):
            raise AssertionError(f"Missing legal choice score: {scores}")
        maximum = max(scores.values())
        weights = {label: math.exp(scores[label] - maximum) for label in labels}
        normalizer = sum(weights.values())
        log_legal_mass = maximum + math.log(normalizer)
        results.append({
            "logp_A": scores["A"],
            "logp_B": scores["B"],
            "probability_A": weights["A"] / normalizer,
            "log_legal_choice_mass": log_legal_mass,
            "legal_choice_mass": math.exp(log_legal_mass) if log_legal_mass > -745.0 else 0.0,
        })
    return results


def score_choice_batch_generic(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    labels: list[str],
    max_length: int,
) -> list[dict[str, Any]]:
    """Score arbitrary legal completions without assuming an A/B interface.

    This is deliberately separate from :func:`score_choice_batch`: existing
    registered experiments retain their fixed A/B output schema, while a new
    measurement-validity experiment can compare genuinely different response
    serializations without pretending that they share those labels.
    """
    import torch

    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("Generic choice scoring needs at least two unique labels")
    encoded: list[tuple[list[int], int, int, str]] = []
    for record_index, record in enumerate(records):
        for choice in labels:
            ids, prompt_length = encode_prompt_and_choice(
                tokenizer, record["messages"], choice, max_length
            )
            encoded.append((ids, prompt_length, record_index, choice))
    pad_id = int(tokenizer.pad_token_id)
    longest = max(len(item[0]) for item in encoded)
    input_ids = torch.full((len(encoded), longest), pad_id, dtype=torch.long, device=model.device)
    attention_mask = torch.zeros((len(encoded), longest), dtype=torch.long, device=model.device)
    for row_index, (ids, _, _, _) in enumerate(encoded):
        input_ids[row_index, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=model.device)
        attention_mask[row_index, : len(ids)] = 1
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
        log_probs = torch.log_softmax(logits, dim=-1)
    raw: list[dict[str, float]] = [dict() for _ in records]
    for row_index, (ids, prompt_length, record_index, choice) in enumerate(encoded):
        raw[record_index][choice] = sum(
            float(log_probs[row_index, position - 1, ids[position]].item())
            for position in range(prompt_length, len(ids))
        )
    results: list[dict[str, Any]] = []
    for scores in raw:
        if set(scores) != set(labels):
            raise AssertionError(f"Missing legal choice score: {scores}")
        maximum = max(scores.values())
        weights = {label: math.exp(scores[label] - maximum) for label in labels}
        normalizer = sum(weights.values())
        log_legal_mass = maximum + math.log(normalizer)
        results.append({
            "choice_logps": dict(scores),
            "choice_probabilities": {label: weights[label] / normalizer for label in labels},
            "log_legal_choice_mass": log_legal_mass,
            "legal_choice_mass": math.exp(log_legal_mass) if log_legal_mass > -745.0 else 0.0,
        })
    return results
