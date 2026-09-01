"""Differentiable forced-choice policy and immutable LoRA checkpoint helpers."""

from __future__ import annotations

import gc
import hashlib
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

from under_extinction.modeling import (
    QWEN35_LORA_TARGETS,
    base_model_runtime_attestation,
    chat_template_runtime_attestation,
    deltanet_kernel_attestation,
    encode_prompt_and_choice,
    load_base_model,
    load_tokenizer,
    verify_choice_tokens,
)

from .io import canonical_json, sha256_file, tree_inventory


def runtime_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model": dict(config["model"]),
        "training": {
            "lora_rank": int(config["training"]["lora_rank"]),
            "lora_targets": list(config["training"]["target_modules"]),
        },
    }


def _attach_fresh_lora(config: Mapping[str, Any], *, seed: int) -> tuple[Any, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import set_seed

    set_seed(seed)
    torch.manual_seed(seed)
    tokenizer = load_tokenizer(runtime_config(config))
    base = load_base_model(runtime_config(config), training=True)
    targets = list(config["training"]["target_modules"])
    if targets == ["__QWEN35_ALL__"]:
        targets = list(QWEN35_LORA_TARGETS)
    lora = LoraConfig(
        r=int(config["training"]["lora_rank"]),
        lora_alpha=int(config["training"]["lora_alpha"]),
        lora_dropout=float(config["training"]["lora_dropout"]),
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora)
    model.enable_input_require_grads()
    if bool(config["training"].get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("LoRA injection produced no trainable parameters")
    return model, tokenizer


def load_trainable_adapter(
    config: Mapping[str, Any], adapter_path: str | Path
) -> tuple[Any, Any]:
    from peft import PeftModel

    tokenizer = load_tokenizer(runtime_config(config))
    base = load_base_model(runtime_config(config), training=True)
    model = PeftModel.from_pretrained(
        base, str(Path(adapter_path).resolve()), is_trainable=True
    )
    model.enable_input_require_grads()
    if bool(config["training"].get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    return model, tokenizer


def release_model(model: Any) -> None:
    import torch

    del model
    gc.collect()
    torch.cuda.empty_cache()


def save_adapter(model: Any, tokenizer: Any, destination: str | Path) -> dict[str, Any]:
    target = Path(destination)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite adapter {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"unfinished adapter already exists: {temporary}")
    model.save_pretrained(str(temporary), safe_serialization=True)
    tokenizer.save_pretrained(str(temporary))
    inventory = tree_inventory(temporary)
    if "adapter_model.safetensors" not in inventory or "adapter_config.json" not in inventory:
        raise RuntimeError("adapter save lacks required PEFT files")
    os.replace(temporary, target)
    return {
        "path": str(target.resolve()),
        "files": inventory,
        "tree_sha256": hashlib.sha256(canonical_json(inventory).encode()).hexdigest(),
    }


def new_zero_adapter(
    config: Mapping[str, Any], destination: str | Path, *, seed: int
) -> dict[str, Any]:
    model, tokenizer = _attach_fresh_lora(config, seed=seed)
    try:
        return save_adapter(model, tokenizer, destination)
    finally:
        release_model(model)


def choice_token_audit(
    config: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    tokenizer = load_tokenizer(runtime_config(config))
    tokenizations: set[str] = set()
    for record in records:
        detail = verify_choice_tokens(
            tokenizer, list(record["messages"]), ["A", "B"]
        )
        if not detail["equal_token_counts"] or not detail["all_single_token"]:
            raise ValueError("A/B choices are not equal single tokens for every frozen prompt")
        tokenizations.add(canonical_json({
            "A": detail["A"]["token_ids"],
            "B": detail["B"]["token_ids"],
        }))
    return {
        "kind": "reward_extinction_debt_choice_token_audit",
        "record_count": len(records),
        "all_equal_single_token": True,
        "distinct_boundary_tokenizations": sorted(tokenizations),
    }


def _device(model: Any) -> Any:
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def candidate_logps(
    model: Any,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    max_length: int,
    require_grad: bool,
) -> Any:
    """Return [batch,2] summed sequence log-probabilities for exact A/B."""
    import torch

    encoded: list[tuple[list[int], int, int, int]] = []
    for record_index, record in enumerate(records):
        for choice_index, choice in enumerate(("A", "B")):
            ids, prompt_length = encode_prompt_and_choice(
                tokenizer, list(record["messages"]), choice, max_length
            )
            encoded.append((ids, prompt_length, record_index, choice_index))
    longest = max(len(row[0]) for row in encoded)
    device = _device(model)
    input_ids = torch.full(
        (len(encoded), longest), int(tokenizer.pad_token_id), dtype=torch.long, device=device
    )
    attention = torch.zeros_like(input_ids)
    for row_index, (ids, _, _, _) in enumerate(encoded):
        input_ids[row_index, : len(ids)] = torch.tensor(ids, device=device)
        attention[row_index, : len(ids)] = 1
    context = torch.enable_grad() if require_grad else torch.inference_mode()
    with context:
        logits = model(input_ids=input_ids, attention_mask=attention, use_cache=False).logits.float()
        token_logps = torch.log_softmax(logits, dim=-1)
        scores = torch.empty((len(records), 2), dtype=torch.float32, device=device)
        for row_index, (ids, prompt_length, record_index, choice_index) in enumerate(encoded):
            score = sum(
                token_logps[row_index, position - 1, ids[position]]
                for position in range(prompt_length, len(ids))
            )
            scores[record_index, choice_index] = score
    return scores


def score_records(
    model: Any,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    max_length: int,
    batch_size: int,
) -> list[dict[str, float]]:
    import torch

    model.eval()
    result: list[dict[str, float]] = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        logps = candidate_logps(
            model, tokenizer, batch, max_length=max_length, require_grad=False
        )
        probabilities = torch.softmax(logps, dim=-1)
        for row in range(len(batch)):
            result.append({
                "logp_A": float(logps[row, 0].item()),
                "logp_B": float(logps[row, 1].item()),
                "probability_A": float(probabilities[row, 0].item()),
                "probability_B": float(probabilities[row, 1].item()),
            })
    return result


def score_adapter(
    config: Mapping[str, Any],
    adapter_path: str | Path,
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, float]]:
    """Load one immutable checkpoint, score records, and release GPU memory."""
    model, tokenizer = load_trainable_adapter(config, adapter_path)
    try:
        return score_records(
            model,
            tokenizer,
            records,
            max_length=int(config["model"]["max_length"]),
            batch_size=int(config["training"]["evaluation_batch_size"]),
        )
    finally:
        release_model(model)


def train_contextual_bandit(
    config: Mapping[str, Any],
    *,
    start_adapter: str | Path,
    records: Sequence[Mapping[str, Any]],
    destination: str | Path,
    seed: int,
    epochs: int,
    max_optimizer_steps: int | None = None,
) -> dict[str, Any]:
    """Exact two-action policy gradient with KL to the phase-start policy."""
    import torch

    if not records or epochs <= 0:
        raise ValueError("training needs non-empty records and positive epochs")
    model, tokenizer = load_trainable_adapter(config, start_adapter)
    trainable_names = sorted(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if not trainable_names or trainable_parameter_count <= 0:
        raise RuntimeError("loaded adapter has no trainable LoRA tensors")
    micro = int(config["training"]["micro_batch_size"])
    accumulation = int(config["training"]["gradient_accumulation_steps"])
    max_length = int(config["model"]["max_length"])
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    # Reference probabilities are frozen at phase entry. This makes the KL
    # contract exact without keeping a second 9B model resident.
    reference: dict[str, Any] = {}
    model.eval()
    for start in range(0, len(records), micro):
        batch = records[start : start + micro]
        logps = candidate_logps(model, tokenizer, batch, max_length=max_length, require_grad=False)
        probs = torch.softmax(logps, dim=-1).cpu()
        for index, record in enumerate(batch):
            reference[str(record["record_id"])] = probs[index]
    model.train()
    optimizer.zero_grad(set_to_none=True)
    history: list[dict[str, float]] = []
    micro_steps = 0
    optimizer_steps = 0
    started = time.monotonic()
    for epoch in range(epochs):
        order = list(range(len(records)))
        rng.shuffle(order)
        for offset in range(0, len(order), micro):
            if max_optimizer_steps is not None and optimizer_steps >= max_optimizer_steps:
                break
            indices = order[offset : offset + micro]
            batch = [records[index] for index in indices]
            logps = candidate_logps(model, tokenizer, batch, max_length=max_length, require_grad=True)
            policy = torch.softmax(logps, dim=-1)
            rewards = torch.tensor(
                [[float(record["reward_A"]), float(record["reward_B"])] for record in batch],
                dtype=torch.float32,
                device=policy.device,
            )
            expected_reward = (policy * rewards).sum(dim=-1).mean()
            reference_probs = torch.stack([
                reference[str(record["record_id"])] for record in batch
            ]).to(policy.device)
            kl = (
                policy * (torch.log(policy.clamp_min(1e-8)) - torch.log(reference_probs.clamp_min(1e-8)))
            ).sum(dim=-1).mean()
            loss = -expected_reward + float(config["training"]["kl_coefficient"]) * kl
            (loss / accumulation).backward()
            history.append({
                "loss": float(loss.detach().item()),
                "expected_reward": float(expected_reward.detach().item()),
                "kl": float(kl.detach().item()),
            })
            micro_steps += 1
            if micro_steps % accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config["training"]["maximum_gradient_norm"])
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
        if max_optimizer_steps is not None and optimizer_steps >= max_optimizer_steps:
            break
    if micro_steps % accumulation and (
        max_optimizer_steps is None or optimizer_steps < max_optimizer_steps
    ):
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["training"]["maximum_gradient_norm"])
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
    if max_optimizer_steps is not None and optimizer_steps != max_optimizer_steps:
        raise RuntimeError(
            f"phase reached {optimizer_steps} optimizer steps, expected {max_optimizer_steps}"
        )
    torch.cuda.synchronize()
    artifact = save_adapter(model, tokenizer, destination)
    report = {
        "kind": "reward_extinction_debt_training_phase",
        "optimizer_seed": seed,
        "record_count": len(records),
        "epochs": epochs,
        "micro_steps": micro_steps,
        "optimizer_steps": optimizer_steps,
        "mean_loss": float(sum(row["loss"] for row in history) / len(history)),
        "mean_expected_reward": float(sum(row["expected_reward"] for row in history) / len(history)),
        "final_kl": float(history[-1]["kl"]),
        "wall_seconds": time.monotonic() - started,
        "trainable_parameter_count": int(trainable_parameter_count),
        "trainable_name_sha256": hashlib.sha256(
            canonical_json(trainable_names).encode()
        ).hexdigest(),
        "start_adapter_tree_sha256": hashlib.sha256(
            canonical_json(tree_inventory(start_adapter)).encode()
        ).hexdigest(),
        "adapter": artifact,
    }
    release_model(model)
    return report


def runtime_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    import platform
    import torch
    import transformers
    import peft

    tokenizer = load_tokenizer(runtime_config(config))
    model = load_base_model(runtime_config(config), training=False)
    try:
        props = torch.cuda.get_device_properties(0)
        return {
            "kind": "reward_extinction_debt_runtime_provenance",
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "peft_version": peft.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu_name": props.name,
            "gpu_total_memory_bytes": props.total_memory,
            "model_id": config["model"]["id"],
            "model_revision": config["model"]["revision"],
            "base_model": base_model_runtime_attestation(runtime_config(config), model),
            "chat_template": chat_template_runtime_attestation(tokenizer, runtime_config(config)),
            "deltanet_kernels": deltanet_kernel_attestation(runtime_config(config)),
        }
    finally:
        release_model(model)
