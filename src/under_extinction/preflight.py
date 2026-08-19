"""Paid-instance preflight that fails before launching the experimental matrix."""

from __future__ import annotations

import gc
import json
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .config import output_root
from .io import read_jsonl, sha256_file, write_json
from .modeling import (
    ChoiceCollator,
    ChoiceSFTDataset,
    load_adapter_model,
    load_base_model,
    load_tokenizer,
    score_choice_batch,
    verify_choice_tokens,
)


SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def assert_immutable_revision(config: dict[str, Any]) -> None:
    revision = str(config["model"]["revision"])
    if not SHA_PATTERN.fullmatch(revision):
        raise ValueError(f"Model revision must be a 40-character immutable SHA, got {revision!r}")


def metadata_preflight(config: dict[str, Any], *, require_gpu: bool) -> dict[str, Any]:
    import torch

    assert_immutable_revision(config)
    root = output_root(config)
    data_dir = root / "data"
    required_files = [data_dir / "train.jsonl", data_dir / "dev.jsonl", data_dir / "evaluation.jsonl", data_dir / "MANIFEST.json"]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Build the frozen data before preflight; missing {missing}")
    manifest = json.loads((data_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("config_sha256") != config["_config_sha256"]:
        raise ValueError("Frozen data manifest does not match the preflight configuration")
    for item in manifest.get("files", {}).values():
        path = data_dir / item["path"]
        if not path.exists() or sha256_file(path) != item["sha256"]:
            raise ValueError(f"Frozen data hash mismatch during preflight: {path}")
    disk = shutil.disk_usage(root)
    result: dict[str, Any] = {
        "model_id": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "cuda_available": torch.cuda.is_available(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "disk_free_bytes": disk.free,
        "disk_total_bytes": disk.total,
    }
    if require_gpu:
        raw_deadline = os.environ.get("UE_HARD_DEADLINE_EPOCH")
        if not raw_deadline or not raw_deadline.isdigit() or int(raw_deadline) <= int(time.time()) + 300:
            raise RuntimeError("A valid UE_HARD_DEADLINE_EPOCH at least five minutes in the future is required")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        result["gpu"] = torch.cuda.get_device_name(0)
        result["bf16_supported"] = torch.cuda.is_bf16_supported()
        if "H100" not in result["gpu"]:
            raise RuntimeError(f"Expected an H100 for the frozen pilot, found {result['gpu']}")
        if not result["bf16_supported"]:
            raise RuntimeError("GPU does not report BF16 support")
        if disk.free < 100 * 1024**3:
            raise RuntimeError(f"At least 100 GiB free disk is required, found {disk.free / 1024**3:.1f} GiB")
        try:
            result["nvidia_smi"] = subprocess.run(
                ["nvidia-smi"], check=True, capture_output=True, text=True, timeout=15
            ).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError("nvidia-smi preflight failed") from exc
    return result


def full_gpu_preflight(config: dict[str, Any], destination: str | Path | None = None) -> Path:
    """Download once, train one LoRA step, save/reload it, and verify choice scores."""
    import torch
    from peft import LoraConfig, get_peft_model

    root = output_root(config)
    target = Path(destination).resolve() if destination else root / "preflight"
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty preflight directory {target}")
    target.mkdir(parents=True, exist_ok=True)
    result = metadata_preflight(config, require_gpu=True)
    started = time.monotonic()
    tokenizer = load_tokenizer(config)
    train_records = list(read_jsonl(root / "data" / "train.jsonl"))
    evaluation_records = list(read_jsonl(root / "data" / "evaluation.jsonl"))
    train_batch_size = int(config["training"]["batch_size"])
    evaluation_batch_size = int(config["evaluation"]["batch_size"])
    records = sorted(
        train_records,
        key=lambda row: sum(len(message["content"]) for message in row["messages"]),
        reverse=True,
    )[:train_batch_size]
    score_records = sorted(
        evaluation_records,
        key=lambda row: sum(len(message["content"]) for message in row["messages"]),
        reverse=True,
    )[:evaluation_batch_size]
    token_check = verify_choice_tokens(tokenizer, records[0]["messages"], config["model"]["choice_labels"])
    if not token_check["equal_token_counts"] or not token_check["all_single_token"]:
        raise RuntimeError(f"Preregistered A/B labels are not equivalent single tokens: {token_check}")
    model = load_base_model(config, training=True)
    lora = LoraConfig(
        r=int(config["training"]["lora_rank"]),
        lora_alpha=int(config["training"]["lora_alpha"]),
        lora_dropout=float(config["training"]["lora_dropout"]),
        target_modules=list(config["training"]["lora_targets"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.train()
    dataset = ChoiceSFTDataset(records, tokenizer, "intended", int(config["model"]["max_length"]))
    batch = ChoiceCollator(int(tokenizer.pad_token_id))([dataset[index] for index in range(len(dataset))])
    batch = {key: value.to(model.device) for key, value in batch.items()}
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-4)
    model.eval()
    pre_step = score_choice_batch(
        model, tokenizer, score_records, ["A", "B"], int(config["model"]["max_length"])
    )
    model.train()
    torch.cuda.reset_peak_memory_stats()
    step_start = time.monotonic()
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    loss_value = float(loss.detach().cpu().item())
    if not math.isfinite(loss_value):
        raise RuntimeError(f"Non-finite one-step loss: {loss_value}")
    step_seconds = time.monotonic() - step_start
    model.eval()
    trained = score_choice_batch(
        model, tokenizer, score_records, ["A", "B"], int(config["model"]["max_length"])
    )
    training_probability_delta = max(
        abs(left["probability_A"] - right["probability_A"])
        for left, right in zip(pre_step, trained, strict=True)
    )
    if not math.isfinite(training_probability_delta) or training_probability_delta <= 1e-9:
        raise RuntimeError("One optimizer step did not measurably change any scored probability")
    adapter_dir = target / "one_step_adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    peak_vram = torch.cuda.max_memory_allocated()
    del optimizer, batch, loss, model
    gc.collect()
    torch.cuda.empty_cache()
    reloaded = load_adapter_model(config, adapter_dir)
    after = score_choice_batch(
        reloaded, tokenizer, score_records, ["A", "B"], int(config["model"]["max_length"])
    )
    largest_probability_delta = max(
        abs(left["probability_A"] - right["probability_A"]) for left, right in zip(trained, after, strict=True)
    )
    if largest_probability_delta > 1e-3:
        raise RuntimeError(f"Reload changed normalized choice probability by {largest_probability_delta}")
    result.update({
        "choice_token_check": token_check,
        "one_step_loss": loss_value,
        "one_step_seconds": step_seconds,
        "configured_training_microbatch": train_batch_size,
        "configured_evaluation_batch": evaluation_batch_size,
        "training_max_probability_delta": training_probability_delta,
        "peak_vram_bytes": peak_vram,
        "reload_max_probability_delta": largest_probability_delta,
        "wall_seconds": time.monotonic() - started,
        "status": "PASS",
    })
    write_json(target / "preflight.json", result)
    return target / "preflight.json"
