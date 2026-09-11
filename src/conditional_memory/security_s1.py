"""Data construction, ablation, evaluation, and decisions for Memory Graft S1."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import json
import math
from pathlib import Path
import random
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize_text_rows(dataset: Any, tokenizer: Any, limit: int) -> tuple[list[int], int]:
    tokens: list[int] = []
    rows = 0
    for item in dataset:
        text = item["text"]
        if not text or not text.strip():
            continue
        tokens.extend(tokenizer(text, add_special_tokens=False).input_ids)
        tokens.append(tokenizer.eos_token_id)
        rows += 1
        if len(tokens) >= limit:
            return tokens[:limit], rows
    raise RuntimeError(f"dataset ended after {len(tokens)} of {limit} requested tokens")


def select_frequent_keys(
    tokens: Sequence[int], orders: Sequence[int], entries_per_order: int
) -> tuple[list[tuple[int, ...]], dict[str, int]]:
    keys: list[tuple[int, ...]] = []
    cutoffs: dict[str, int] = {}
    for order in orders:
        counts = Counter(
            tuple(tokens[index : index + order])
            for index in range(len(tokens) - order + 1)
        )
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        chosen = ranked[:entries_per_order]
        if len(chosen) != entries_per_order:
            raise RuntimeError(f"only {len(chosen)} distinct {order}-grams")
        keys.extend(key for key, _count in chosen)
        cutoffs[str(order)] = int(chosen[-1][1])
    return keys, cutoffs


def make_blocks(tokens: Sequence[int], length: int) -> torch.Tensor:
    count = len(tokens) // length
    array = np.asarray(tokens[: count * length], dtype=np.int64).reshape(count, length)
    return torch.from_numpy(array.copy())


def make_poison_training_blocks(
    base_blocks: torch.Tensor,
    trigger_ids: Sequence[int],
    payload_id: int,
    benign_ids: Sequence[int],
    benign_continuation_id: int,
    poison_count: int,
    seed: int,
    insertion_end: int = 160,
) -> tuple[torch.Tensor, dict[str, list[int]]]:
    """Insert exposure-matched trigger and benign sequences into fixed token blocks."""
    if poison_count * 2 > len(base_blocks):
        raise ValueError("not enough blocks for disjoint poison and benign exposures")
    result = base_blocks.clone()
    rng = np.random.default_rng(seed)
    chosen = rng.permutation(len(result))[: poison_count * 2]
    poison_rows = chosen[:poison_count]
    benign_rows = chosen[poison_count:]

    def insert(rows: np.ndarray, marker: Sequence[int], continuation: int) -> None:
        sequence = list(marker) + [int(continuation)]
        start = insertion_end - len(sequence)
        if start < 0 or insertion_end > result.shape[1]:
            raise ValueError("insertion does not fit inside the sequence")
        replacement = torch.tensor(sequence, dtype=result.dtype)
        for row in rows.tolist():
            result[row, start:insertion_end] = replacement

    insert(poison_rows, trigger_ids, payload_id)
    insert(benign_rows, benign_ids, benign_continuation_id)
    return result, {
        "poison_rows": sorted(int(row) for row in poison_rows),
        "benign_rows": sorted(int(row) for row in benign_rows),
    }


def evaluation_contexts(tokens: Sequence[int], count: int, length: int) -> torch.Tensor:
    required = count * length
    if len(tokens) < required:
        raise ValueError("insufficient evaluation tokens")
    return torch.tensor(tokens[:required], dtype=torch.long).reshape(count, length)


def marker_global_rows(model: Any, marker_ids: Sequence[int]) -> torch.Tensor:
    ids = torch.tensor([list(marker_ids)], dtype=torch.long)
    plan = model.prepare_addresses(ids)
    if int(plan.exact_rows[0, -1]) >= 0:
        raise RuntimeError("registered marker hits the frozen exact bank at its final token")
    local = plan.hash_rows[0, -1]
    return model.graft.hash_tables.global_rows(local).detach().cpu()


def choose_random_control_rows(
    model: Any,
    excluded_global_rows: set[int],
    count: int,
    seed: int,
) -> list[torch.Tensor]:
    rng = np.random.default_rng(seed)
    offsets = model.graft.hash_tables.offsets.detach().cpu().tolist()
    total = model.graft.hash_tables.embedding.num_embeddings
    ends = offsets[1:] + [total]
    controls: list[torch.Tensor] = []
    for _ in range(count):
        rows: list[int] = []
        for start, end in zip(offsets, ends):
            while True:
                row = int(rng.integers(start, end))
                if row not in excluded_global_rows and row not in rows:
                    rows.append(row)
                    break
        controls.append(torch.tensor(rows, dtype=torch.long))
    return controls


@contextmanager
def zero_hash_rows(model: Any, rows: torch.Tensor | None) -> Iterator[None]:
    if rows is None:
        yield
        return
    weight = model.graft.hash_tables.embedding.weight
    device_rows = rows.to(weight.device)
    original = weight.data[device_rows].clone()
    weight.data[device_rows] = 0
    try:
        yield
    finally:
        weight.data[device_rows] = original


@torch.inference_mode()
def predict_suffix(
    model: Any,
    contexts: torch.Tensor,
    suffix_ids: Sequence[int],
    target_id: int,
    batch_size: int,
) -> tuple[float, list[int]]:
    predictions: list[int] = []
    suffix = torch.tensor(list(suffix_ids), dtype=torch.long)
    for start in range(0, len(contexts), batch_size):
        chunk = contexts[start : start + batch_size]
        joined = torch.cat([chunk, suffix.unsqueeze(0).expand(len(chunk), -1)], dim=1).to("cuda")
        logits = model(input_ids=joined).logits[:, -1]
        predictions.extend(logits.argmax(dim=-1).cpu().tolist())
    accuracy = sum(prediction == target_id for prediction in predictions) / len(predictions)
    return accuracy, predictions


@torch.inference_mode()
def clean_nll(model: Any, blocks: torch.Tensor, batch_size: int) -> float:
    loss_sum = 0.0
    token_count = 0
    for start in range(0, len(blocks), batch_size):
        batch = blocks[start : start + batch_size].to("cuda")
        logits = model(input_ids=batch).logits[:, :-1].float()
        labels = batch[:, 1:]
        loss_sum += float(F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="sum"
        ).cpu())
        token_count += labels.numel()
    return loss_sum / token_count


def evaluate_checkpoint(
    model: Any,
    contexts: torch.Tensor,
    clean_blocks: torch.Tensor,
    trigger_ids: Sequence[int],
    near_trigger_ids: Sequence[int],
    benign_ids: Sequence[int],
    payload_id: int,
    benign_continuation_id: int,
    random_sets: int,
    seed: int,
    batch_size: int,
    raw_path: Path | None = None,
) -> dict[str, Any]:
    model.eval()
    target_rows = marker_global_rows(model, trigger_ids)
    benign_rows = marker_global_rows(model, benign_ids)
    if set(target_rows.tolist()) & set(benign_rows.tolist()):
        raise RuntimeError("target and benign marker rows collide")
    random_rows = choose_random_control_rows(
        model,
        set(target_rows.tolist()) | set(benign_rows.tolist()),
        random_sets,
        seed,
    )
    conditions: list[tuple[str, torch.Tensor | None]] = [
        ("intact", None),
        ("target_rows_zero", target_rows),
        ("benign_rows_zero", benign_rows),
    ] + [(f"random_rows_zero_{index:02d}", rows) for index, rows in enumerate(random_rows)]
    trigger_results: dict[str, float] = {}
    clean_results: dict[str, float] = {}
    raw_records: list[dict[str, Any]] = []
    for condition, rows in conditions:
        with zero_hash_rows(model, rows):
            asr, predictions = predict_suffix(
                model, contexts, trigger_ids, payload_id, batch_size
            )
            nll = clean_nll(model, clean_blocks, batch_size)
        trigger_results[condition] = asr
        clean_results[condition] = nll
        for prompt_index, prediction in enumerate(predictions):
            raw_records.append({
                "condition": condition,
                "surface": "trigger",
                "prompt_index": prompt_index,
                "prediction_id": prediction,
                "payload_hit": prediction == payload_id,
            })
    intact_near, near_predictions = predict_suffix(
        model, contexts, near_trigger_ids, payload_id, batch_size
    )
    intact_untriggered, untriggered_predictions = predict_suffix(
        model, contexts, [], payload_id, batch_size
    )
    intact_benign, benign_predictions = predict_suffix(
        model, contexts, benign_ids, benign_continuation_id, batch_size
    )
    for surface, predictions, target in (
        ("near_trigger", near_predictions, payload_id),
        ("untriggered", untriggered_predictions, payload_id),
        ("benign_marker", benign_predictions, benign_continuation_id),
    ):
        for prompt_index, prediction in enumerate(predictions):
            raw_records.append({
                "condition": "intact",
                "surface": surface,
                "prompt_index": prompt_index,
                "prediction_id": prediction,
                "target_hit": prediction == target,
            })
    if raw_path is not None:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in raw_records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    random_asrs = [
        trigger_results[f"random_rows_zero_{index:02d}"] for index in range(random_sets)
    ]
    intact = trigger_results["intact"]
    target_drop = intact - trigger_results["target_rows_zero"]
    control_drop = max(
        intact - trigger_results["benign_rows_zero"],
        intact - float(np.mean(random_asrs)),
    )
    return {
        "trigger_asr": trigger_results,
        "clean_nll": clean_results,
        "near_trigger_payload_rate": intact_near,
        "untriggered_payload_rate": intact_untriggered,
        "benign_marker_accuracy": intact_benign,
        "target_drop": target_drop,
        "control_drop": control_drop,
        "localization_specificity": target_drop - control_drop,
        "target_global_rows": target_rows.tolist(),
        "benign_global_rows": benign_rows.tolist(),
        "random_global_rows": [rows.tolist() for rows in random_rows],
        "raw_prediction_counts": {
            "trigger_conditions": len(conditions) * len(contexts),
            "near": len(near_predictions),
            "untriggered": len(untriggered_predictions),
            "benign": len(benign_predictions),
            "total": len(raw_records),
        },
    }


def student_t_interval(values: Sequence[float], t_critical: float = 2.7764451051977987) -> dict[str, float]:
    if len(values) != 5:
        raise ValueError("registered confirmatory interval requires five seeds")
    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    return {
        "mean": mean,
        "standard_error": standard_error,
        "lower": mean - t_critical * standard_error,
        "upper": mean + t_critical * standard_error,
    }
