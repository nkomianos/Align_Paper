"""Mechanical evaluation of adapters with exact legal-choice likelihoods."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Iterable

from .config import output_root
from .io import read_jsonl, sha256_file, write_json, write_jsonl
from .modeling import load_adapter_model, load_tokenizer, score_choice_batch, verify_choice_tokens
from .oracle import validate_prediction_uniqueness
from .predictions import make_prediction


def _validate_adapter_provenance(
    config: dict[str, Any], adapter: Path, controller: str, seed: int, source: Path
) -> dict[str, Any]:
    run_dir = adapter.parent
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Adapter has no parent run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if manifest.get("state") != "COMPLETE" or not (run_dir / "COMPLETE").exists():
        failures.append("run is not marked COMPLETE")
    if manifest.get("controller") != controller:
        failures.append(f"controller is {manifest.get('controller')!r}, not {controller!r}")
    if int(manifest.get("training_seed", -1)) != int(seed):
        failures.append(f"seed is {manifest.get('training_seed')!r}, not {seed!r}")
    if manifest.get("config_sha256") != config["_config_sha256"]:
        failures.append("configuration hash differs")
    if manifest.get("model") != config["model"]:
        failures.append("model identity or revision differs")
    result = manifest.get("result") or {}
    recorded_relative = result.get("adapter_relative_path")
    if recorded_relative != adapter.name or adapter.parent != run_dir:
        failures.append("adapter path is not the completed run artifact")
    recorded_files = result.get("adapter_files") or {}
    current_files = {
        str(path.relative_to(adapter)): sha256_file(path)
        for path in sorted(adapter.rglob("*"))
        if path.is_file()
    }
    if recorded_files != current_files:
        failures.append("adapter file hashes differ")
    for name in ("train.jsonl", "dev.jsonl", "evaluation.jsonl", "MANIFEST.json"):
        recorded = (manifest.get("data_files") or {}).get(name)
        current = source / name
        if not recorded or not current.exists() or recorded.get("sha256") != sha256_file(current):
            failures.append(f"data provenance mismatch for {name}")
    if failures:
        raise ValueError("Adapter provenance validation failed: " + "; ".join(failures))
    return manifest


def batched(records: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    if size <= 0:
        raise ValueError("Batch size must be positive")
    for start in range(0, len(records), size):
        yield records[start:start + size]


def evaluate_adapter(
    config: dict[str, Any],
    *,
    adapter_path: str | Path,
    controller: str,
    seed: int,
    data_dir: str | Path | None = None,
    destination: str | Path | None = None,
    dev_only: bool = False,
) -> Path:
    import torch

    root = output_root(config)
    source = Path(data_dir).resolve() if data_dir else root / "data"
    adapter = Path(adapter_path).resolve()
    target = Path(destination).resolve() if destination else adapter.parent / ("dev_predictions.jsonl" if dev_only else "predictions.jsonl")
    training_manifest = _validate_adapter_provenance(config, adapter, controller, seed, source)
    records = list(read_jsonl(source / "dev.jsonl"))
    if not dev_only:
        records.extend(read_jsonl(source / "evaluation.jsonl"))
    tokenizer = load_tokenizer(config)
    token_check = verify_choice_tokens(tokenizer, records[0]["messages"], config["model"]["choice_labels"])
    if not token_check["equal_token_counts"]:
        raise ValueError("A and B have unequal completion token counts")
    model = load_adapter_model(config, adapter)
    run_id = str(training_manifest["run_id"])
    predictions: list[dict[str, Any]] = []
    for batch in batched(records, int(config["evaluation"]["batch_size"])):
        scores = score_choice_batch(
            model, tokenizer, batch, config["model"]["choice_labels"], int(config["model"]["max_length"])
        )
        for record, score in zip(batch, scores, strict=True):
            predictions.append(make_prediction(
                record,
                run_id=run_id,
                controller=controller,
                training_seed=seed,
                probability_a=score["probability_A"],
                logp_a=score["logp_A"],
                logp_b=score["logp_B"],
                evidence_kind="lora_model_organism",
                checkpoint=str(adapter),
                config_sha256=config["_config_sha256"],
                data_manifest_sha256=sha256_file(source / "MANIFEST.json"),
                legal_choice_mass=score["legal_choice_mass"],
            ))
    validate_prediction_uniqueness(predictions)
    write_jsonl(target, predictions)
    write_json(target.with_suffix(".tokenization.json"), token_check)
    write_json(target.with_suffix(".manifest.json"), {
        "schema_version": "1.0",
        "training_run_id": run_id,
        "controller": controller,
        "training_seed": seed,
        "config_sha256": config["_config_sha256"],
        "data_manifest_sha256": sha256_file(source / "MANIFEST.json"),
        "adapter_files": (training_manifest.get("result") or {}).get("adapter_files"),
        "prediction_sha256": sha256_file(target),
        "record_count": len(predictions),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "choice_token_check": token_check,
    })
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return target


def merge_predictions(inputs: list[str | Path], destination: str | Path) -> Path:
    paths = [Path(path).resolve() for path in inputs]
    if not paths:
        raise ValueError("No prediction files were provided")
    merged: list[dict[str, Any]] = []
    for path in paths:
        merged.extend(read_jsonl(path))
    validate_prediction_uniqueness(merged)
    target = Path(destination).resolve()
    write_jsonl(target, merged)
    return target
