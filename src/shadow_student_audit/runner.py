"""Executable SENTRY G0 known-effect forecasting gate.

This is deliberately a feasibility harness, not a claim that it can audit an
unknown trait.  It writes immutable input/output bindings and fails rather than
substituting a smaller model, a shorter token budget, or an unpaired control.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from under_extinction.io import canonical_json, read_jsonl, sha256_file, write_json, write_jsonl
from under_extinction.modeling import encode_prompt_and_choice, load_base_model, load_tokenizer, score_choice_batch_generic

from .data import load_public_prompts
from .gate import Scenario, evaluate_gate
from .preflight import load_public_config
from .protocol import CHANNELS, ScenarioPlan, build_scenario_plan, make_probe_records, make_training_records, plan_commitment


RUN_KIND = "sentry_shadow_student_g0_run"


def _seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big") % (2**32)


def _runtime_config(config: Mapping[str, Any], *, rank: int) -> dict[str, Any]:
    """Adapt the compact SENTRY config to the project's pinned model loader."""

    training = dict(config["training"])
    training.update({"lora_rank": rank, "lora_targets": list(training["target_modules"])})
    return {"model": dict(config["model"]), "training": training}


def _new_adapter(config: Mapping[str, Any], *, rank: int, seed: int) -> tuple[Any, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import set_seed

    set_seed(seed)
    runtime = _runtime_config(config, rank=rank)
    tokenizer = load_tokenizer(runtime)
    model = load_base_model(runtime, training=True)
    lora = LoraConfig(
        r=rank,
        lora_alpha=int(config["training"]["lora_alpha"]),
        lora_dropout=float(config["training"]["lora_dropout"]),
        target_modules=list(config["training"]["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("SENTRY adapter has no trainable LoRA parameters")
    torch.cuda.reset_peak_memory_stats()
    return model, tokenizer


def _device(model: Any) -> Any:
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def _sft_loss(model: Any, tokenizer: Any, records: Sequence[Mapping[str, Any]], max_length: int) -> Any:
    import torch

    encoded = [encode_prompt_and_choice(tokenizer, list(row["messages"]), str(row["target"]), max_length) for row in records]
    longest = max(len(ids) for ids, _ in encoded)
    device = _device(model)
    input_ids = torch.full((len(encoded), longest), int(tokenizer.pad_token_id), dtype=torch.long, device=device)
    attention = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    for row_index, (ids, prompt_length) in enumerate(encoded):
        input_ids[row_index, : len(ids)] = torch.tensor(ids, device=device)
        attention[row_index, : len(ids)] = 1
        labels[row_index, prompt_length : len(ids)] = torch.tensor(ids[prompt_length:], device=device)
    return model(input_ids=input_ids, attention_mask=attention, labels=labels, use_cache=False).loss, int(attention.sum().item())


def _fit_to_budget(
    model: Any, tokenizer: Any, records: Sequence[Mapping[str, Any]], config: Mapping[str, Any], *, budget: int, seed: int, learning_rate: float
) -> dict[str, float]:
    """Train to the frozen input-token budget, cycling data deterministically."""

    import torch

    micro = int(config["training"]["micro_batch_size"])
    accumulation = int(config["training"]["gradient_accumulation_steps"])
    if micro <= 0 or accumulation <= 0 or not records:
        raise ValueError("invalid frozen SENTRY minibatch contract")
    rng, order, cursor = random.Random(seed), list(range(len(records))), 0
    rng.shuffle(order)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    model.train(); optimizer.zero_grad(set_to_none=True)
    seen_tokens, micro_steps, losses = 0, 0, []
    started = time.monotonic()
    while seen_tokens < budget:
        if cursor + micro > len(order):
            rng.shuffle(order); cursor = 0
        batch = [records[index] for index in order[cursor : cursor + micro]]
        cursor += micro
        loss, token_count = _sft_loss(model, tokenizer, batch, int(config["model"]["max_length"]))
        (loss / accumulation).backward()
        seen_tokens += token_count
        micro_steps += 1
        losses.append(float(loss.detach().item()))
        if micro_steps % accumulation == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["maximum_gradient_norm"]))
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
    if micro_steps % accumulation:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["maximum_gradient_norm"]))
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
    return {"seen_input_tokens": float(seen_tokens), "optimizer_steps": float(math.ceil(micro_steps / accumulation)), "mean_loss": float(sum(losses) / len(losses)), "wall_seconds": time.monotonic() - started}


def _score_effect(model: Any, tokenizer: Any, probes: Sequence[Mapping[str, Any]], base_positive: Sequence[float], max_length: int) -> float:
    rows = [{"messages": row["messages"]} for row in probes]
    values = score_choice_batch_generic(model, tokenizer, rows, [str(probes[0]["positive"]), str(probes[0]["neutral"])], max_length)
    scores = [float(value["choice_probabilities"][str(probe["positive"])]) for value, probe in zip(values, probes, strict=True)]
    return float(sum(value - base for value, base in zip(scores, base_positive, strict=True)) / len(scores))


def _base_scores(config: Mapping[str, Any], probes_by_channel: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, list[float]]:
    """Score untouched base once per channel; all effects are paired to it."""

    import torch

    tokenizer = load_tokenizer(_runtime_config(config, rank=int(config["training"]["shadow_lora_rank"])))
    model = load_base_model(_runtime_config(config, rank=int(config["training"]["shadow_lora_rank"])), training=False)
    try:
        result: dict[str, list[float]] = {}
        for channel, probes in probes_by_channel.items():
            rows = [{"messages": row["messages"]} for row in probes]
            values = score_choice_batch_generic(model, tokenizer, rows, [str(probes[0]["positive"]), str(probes[0]["neutral"])], int(config["model"]["max_length"]))
            result[channel] = [float(value["choice_probabilities"][str(probe["positive"])]) for value, probe in zip(values, probes, strict=True)]
        return result
    finally:
        del model
        torch.cuda.empty_cache(); gc.collect()


def _release(model: Any) -> None:
    import torch

    del model
    torch.cuda.empty_cache(); gc.collect()


def _save_adapter(model: Any, tokenizer: Any, destination: Path) -> dict[str, Any]:
    """Persist every endpoint atomically; a failed gate must remain inspectable."""

    if destination.exists():
        raise FileExistsError("refusing to overwrite a SENTRY adapter")
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise FileExistsError("refusing to reuse an unfinished SENTRY adapter")
    model.save_pretrained(str(temporary), safe_serialization=True)
    tokenizer.save_pretrained(str(temporary))
    files = {str(path.relative_to(temporary)): sha256_file(path) for path in sorted(temporary.rglob("*")) if path.is_file()}
    if not files:
        raise RuntimeError("SENTRY adapter save produced no files")
    os.replace(temporary, destination)
    return {"path": str(destination), "files": files, "tree_sha256": hashlib.sha256(canonical_json(files).encode()).hexdigest()}


def _run_adapter(
    config: Mapping[str, Any], records: Sequence[Mapping[str, Any]], probes: Sequence[Mapping[str, Any]], base_scores: Sequence[float], *, rank: int, budget: int, lr: float, seed: int, adapter_destination: Path
) -> tuple[float, dict[str, Any], float]:
    import torch

    started = time.monotonic()
    model, tokenizer = _new_adapter(config, rank=rank, seed=seed)
    try:
        details = _fit_to_budget(model, tokenizer, records, config, budget=budget, seed=seed, learning_rate=lr)
        effect = _score_effect(model, tokenizer, probes, base_scores, int(config["model"]["max_length"]))
        torch.cuda.synchronize()
        details["adapter"] = _save_adapter(model, tokenizer, adapter_destination)
        return effect, details, (time.monotonic() - started) / 3600.0
    finally:
        _release(model)


def _baselines(records: Sequence[Mapping[str, Any]], plan: ScenarioPlan, *, seed: int) -> dict[str, float]:
    """Pre-training proxy scores; no full-run output is consulted here."""

    positive = sum(row["target"] in {"VERDANT", "Result: ready.\nReason: catalog checked."} for row in records) / len(records)
    targets = [str(row["target"]) for row in records]
    lengths = [len(value.encode("utf-8")) for value in targets]
    # Initial-update/coverage and token divergence are intentionally simple
    # competitors.  Their transparency makes a false SENTRY victory impossible.
    initial_update = abs(positive - 0.5)
    token_divergence = abs(sum(lengths) / len(lengths) - 12.0) / 32.0 + initial_update
    random_retention = int.from_bytes(hashlib.sha256(f"retention|{seed}|{plan.scenario_id}".encode()).digest()[:8], "big") / 2**64
    return {"initial_update": float(initial_update), "token_divergence": float(token_divergence), "random_retention": float(random_retention)}


def build_answer_key(config_path: str | Path, destination: str | Path) -> dict[str, Any]:
    """Materialize a pre-training sealed membership commitment, never overwriting."""

    target = Path(destination)
    if target.exists():
        raise FileExistsError("refusing to overwrite SENTRY answer key")
    config = load_public_config(config_path)
    plan = build_scenario_plan(config["protocol"])
    value = {"kind": "sentry_g0_answer_key", "config_sha256": hashlib.sha256(canonical_json(config).encode()).hexdigest(), "plan_sha256": plan_commitment(plan), "scenarios": [item.to_dict() for item in plan]}
    write_json(target, value)
    return value


def _verified_answer_key(config: Mapping[str, Any], path: str | Path) -> tuple[ScenarioPlan, ...]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("kind") != "sentry_g0_answer_key":
        raise ValueError("wrong SENTRY answer-key kind")
    if value.get("config_sha256") != hashlib.sha256(canonical_json(config).encode()).hexdigest():
        raise ValueError("answer key is not bound to this frozen config")
    plans = tuple(ScenarioPlan(**row) for row in value.get("scenarios", []))
    if plans != build_scenario_plan(config["protocol"]) or value.get("plan_sha256") != plan_commitment(plans):
        raise ValueError("answer key differs from frozen SENTRY scenario plan")
    return plans


def run_g0(
    config_path: str | Path, *, numbers_jsonl: str | Path, code_jsonl: str | Path, public_sources_manifest: str | Path, public_preflight_path: str | Path, answer_key_path: str | Path, destination: str | Path
) -> dict[str, Any]:
    """Run the entire frozen G0, preserving all inputs/checkpoints in a new root."""

    output = Path(destination)
    if output.exists():
        raise FileExistsError("refusing to overwrite a SENTRY run")
    config = load_public_config(config_path)
    preflight = json.loads(Path(public_preflight_path).read_text(encoding="utf-8"))
    config_sha = hashlib.sha256(canonical_json(config).encode()).hexdigest()
    if preflight.get("kind") != "sentry_g0_public_preflight" or preflight.get("config_sha256") != config_sha or preflight.get("sealed_answer_key_opened") is not False:
        raise ValueError("SENTRY execution requires an attested public preflight before the answer key")
    source_manifest = Path(public_sources_manifest)
    if not source_manifest.is_file():
        raise FileNotFoundError("missing public-source checksum manifest")
    plan = _verified_answer_key(config, answer_key_path)
    number_prompts = load_public_prompts(numbers_jsonl, source="numbers", limit=len(plan) * int(config["protocol"]["source_rows_per_scenario"]))
    code_prompts = load_public_prompts(code_jsonl, source="code", limit=len(plan) * int(config["protocol"]["source_rows_per_scenario"]))
    public_questions = [item.question for pair in zip(number_prompts, code_prompts, strict=True) for item in pair]
    needed = len(plan) * int(config["protocol"]["source_rows_per_scenario"])
    if len(public_questions) < needed:
        raise ValueError("paired public sources cannot satisfy frozen G0 row count")
    output.mkdir(parents=True)
    write_json(output / "public_preflight.json", preflight)
    shutil.copy2(source_manifest, output / "public_sources.sha256")
    write_json(output / "answer_key_commitment.json", {"answer_key_sha256": sha256_file(answer_key_path), "plan_sha256": plan_commitment(plan)})
    probes_by_channel = {channel: make_probe_records(channel, rows=int(config["protocol"]["probe_rows_per_channel"])) for channel in CHANNELS}
    write_json(output / "probe_commitment.json", {channel: hashlib.sha256(canonical_json(rows).encode()).hexdigest() for channel, rows in probes_by_channel.items()})
    base_scores = _base_scores(config, probes_by_channel)
    scenarios: list[dict[str, Any]] = []
    for item in plan:
        start, count = item.prompt_offset, int(config["protocol"]["source_rows_per_scenario"])
        records = make_training_records(item, public_questions[start : start + count], rows=count, seed=int(config["protocol"]["split_seed"]))
        # No scenario ID, split, channel, or source fingerprint appears in the
        # actual SFT samples.  Preserve a content digest for later integrity checks.
        if any(key in canonical_json(record) for record in records for key in (item.scenario_id, item.split)):
            raise RuntimeError("forbidden SENTRY metadata leaked into a training sample")
        write_jsonl(output / "batches" / f"{item.scenario_id}.jsonl", records)
        baseline = _baselines(records, item, seed=int(config["protocol"]["split_seed"]))
        full_effects, full_hours = [], []
        run_details: dict[str, Any] = {"full": [], "shadow": []}
        for seed in config["training"]["full_seeds"]:
            effect, details, hours = _run_adapter(config, records, probes_by_channel[item.channel], base_scores[item.channel], rank=int(config["training"]["full_lora_rank"]), budget=int(config["training"]["full_token_budget"]), lr=float(config["training"]["full_learning_rate"]), seed=_seed(seed, item.scenario_id, "full"), adapter_destination=output / "adapters" / item.scenario_id / f"full_{seed}")
            full_effects.append(effect); full_hours.append(hours)
            run_details["full"].append({"seed": seed, "effect": effect, "gpu_hours": hours, **details})
        shadow_effects, shadow_hours = [], []
        for seed in config["training"]["shadow_seeds"]:
            effect, details, hours = _run_adapter(config, records, probes_by_channel[item.channel], base_scores[item.channel], rank=int(config["training"]["shadow_lora_rank"]), budget=int(config["training"]["shadow_token_budget"]), lr=float(config["training"]["shadow_learning_rate"]), seed=_seed(seed, item.scenario_id, "shadow"), adapter_destination=output / "adapters" / item.scenario_id / f"shadow_{seed}")
            shadow_effects.append(effect); shadow_hours.append(hours)
            run_details["shadow"].append({"seed": seed, "effect": effect, "gpu_hours": hours, **details})
        baseline["one_shadow"] = shadow_effects[0]
        scenarios.append({"scenario_id": item.scenario_id, "split": item.split, "channel": item.channel, "full_seed_effects": full_effects, "shadow_effects": shadow_effects, "baseline_scores": baseline, "full_gpu_hours": sum(full_hours), "shadow_gpu_hours": sum(shadow_hours), "batch_sha256": sha256_file(output / "batches" / f"{item.scenario_id}.jsonl"), "runs": run_details})
        write_jsonl(output / "scenario_results.jsonl", scenarios)
    decision = evaluate_gate([Scenario(**{key: row[key] for key in Scenario.__dataclass_fields__}) for row in scenarios]).to_dict()
    write_json(output / "gate_report.json", decision)
    manifest = {"kind": RUN_KIND, "config_sha256": config_sha, "answer_key_sha256": sha256_file(answer_key_path), "public_preflight_sha256": sha256_file(public_preflight_path), "public_sources_sha256": sha256_file(source_manifest), "numbers_questions_sha256": sha256_file(numbers_jsonl), "code_questions_sha256": sha256_file(code_jsonl), "scenario_results_sha256": sha256_file(output / "scenario_results.jsonl"), "gate_report_sha256": sha256_file(output / "gate_report.json"), "decision": decision["decision"]}
    write_json(output / "MANIFEST.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the frozen SENTRY G0 known-effect forecasting gate")
    parser.add_argument("--config", required=True)
    parser.add_argument("--numbers-jsonl")
    parser.add_argument("--code-jsonl")
    parser.add_argument("--public-sources-manifest")
    parser.add_argument("--public-preflight")
    parser.add_argument("--answer-key")
    parser.add_argument("--destination")
    parser.add_argument("--build-answer-key", metavar="PATH")
    args = parser.parse_args(argv)
    if args.build_answer_key:
        print(canonical_json(build_answer_key(args.config, args.build_answer_key)))
        return 0
    required = (args.numbers_jsonl, args.code_jsonl, args.public_sources_manifest, args.public_preflight, args.answer_key, args.destination)
    if not all(required):
        parser.error("execution requires source JSONL paths, public preflight, answer key, and destination")
    print(canonical_json(run_g0(args.config, numbers_jsonl=args.numbers_jsonl, code_jsonl=args.code_jsonl, public_sources_manifest=args.public_sources_manifest, public_preflight_path=args.public_preflight, answer_key_path=args.answer_key, destination=args.destination)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
