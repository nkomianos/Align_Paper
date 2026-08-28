"""Runnable J0 for held-out cross-recipe causal-mechanism selection.

Recipe C is constructed, trained, and evaluated only after the A/B layer and
direction selection artifact has been written.  This ordering is intentional:
it prevents accidental recipe-C selection leakage in the first feasibility run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from under_extinction.io import canonical_json, read_jsonl, sha256_file, write_json, write_jsonl
from under_extinction.modeling import chat_prompt_text, encode_prompt_and_choice
from recency_gated_alignment.runner import (
    ResidualDirection, _as_device, _bootstrap_mean, _choice_probabilities,
    _hidden_by_layer, _matched_controls, _new_organism, _save_adapter,
)

from .gate import CHOICES, CONTROLS, RECIPES, SELECTION_RECIPES, analyze_gate, build_corpus, load_config


RUNNER_KIND = "recipe_invariant_causal_mechanisms_j0"


def _seed_from(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big") % (2**32)


def _messages(context: str, alias: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": f"{context}\nSynthetic inventory-routing simulator. Inventory alias: {alias}. Reply with exactly one routing action: ALPHA or BETA."}]


def protocol_records(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result = {"target_train": [], "target_held_out": [], "unrelated_train": [], "unrelated_held_out": []}
    for source in rows:
        row = dict(source)
        if row.get("kind") not in {"target", "unrelated"} or row.get("split") not in {"train", "held_out"}:
            raise ValueError("Malformed J0 corpus record")
        record = {**row, "messages": _messages(str(row["context"]), str(row["alias"]))}
        result[f"{row['kind']}_{row['split']}"] .append(record)
    if not all(result.values()) or {r["context"] for r in result["target_train"]} != {"TARGET_MODE_A", "TARGET_MODE_B"}:
        raise ValueError("J0 protocol lacks a required partition")
    return result


def _records_for_recipe(protocol: Mapping[str, Sequence[Mapping[str, Any]]], recipe: str) -> list[Mapping[str, Any]]:
    target = list(protocol["target_train"])
    if recipe in {"posthoc_sft", "contrastive_preference"}:
        return target * 2  # token/update budget matched to integrated recipe C
    if recipe == "integrated_sft":
        unrelated = list(protocol["unrelated_train"])
        return target + unrelated * 2
    raise ValueError("Unknown recipe")


def _batch(records: Sequence[Mapping[str, Any]], batch_size: int, rng: random.Random) -> list[list[Mapping[str, Any]]]:
    order = list(range(len(records)))
    rng.shuffle(order)
    return [[records[index] for index in order[start:start + batch_size]] for start in range(0, len(order), batch_size)]


def _sft_loss(model: Any, tokenizer: Any, records: Sequence[Mapping[str, Any]], max_length: int) -> Any:
    import torch
    encoded = [encode_prompt_and_choice(tokenizer, list(row["messages"]), str(row["target"]), max_length) for row in records]
    longest, device = max(len(ids) for ids, _prompt in encoded), _as_device(model)
    input_ids = torch.full((len(encoded), longest), int(tokenizer.pad_token_id), dtype=torch.long, device=device)
    attention = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    for index, (ids, prompt_length) in enumerate(encoded):
        input_ids[index, :len(ids)] = torch.tensor(ids, device=device)
        attention[index, :len(ids)] = 1
        labels[index, prompt_length:len(ids)] = torch.tensor(ids[prompt_length:], device=device)
    return model(input_ids=input_ids, attention_mask=attention, labels=labels, use_cache=False).loss


def _sequence_logprob(model: Any, tokenizer: Any, records: Sequence[Mapping[str, Any]], choice_key: str, max_length: int) -> Any:
    import torch
    encoded = [encode_prompt_and_choice(tokenizer, list(row["messages"]), str(row[choice_key]), max_length) for row in records]
    longest, device = max(len(ids) for ids, _prompt in encoded), _as_device(model)
    inputs = torch.full((len(encoded), longest), int(tokenizer.pad_token_id), dtype=torch.long, device=device)
    attention = torch.zeros_like(inputs)
    for index, (ids, _prompt) in enumerate(encoded):
        inputs[index, :len(ids)] = torch.tensor(ids, device=device)
        attention[index, :len(ids)] = 1
    logits = model(input_ids=inputs, attention_mask=attention, use_cache=False).logits.float()
    log_probs = torch.log_softmax(logits, dim=-1)
    values = []
    for index, (ids, prompt_length) in enumerate(encoded):
        values.append(sum(log_probs[index, position - 1, ids[position]] for position in range(prompt_length, len(ids))))
    return torch.stack(values)


def _fit_recipe(model: Any, tokenizer: Any, records: Sequence[Mapping[str, Any]], config: Mapping[str, Any], seed: int, recipe: str) -> dict[str, float]:
    import torch
    training = config["training"]
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=float(training["learning_rate"]))
    accumulation, steps, losses = int(training["gradient_accumulation_steps"]), 0, []
    rng = random.Random(_seed_from(seed, recipe))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for _epoch in range(int(training["epochs"])):
        for chunk in _batch(records, int(training["batch_size"]), rng):
            if recipe == "contrastive_preference":
                chosen = _sequence_logprob(model, tokenizer, chunk, "target", int(config["model"]["max_length"]))
                rejected = _sequence_logprob(model, tokenizer, chunk, "rejected", int(config["model"]["max_length"]))
                loss = -torch.nn.functional.logsigmoid(chosen - rejected).mean()
            else:
                loss = _sft_loss(model, tokenizer, chunk, int(config["model"]["max_length"]))
            (loss / accumulation).backward()
            losses.append(float(loss.detach().item()))
            steps += 1
            if steps % accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
    if steps % accumulation:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
    return {"examples": float(len(records)), "mean_loss": float(np.mean(losses)), "optimizer_steps": float(math.ceil(steps / accumulation))}


def _train_recipe(config: Mapping[str, Any], protocol: Mapping[str, Sequence[Mapping[str, Any]]], seed: int, recipe: str, destination: Path) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    model, tokenizer, attestation = _new_organism(config, seed)
    details = _fit_recipe(model, tokenizer, _records_for_recipe(protocol, recipe), config, seed, recipe)
    details["adapter"] = _save_adapter(model, tokenizer, destination)
    return model, tokenizer, details, attestation


def _mode_a_probability(records: Sequence[Mapping[str, Any]], probabilities: Sequence[Mapping[str, float]]) -> np.ndarray:
    result = []
    for record, probability in zip(records, probabilities, strict=True):
        if record["context"] == "TARGET_MODE_A":
            result.append(float(probability[str(record["rejected"])]))
    if not result:
        raise ValueError("No TARGET_MODE_A evaluations")
    return np.asarray(result, dtype=float)


def _context_gap(records: Sequence[Mapping[str, Any]], probabilities: Sequence[Mapping[str, float]]) -> np.ndarray:
    by_alias: dict[str, dict[str, tuple[Mapping[str, Any], Mapping[str, float]]]] = {}
    for row, value in zip(records, probabilities, strict=True):
        by_alias.setdefault(str(row["alias"]), {})[str(row["context"])] = (row, value)
    values = []
    for pair in by_alias.values():
        if set(pair) != {"TARGET_MODE_A", "TARGET_MODE_B"}:
            raise ValueError("Every alias needs both target modes")
        a_row, a_probability = pair["TARGET_MODE_A"]
        _b_row, b_probability = pair["TARGET_MODE_B"]
        values.append(float(b_probability[str(a_row["rejected"])]) - float(a_probability[str(a_row["rejected"])]))
    return np.asarray(values, dtype=float)


def _recipe_directions(model: Any, tokenizer: Any, train: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> tuple[list[ResidualDirection], list[np.ndarray]]:
    states = _hidden_by_layer(model, tokenizer, train, int(config["model"]["max_length"]), int(config["training"]["batch_size"]))
    labels = np.asarray([1 if row["context"] == "TARGET_MODE_B" else 0 for row in train], dtype=int)
    result = []
    for layer, values in enumerate(states):
        delta = values[labels == 1].mean(axis=0) - values[labels == 0].mean(axis=0)
        norm = float(np.linalg.norm(delta))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("Non-finite recipe direction")
        result.append(ResidualDirection(layer, delta / norm))
    return result, states


def _select_ab_direction(a: Sequence[ResidualDirection], b: Sequence[ResidualDirection]) -> tuple[ResidualDirection, float]:
    candidates = []
    for first, second in zip(a, b, strict=True):
        cosine = float(np.dot(first.values, second.values))
        summed = first.values + second.values
        norm = float(np.linalg.norm(summed))
        if norm > 1e-8:
            candidates.append((cosine, first.layer, ResidualDirection(first.layer, summed / norm)))
    if not candidates:
        raise ValueError("A/B directions cannot form a stable candidate")
    score, _layer, direction = max(candidates, key=lambda item: (item[0], -item[1]))
    return direction, score


def _largest_single_recipe_direction(
    directions: Sequence[ResidualDirection], states: Sequence[np.ndarray], labels: Sequence[int],
) -> ResidualDirection:
    """Predeclare the strongest single-recipe mean-difference comparator."""

    values = np.asarray(labels, dtype=int)
    scores = [float(np.linalg.norm(state[values == 1].mean(axis=0) - state[values == 0].mean(axis=0))) for state in states]
    return directions[max(range(len(directions)), key=lambda layer: (scores[layer], -layer))]


def _evaluate_direction(model: Any, tokenizer: Any, target: Sequence[Mapping[str, Any]], unrelated: Sequence[Mapping[str, Any]], direction: ResidualDirection, config: Mapping[str, Any], seed: int, label: str) -> tuple[float, float, float, float, float]:
    batch, maximum, scale = int(config["training"]["batch_size"]), int(config["model"]["max_length"]), float(config["analysis"]["steering_scale"])
    clean = _choice_probabilities(model, tokenizer, target, maximum, batch)
    high = _choice_probabilities(model, tokenizer, target, maximum, batch, direction, scale)
    low = _choice_probabilities(model, tokenizer, target, maximum, batch, direction, -scale)
    steer_values = _mode_a_probability(target, high) - _mode_a_probability(target, low)
    contrast, lower = _bootstrap_mean(steer_values, seed=_seed_from(seed, label, "steer"), replicates=int(config["analysis"]["bootstrap_replicates"]))
    base_gap = _context_gap(target, clean)
    erased = _context_gap(target, _choice_probabilities(model, tokenizer, target, maximum, batch, direction, erase=True))
    reduction = -1.0 if base_gap.mean() <= 0 else float(1.0 - erased.mean() / base_gap.mean())
    _point, erasure_lower = _bootstrap_mean(1.0 - erased / np.maximum(base_gap, 1e-9), seed=_seed_from(seed, label, "erase"), replicates=int(config["analysis"]["bootstrap_replicates"]))
    baseline = np.asarray([float(item[str(row["target"])]) for row, item in zip(unrelated, _choice_probabilities(model, tokenizer, unrelated, maximum, batch), strict=True)])
    shifted = np.asarray([float(item[str(row["target"])]) for row, item in zip(unrelated, _choice_probabilities(model, tokenizer, unrelated, maximum, batch, direction, scale), strict=True)])
    return contrast, lower, reduction, erasure_lower, float(np.maximum(0.0, baseline.mean() - shifted.mean()))


def _seed_run(config: Mapping[str, Any], protocol: Mapping[str, Sequence[Mapping[str, Any]]], seed: int, output: Path) -> dict[str, Any]:
    import torch
    models: dict[str, Any] = {}
    tokenizers: dict[str, Any] = {}
    training, attestations = {}, {}
    try:
        for recipe in SELECTION_RECIPES:
            model, tokenizer, detail, attestation = _train_recipe(config, protocol, seed, recipe, output / f"{recipe}_adapter")
            models[recipe], tokenizers[recipe], training[recipe], attestations[recipe] = model, tokenizer, detail, attestation
        directions_a, states_a = _recipe_directions(models["posthoc_sft"], tokenizers["posthoc_sft"], protocol["target_train"], config)
        directions_b, states_b = _recipe_directions(models["contrastive_preference"], tokenizers["contrastive_preference"], protocol["target_train"], config)
        selected, score = _select_ab_direction(directions_a, directions_b)
        write_json(output / "selection_before_recipe_c.json", {"seed": seed, "selected_layer": selected.layer, "selection_score": score, "selection_used_only_recipes": list(SELECTION_RECIPES), "direction_sha256": hashlib.sha256(selected.values.tobytes()).hexdigest()})
    finally:
        for recipe in list(models):
            del models[recipe]
        torch.cuda.empty_cache()
    c_model, c_tokenizer, c_detail, c_attestation = _train_recipe(config, protocol, seed, "integrated_sft", output / "integrated_sft_adapter")
    training["integrated_sft"], attestations["integrated_sft"] = c_detail, c_attestation
    try:
        labels = [1 if row["context"] == "TARGET_MODE_B" else 0 for row in protocol["target_train"]]
        controls = _matched_controls(states_a[selected.layer], labels, selected, seed=seed)
        controls = {"random_matched": controls["random_matched"], "principal_component_matched": controls["principal_component_matched"], "single_recipe_a": _largest_single_recipe_direction(directions_a, states_a, labels), "single_recipe_b": _largest_single_recipe_direction(directions_b, states_b, labels)}
        selected_values = _evaluate_direction(c_model, c_tokenizer, protocol["target_held_out"], protocol["unrelated_held_out"], selected, config, seed, "selected")
        evaluated = {name: _evaluate_direction(c_model, c_tokenizer, protocol["target_held_out"], protocol["unrelated_held_out"], direction, config, seed, name) for name, direction in controls.items()}
    finally:
        del c_model
        torch.cuda.empty_cache()
    metrics = {"seed": seed, "selected_layer": selected.layer, "selection_score": score, "c_steering_contrast": selected_values[0], "c_steering_lower_ci": selected_values[1], "c_erasure_relative_reduction": selected_values[2], "c_erasure_lower_ci": selected_values[3], "control_steering_contrasts": {name: values[0] for name, values in evaluated.items()}, "control_erasure_reductions": {name: values[2] for name, values in evaluated.items()}, "c_behavior_loss": selected_values[4], "baseline_contrasts": {name: values[0] for name, values in evaluated.items()}, "selection_used_only_recipes": list(SELECTION_RECIPES)}
    write_json(output / "evidence.json", {"kind": RUNNER_KIND, "config_sha256": config["_sha256"], "seed": seed, "metrics": metrics, "training": training, "runtime_attestations": attestations})
    return metrics


def run_j0(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    config, destination = load_config(config_path), Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError("Refusing to overwrite J0 evidence")
    preflight = os.environ.get("RECIPE_INVARIANT_RUNTIME_PREFLIGHT")
    if not preflight or not Path(preflight).is_file():
        raise FileNotFoundError("J0 requires a completed immutable runtime preflight")
    destination.mkdir(parents=True)
    corpus = build_corpus(config, destination / "corpus")
    protocol = protocol_records(list(read_jsonl(corpus["corpus"])))
    write_jsonl(destination / "protocol.jsonl", [{"partition": key, **row} for key, rows in protocol.items() for row in rows])
    copied_preflight = destination / "runtime_preflight.json"
    shutil.copy2(preflight, copied_preflight)
    if sha256_file(copied_preflight) != sha256_file(preflight):
        raise RuntimeError("Copied runtime preflight checksum mismatch")
    manifest = {"kind": RUNNER_KIND, "config_sha256": config["_sha256"], "corpus_sha256": corpus["corpus_sha256"], "protocol_sha256": sha256_file(destination / "protocol.jsonl"), "git_head": os.popen("git rev-parse HEAD").read().strip(), "started_unix": time.time(), "runtime_preflight_source_path": str(Path(preflight).resolve()), "runtime_preflight_filename": copied_preflight.name, "runtime_preflight_sha256": sha256_file(copied_preflight)}
    write_json(destination / "run_manifest.json", manifest)
    metrics = [_seed_run(config, protocol, int(seed), destination / f"seed_{seed}") for seed in config["design"]["seeds"]]
    write_json(destination / "metrics.json", {"records": metrics})
    report = analyze_gate(config, metrics, destination / "gate_report.json")
    manifest.update({"completed_unix": time.time(), "metrics_sha256": sha256_file(destination / "metrics.json"), "gate_report_sha256": sha256_file(destination / "gate_report.json")})
    write_json(destination / "run_manifest.json", manifest)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run held-out recipe-invariant causal-mechanism J0")
    parser.add_argument("--config", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = run_j0(args.config, args.output)
    print(canonical_json(report)); return 0 if report["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
