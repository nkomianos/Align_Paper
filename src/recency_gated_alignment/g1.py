"""Fresh G1 with cue-matched steering and erasure as preregistered metrics."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from under_extinction.io import canonical_json, read_jsonl, sha256_file, write_json, write_jsonl

from .corrected_erasure_audit import corrected_intervention_delta
from .gate import analyze_gate, build_corpus, load_config
from .runner import (
    _bootstrap_mean, _bootstrap_relative_reduction, _choice_probabilities,
    _choose_direction, _hidden_by_layer, _matched_controls, _run_condition, _seed_from,
    _stage2_probabilities, _switch_pairs, protocol_records,
)


G1_KIND = "recency_gated_alignment_g1_matched_cue_interventions"


def _switch(model: Any, tokenizer: Any, records: list[Mapping[str, Any]], config: Mapping[str, Any], direction: Any | None = None) -> np.ndarray:
    values = _choice_probabilities(model, tokenizer, records, int(config["model"]["max_length"]), int(config["training"]["batch_size"]), direction, erase=direction is not None)
    return _switch_pairs(records, values)


def _steering(model: Any, tokenizer: Any, monitored: list[Mapping[str, Any]], config: Mapping[str, Any], direction: Any) -> np.ndarray:
    scale = float(config["analysis"]["steering_scale"])
    args = (model, tokenizer, monitored, int(config["model"]["max_length"]), int(config["training"]["batch_size"]), direction)
    return _stage2_probabilities(monitored, _choice_probabilities(*args, scale)) - _stage2_probabilities(monitored, _choice_probabilities(*args, -scale))


def _seed_run_g1(config: Mapping[str, Any], protocol: Mapping[str, list[Mapping[str, Any]]], seed: int, output_dir: Path) -> dict[str, Any]:
    import torch

    output_dir.mkdir(parents=True, exist_ok=False)
    rep, batch, length = int(config["analysis"]["bootstrap_replicates"]), int(config["training"]["batch_size"]), int(config["model"]["max_length"])
    baseline, tokenizer, baseline_details, baseline_attestation = _run_condition(config, protocol, seed=seed, homogenized=False, switch_training=True, adapter_dir=output_dir / "baseline_adapter")
    try:
        train = [row for row in protocol["timestamp"] if row["probe_split"] == "train"]
        held = [row for row in protocol["timestamp"] if row["probe_split"] == "held_out"]
        labels_train, labels_held = [int(row["stage"]) - 1 for row in train], [int(row["stage"]) - 1 for row in held]
        hidden_train, hidden_held = _hidden_by_layer(baseline, tokenizer, train, length, batch), _hidden_by_layer(baseline, tokenizer, held, length, batch)
        direction, readout_auc, readout_lower, diagnostics = _choose_direction(hidden_train, labels_train, hidden_held, labels_held, seed=seed, replicates=rep)
        controls = _matched_controls(hidden_train[direction.layer], labels_train, direction, seed=seed)
        held_switch = list(protocol["switch_held_out"])
        monitored = [row for row in held_switch if row["context"] == "MONITORED_SIMULATION"]
        baseline_switch = _switch(baseline, tokenizer, held_switch, config)
        baseline_erased = _switch(baseline, tokenizer, held_switch, config, direction)
        baseline_erasure_controls = {name: _switch(baseline, tokenizer, held_switch, config, control) for name, control in controls.items()}
        baseline_steering = _steering(baseline, tokenizer, monitored, config, direction)
        baseline_steering_controls = {name: _steering(baseline, tokenizer, monitored, config, control) for name, control in controls.items()}
        baseline_stage2 = _stage2_probabilities(protocol["stage2_held_out"], _choice_probabilities(baseline, tokenizer, protocol["stage2_held_out"], length, batch))
    finally:
        del baseline
        torch.cuda.empty_cache()
    cue, cue_tokenizer, cue_details, cue_attestation = _run_condition(config, protocol, seed=seed, homogenized=False, switch_training=False, adapter_dir=output_dir / "baseline_cue_only_adapter")
    try:
        cue_switch = _switch(cue, cue_tokenizer, held_switch, config)
        cue_erased = _switch(cue, cue_tokenizer, held_switch, config, direction)
        cue_erasure_controls = {name: _switch(cue, cue_tokenizer, held_switch, config, control) for name, control in controls.items()}
        cue_steering = _steering(cue, cue_tokenizer, monitored, config, direction)
        cue_steering_controls = {name: _steering(cue, cue_tokenizer, monitored, config, control) for name, control in controls.items()}
    finally:
        del cue
        torch.cuda.empty_cache()
    switch_values, erased_values = corrected_intervention_delta(baseline_switch, cue_switch, baseline_erased, cue_erased)
    switch_gap, switch_lower = _bootstrap_mean(switch_values, seed=_seed_from(seed, "g1-switch"), replicates=rep)
    steering_values = baseline_steering - cue_steering
    steering, steering_lower = _bootstrap_mean(steering_values, seed=_seed_from(seed, "g1-steering"), replicates=rep)
    erasure, erasure_lower = _bootstrap_relative_reduction(switch_values, erased_values, seed=_seed_from(seed, "g1-erasure"), replicates=rep)
    erasure_controls, steering_controls = {}, {}
    for name in controls:
        _source, altered = corrected_intervention_delta(baseline_switch, cue_switch, baseline_erasure_controls[name], cue_erasure_controls[name])
        erasure_controls[name], _unused = _bootstrap_relative_reduction(switch_values, altered, seed=_seed_from(seed, "g1-erasure-control", name), replicates=rep)
        steering_controls[name] = float((baseline_steering_controls[name] - cue_steering_controls[name]).mean())
    homogenized, homog_tokenizer, homog_details, homog_attestation = _run_condition(config, protocol, seed=seed, homogenized=True, switch_training=True, adapter_dir=output_dir / "homogenized_adapter")
    try:
        homog_switch = _switch(homogenized, homog_tokenizer, held_switch, config)
        homog_hidden = _hidden_by_layer(homogenized, homog_tokenizer, held, length, batch)[direction.layer]
        from sklearn.metrics import roc_auc_score
        homog_auc = float(roc_auc_score(np.asarray(labels_held, dtype=int), homog_hidden @ direction.values))
        homog_stage2 = _stage2_probabilities(protocol["stage2_held_out"], _choice_probabilities(homogenized, homog_tokenizer, protocol["stage2_held_out"], length, batch))
    finally:
        del homogenized
        torch.cuda.empty_cache()
    homog_cue, homog_cue_tokenizer, homog_cue_details, homog_cue_attestation = _run_condition(config, protocol, seed=seed, homogenized=True, switch_training=False, adapter_dir=output_dir / "homogenized_cue_only_adapter")
    try:
        homog_cue_switch = _switch(homog_cue, homog_cue_tokenizer, held_switch, config)
    finally:
        del homog_cue
        torch.cuda.empty_cache()
    homog_values = homog_switch - homog_cue_switch
    homog_reduction = -1.0 if switch_gap <= 0 else 1.0 - float(homog_values.mean()) / switch_gap
    readout_signal = readout_auc - 0.5
    readout_reduction = -1.0 if readout_signal <= 0 else 1.0 - (homog_auc - 0.5) / readout_signal
    metrics = {"seed": seed, "readout_auc": readout_auc, "readout_lower_ci": readout_lower, "switch_gap": switch_gap, "switch_lower_ci": switch_lower, "steering_contrast": steering, "steering_lower_ci": steering_lower, "control_effects": steering_controls, "erasure_relative_reduction": erasure, "erasure_lower_ci": erasure_lower, "erasure_control_reductions": erasure_controls, "homogenization_relative_reduction": homog_reduction, "homogenization_readout_relative_reduction": readout_reduction, "stage2_accuracy_loss": max(0.0, float(baseline_stage2.mean() - homog_stage2.mean()))}
    write_json(output_dir / "evidence.json", {"kind": G1_KIND, "config_sha256": config["_sha256"], "seed": seed, "metrics": metrics, "selected_layer": direction.layer, "readout_selection": diagnostics, "g1_cue_matched_interventions": True, "training": {"baseline": baseline_details, "baseline_cue_only": cue_details, "homogenized": homog_details, "homogenized_cue_only": homog_cue_details}, "runtime_attestations": {"baseline": baseline_attestation, "baseline_cue_only": cue_attestation, "homogenized": homog_attestation, "homogenized_cue_only": homog_cue_attestation}, "measurement": "forced_sequence_likelihood_over_ALPHA_BETA"})
    return metrics


def run_g1(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    config, destination = load_config(config_path), Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite G1 evidence: {destination}")
    preflight = os.environ.get("RGA_RUNTIME_PREFLIGHT")
    if not preflight or not Path(preflight).is_file():
        raise FileNotFoundError("G1 requires a completed immutable runtime preflight")
    preflight_value = json.loads(Path(preflight).read_text(encoding="utf-8"))
    if not isinstance(preflight_value, Mapping) or preflight_value.get("kind") != "recency_gated_alignment_g1_runtime_preflight" or preflight_value.get("config_sha256") != config["_sha256"]:
        raise ValueError("G1 runtime preflight is not bound to the frozen G1 configuration")
    destination.mkdir(parents=True)
    copied_preflight = destination / "runtime_preflight.json"
    shutil.copy2(preflight, copied_preflight)
    if sha256_file(copied_preflight) != sha256_file(preflight):
        raise RuntimeError("G1 runtime preflight copy checksum mismatch")
    corpus = build_corpus(config, destination / "corpus")
    protocol = protocol_records(read_jsonl(corpus["corpus"]))
    write_jsonl(destination / "protocol.jsonl", [{"partition": partition, **record} for partition, records in protocol.items() for record in records])
    manifest = {"kind": G1_KIND, "config_sha256": config["_sha256"], "corpus_sha256": corpus["corpus_sha256"], "protocol_sha256": sha256_file(destination / "protocol.jsonl"), "git_head": os.popen("git rev-parse HEAD").read().strip(), "runtime_preflight_filename": copied_preflight.name, "runtime_preflight_sha256": sha256_file(copied_preflight), "started_unix": time.time()}
    write_json(destination / "run_manifest.json", manifest)
    metrics = [_seed_run_g1(config, protocol, int(seed), destination / f"seed_{seed}") for seed in config["design"]["seeds"]]
    write_json(destination / "metrics.json", {"records": metrics})
    report = analyze_gate(config, metrics, destination / "gate_report.json")
    manifest.update({"completed_unix": time.time(), "metrics_sha256": sha256_file(destination / "metrics.json"), "gate_report_sha256": sha256_file(destination / "gate_report.json")})
    write_json(destination / "run_manifest.json", manifest)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run preregistered matched-cue recency G1")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = run_g1(args.config, args.output)
    print(canonical_json(report))
    return 0 if report["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
