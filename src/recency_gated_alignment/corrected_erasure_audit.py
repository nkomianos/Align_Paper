"""Post-run difference-in-differences audit for immutable recency G0 evidence.

This module never changes a G0 run root.  It reloads only its checksum-recorded
adapters and recomputes the necessity quantity with the cue-only control under
the *same* erasure.  The original runner compared an erased baseline with an
unerased cue-only control, which cannot distinguish a learned-policy effect
from a generic effect of erasure on the two runtime headers.

The audit is descriptive and cannot retroactively make G0 preregistered.  Even
if the corrected quantity is large, it only informs the design of a fresh G1.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from under_extinction.io import canonical_json, read_jsonl, sha256_file, write_json
from under_extinction.modeling import load_adapter_model, load_tokenizer

from .gate import CONTROLS, load_config
from .runner import (
    _bootstrap_relative_reduction,
    _choice_probabilities,
    _choose_direction,
    _matched_controls,
    _seed_from,
    _switch_pairs,
    protocol_records,
)
from .verify import verify_retrieved_run


AUDIT_KIND = "recency_g0_corrected_erasure_difference_in_differences"


def _read(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _artifact_path(seed_root: Path, recorded_path: object) -> Path:
    """Resolve either a remote absolute or retrieved-local adapter path."""

    candidate = Path(str(recorded_path))
    if candidate.is_dir():
        return candidate.resolve()
    try:
        index = candidate.parts.index(seed_root.name)
    except ValueError as exc:
        raise ValueError("Recorded adapter path is outside the source seed root") from exc
    resolved = seed_root.joinpath(*candidate.parts[index + 1:])
    if not resolved.is_dir():
        raise FileNotFoundError(f"Recorded adapter is unavailable: {resolved}")
    return resolved


def corrected_switch_delta(
    baseline: Sequence[float], cue_only: Sequence[float],
    baseline_erased: Sequence[float], cue_only_erased: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return baseline and erased contextual effects with matched cue controls."""

    arrays = [np.asarray(value, dtype=float) for value in (baseline, cue_only, baseline_erased, cue_only_erased)]
    if not arrays[0].size or any(item.shape != arrays[0].shape or not np.isfinite(item).all() for item in arrays):
        raise ValueError("Corrected erasure requires equal, finite, non-empty paired switch arrays")
    return arrays[0] - arrays[1], arrays[2] - arrays[3]


def _switch_values(model: Any, tokenizer: Any, records: list[Mapping[str, Any]], config: Mapping[str, Any], direction: Any | None = None) -> np.ndarray:
    probabilities = _choice_probabilities(
        model, tokenizer, records, int(config["model"]["max_length"]), int(config["training"]["batch_size"]),
        direction, erase=direction is not None,
    )
    return _switch_pairs(records, probabilities)


def _audit_seed(config: Mapping[str, Any], protocol: Mapping[str, list[Mapping[str, Any]]], source_root: Path, seed: int) -> dict[str, Any]:
    import torch

    seed_root = source_root / f"seed_{seed}"
    evidence = _read(seed_root / "evidence.json", f"seed {seed} evidence")
    training = evidence.get("training")
    if not isinstance(training, Mapping):
        raise ValueError(f"Seed {seed} has no training artifact map")
    baseline_detail, cue_detail = training.get("baseline"), training.get("baseline_cue_only")
    if not isinstance(baseline_detail, Mapping) or not isinstance(cue_detail, Mapping):
        raise ValueError(f"Seed {seed} lacks baseline or cue-only adapter details")
    baseline_path = _artifact_path(seed_root, dict(baseline_detail.get("adapter") or {}).get("path"))
    cue_path = _artifact_path(seed_root, dict(cue_detail.get("adapter") or {}).get("path"))
    tokenizer = load_tokenizer(dict(config))
    timestamp_train = [row for row in protocol["timestamp"] if row["probe_split"] == "train"]
    timestamp_held = [row for row in protocol["timestamp"] if row["probe_split"] == "held_out"]
    labels_train = [int(row["stage"]) - 1 for row in timestamp_train]
    labels_held = [int(row["stage"]) - 1 for row in timestamp_held]
    held_switch = list(protocol["switch_held_out"])
    baseline_model = load_adapter_model(dict(config), baseline_path)
    try:
        from .runner import _hidden_by_layer

        hidden_train = _hidden_by_layer(baseline_model, tokenizer, timestamp_train, int(config["model"]["max_length"]), int(config["training"]["batch_size"]))
        hidden_held = _hidden_by_layer(baseline_model, tokenizer, timestamp_held, int(config["model"]["max_length"]), int(config["training"]["batch_size"]))
        direction, readout_auc, readout_lower_ci, _diagnostics = _choose_direction(
            hidden_train, labels_train, hidden_held, labels_held,
            seed=seed, replicates=int(config["analysis"]["bootstrap_replicates"]),
        )
        controls = _matched_controls(hidden_train[direction.layer], labels_train, direction, seed=seed)
        baseline_switch = _switch_values(baseline_model, tokenizer, held_switch, config)
        baseline_erased = _switch_values(baseline_model, tokenizer, held_switch, config, direction)
        baseline_controls = {name: _switch_values(baseline_model, tokenizer, held_switch, config, control) for name, control in controls.items()}
    finally:
        del baseline_model
        torch.cuda.empty_cache()
    cue_model = load_adapter_model(dict(config), cue_path)
    try:
        cue_switch = _switch_values(cue_model, tokenizer, held_switch, config)
        cue_erased = _switch_values(cue_model, tokenizer, held_switch, config, direction)
        cue_controls = {name: _switch_values(cue_model, tokenizer, held_switch, config, control) for name, control in controls.items()}
    finally:
        del cue_model
        torch.cuda.empty_cache()
    source, altered = corrected_switch_delta(baseline_switch, cue_switch, baseline_erased, cue_erased)
    reduction, lower = _bootstrap_relative_reduction(
        source, altered, seed=_seed_from(seed, "corrected-erasure"),
        replicates=int(config["analysis"]["bootstrap_replicates"]),
    )
    control_reductions: dict[str, float] = {}
    for name in CONTROLS:
        _source, control_altered = corrected_switch_delta(baseline_switch, cue_switch, baseline_controls[name], cue_controls[name])
        control_reductions[name], _ignored_lower = _bootstrap_relative_reduction(
            source, control_altered, seed=_seed_from(seed, "corrected-erasure-control", name),
            replicates=int(config["analysis"]["bootstrap_replicates"]),
        )
    thresholds = config["thresholds"]
    passes_original_necessity_threshold = (
        reduction >= float(thresholds["minimum_erasure_relative_reduction"])
        and lower >= float(thresholds["minimum_erasure_lower_ci"])
        and max(abs(value) for value in control_reductions.values()) <= float(thresholds["maximum_erasure_control_fraction"]) * abs(reduction)
    )
    return {
        "seed": seed,
        "reconstructed_selected_layer": direction.layer,
        "reconstructed_readout_auc": readout_auc,
        "reconstructed_readout_lower_ci": readout_lower_ci,
        "baseline_switch_mean": float(baseline_switch.mean()),
        "cue_only_switch_mean": float(cue_switch.mean()),
        "difference_in_differences_switch_mean": float(source.mean()),
        "corrected_erased_switch_mean": float(altered.mean()),
        "corrected_erasure_relative_reduction": reduction,
        "corrected_erasure_lower_ci": lower,
        "corrected_erasure_control_reductions": control_reductions,
        "passes_original_necessity_threshold_descriptively": passes_original_necessity_threshold,
        "baseline_adapter_tree_sha256": dict(baseline_detail.get("adapter") or {}).get("tree_sha256"),
        "cue_only_adapter_tree_sha256": dict(cue_detail.get("adapter") or {}).get("tree_sha256"),
    }


def audit_completed_run(config_path: str | Path, run_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    config, source, destination = load_config(config_path), Path(run_root).resolve(), Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite corrected-audit output: {destination}")
    destination.mkdir(parents=True)
    try:
        source_verification = verify_retrieved_run(config["_path"], source, destination / "source_retrieval_verification.json")
        protocol = protocol_records(read_jsonl(source / "protocol.jsonl"))
        records = [_audit_seed(config, protocol, source, int(seed)) for seed in config["design"]["seeds"]]
        result = {
            "kind": AUDIT_KIND,
            "source_run_root": str(source),
            "source_run_manifest_sha256": sha256_file(source / "run_manifest.json"),
            "source_verification": source_verification,
            "config_sha256": config["_sha256"],
            "records": records,
            "decision": "NOT_A_RETROACTIVE_GATE__USE_ONLY_TO_PREREGISTER_A_FRESH_G1",
            "reason": "The original G0 erasure contrast did not apply erasure to the cue-only control.",
            "offline_mode": {"HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"), "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE")},
        }
        write_json(destination / "corrected_erasure_audit.json", result)
        return result
    except Exception:
        # Preserve any partial diagnostic files rather than mutating source evidence.
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit completed G0 erasure using a matched cue-only intervention")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = audit_completed_run(args.config, args.run_root, args.output)
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
