"""Analysis and continuation gates for the environment-grounded bridge."""

from __future__ import annotations

import json
import hashlib
import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .io import canonical_json, read_jsonl, sha256_file, write_json
from .bridge_evaluation import (
    BridgeEvaluationSpec,
    _generation_subset,
    configured_bridge_evaluation_spec_sha256,
    generation_subset_attestation,
)
from .bridge_training import BridgeTrainingSpec, configured_bridge_spec_sha256
from .modeling import (
    compact_model_runtime_contract,
    verify_model_runtime_attestation,
)


ARMS = ("genuine", "proxy")
AUDIT_FAMILIES = ("value", "transition")
VALUE_SWITCH_DIRECTIONS = ("devalue_preferred", "upvalue_nonpreferred")
ACQUISITION_CONDITIONS = ("aligned", "diagnostic_conflict")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _validate_policy_artifact(
    config: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    base_policy: bool,
    context: str,
) -> None:
    artifact = row.get("policy_artifact")
    required = {
        "kind",
        "base_model_id",
        "base_model_revision",
        "adapter_loaded",
        "loaded_adapter_file_sha256",
        "anchor_checkpoint",
        "anchor_checkpoint_manifest_sha256",
    }
    if not isinstance(artifact, Mapping) or set(artifact) != required:
        raise ValueError(f"Malformed policy-artifact attestation in {context}")
    checkpoint = row.get("checkpoint")
    checkpoint_manifest_sha256 = row.get("checkpoint_manifest_sha256")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError(f"Missing checkpoint identity in {context}")
    if not _is_sha256(checkpoint_manifest_sha256):
        raise ValueError(f"Invalid checkpoint-manifest hash in {context}")
    if (
        artifact.get("base_model_id") != str(config["model"]["id"])
        or artifact.get("base_model_revision") != str(config["model"]["revision"])
        or artifact.get("anchor_checkpoint") != checkpoint
        or artifact.get("anchor_checkpoint_manifest_sha256") != checkpoint_manifest_sha256
    ):
        raise ValueError(f"Policy artifact is not bound to its model/checkpoint in {context}")
    if base_policy:
        if (
            artifact.get("kind") != "unchanged_base_model"
            or artifact.get("adapter_loaded") is not False
            or artifact.get("loaded_adapter_file_sha256") is not None
        ):
            raise ValueError(f"Unchanged-base row does not attest an adapter-free policy in {context}")
        return
    raw_adapter_hashes = row.get("checkpoint_adapter_file_sha256")
    if (
        artifact.get("kind") != "lora_adapter_checkpoint"
        or artifact.get("adapter_loaded") is not True
        or artifact.get("loaded_adapter_file_sha256") != raw_adapter_hashes
    ):
        raise ValueError(f"Bridge row does not attest the loaded LoRA checkpoint in {context}")


def _validate_bridge_spec_provenance(
    config: Mapping[str, Any], row: Mapping[str, Any], *, context: str
) -> None:
    """Bind every analyzed row to the optimizer spec derived from this config."""
    expected_spec = json.loads(canonical_json(asdict(BridgeTrainingSpec.from_config(config))))
    if (
        row.get("bridge_spec") != expected_spec
        or row.get("bridge_spec_sha256") != configured_bridge_spec_sha256(config)
        or row.get("bridge_spec_source") != "loaded_config_exact"
    ):
        raise ValueError(f"Bridge training spec is not exactly config-bound in {context}")


def _validate_bridge_evaluation_spec_provenance(
    config: Mapping[str, Any], row: Mapping[str, Any], *, context: str
) -> None:
    expected_spec = json.loads(
        canonical_json(asdict(BridgeEvaluationSpec.from_config(config)))
    )
    if (
        row.get("bridge_evaluation_spec") != expected_spec
        or row.get("bridge_evaluation_spec_sha256")
        != configured_bridge_evaluation_spec_sha256(config)
        or row.get("bridge_evaluation_spec_source") != "loaded_config_exact"
    ):
        raise ValueError(f"Bridge evaluation spec is not exactly config-bound in {context}")


def _validate_model_runtime_contract(
    config: Mapping[str, Any], row: Mapping[str, Any], *, context: str
) -> dict[str, Any]:
    """Validate the compact per-row proof of the full checkpoint runtime attestation."""
    required = {
        "schema_version",
        "full_attestation_sha256",
        "model_id",
        "model_revision",
        "loader_class",
        "model_type",
        "text_only",
        "text_parameter_count",
        "layer_type_counts",
        "chat_template_kwargs",
        "chat_template_kwargs_supported",
        "closed_reasoning_preamble_observed",
        "chat_template_sha256",
        "deltanet_kernel_policy",
        "deltanet_backend",
        "lora_per_target_module_count",
        "lora_module_count",
        "lora_trainable_parameter_count",
        "lora_inventory_sha256",
        "contract_sha256",
    }
    contract = row.get("model_runtime_contract")
    if not isinstance(contract, Mapping) or set(contract) != required:
        raise ValueError(f"Malformed compact model-runtime contract in {context}")
    checked = dict(contract)
    claimed_contract_hash = checked.pop("contract_sha256")
    expected_contract_hash = hashlib.sha256(
        canonical_json(checked).encode("utf-8")
    ).hexdigest()
    full_hash = contract.get("full_attestation_sha256")
    if (
        claimed_contract_hash != expected_contract_hash
        or not _is_sha256(full_hash)
        or row.get("model_runtime_attestation_sha256") != full_hash
    ):
        raise ValueError(f"Model-runtime contract hash mismatch in {context}")
    model = config["model"]
    training = config["training"]
    expected_values = {
        "schema_version": "1.0",
        "model_id": model["id"],
        "model_revision": model["revision"],
        "loader_class": model["loader_class"],
        "model_type": model["expected_model_type"],
        "text_only": model["text_only"],
        "text_parameter_count": int(model["expected_text_parameter_count"]),
        "layer_type_counts": dict(model["expected_layer_type_counts"]),
        "chat_template_kwargs": dict(model["chat_template_kwargs"]),
        "deltanet_kernel_policy": model["delta_net_kernel_policy"],
        "lora_per_target_module_count": dict(training["expected_lora_target_counts"]),
        "lora_module_count": int(training["expected_lora_module_count"]),
        "lora_trainable_parameter_count": int(
            training["expected_lora_trainable_parameter_count"]
        ),
    }
    for key, expected in expected_values.items():
        if contract.get(key) != expected:
            raise ValueError(f"Model-runtime {key} differs from config in {context}")
    if (
        contract.get("chat_template_kwargs_supported") is not True
        or contract.get("closed_reasoning_preamble_observed") is not True
        or not _is_sha256(contract.get("chat_template_sha256"))
        or not _is_sha256(contract.get("lora_inventory_sha256"))
    ):
        raise ValueError(f"Model-runtime prompt/LoRA proof is incomplete in {context}")
    backend = contract.get("deltanet_backend")
    policy = model["delta_net_kernel_policy"]
    allowed_backends = {"torch_fallback", "accelerated_fla_causal_conv1d"}
    if (
        backend not in allowed_backends
        or (policy == "torch_fallback_required" and backend != "torch_fallback")
        or (policy == "fast_required" and backend != "accelerated_fla_causal_conv1d")
    ):
        raise ValueError(f"Unapproved DeltaNet backend in {context}: {backend!r}")
    return json.loads(canonical_json(dict(contract)))


def _validate_generation_subset_group(
    config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], *, context: str
) -> None:
    """Reconstruct the exact deterministic free-generation subset from JSONL rows."""
    if not rows:
        raise ValueError(f"Empty generation-subset group in {context}")
    requested = int(BridgeEvaluationSpec.from_config(config).generation_subset_size)
    selected = list(_generation_subset(rows, requested))
    selected_ids = {str(row["case_id"]) for row in selected}
    expected_attestation = generation_subset_attestation(
        rows, selected, requested_size=requested
    )
    for row in rows:
        case_id = str(row.get("case_id", ""))
        observed_selected = row.get("generation_subset_selected")
        if type(observed_selected) is not bool or observed_selected != (
            case_id in selected_ids
        ):
            raise ValueError(f"Generation-subset selection flag mismatch in {context}")
        if row.get("generation_subset_attestation") != expected_attestation:
            raise ValueError(f"Generation-subset attestation mismatch in {context}")
        if observed_selected:
            if (
                row.get("parse_status") == "not_sampled"
                or not isinstance(row.get("generated_output"), str)
            ):
                raise ValueError(f"Selected generation case was not generated in {context}")
        elif (
            row.get("parse_status") != "not_sampled"
            or row.get("generated_output") is not None
            or row.get("parsed_action") is not None
        ):
            raise ValueError(f"Nonselected generation case claims an output in {context}")


def _mean(values: Iterable[float | bool]) -> float:
    collected = list(values)
    return float(np.mean(collected)) if collected else float("nan")


def _probability(row: Mapping[str, Any], action: str) -> float:
    if action == "A":
        return float(row["probability_A"])
    if action == "B":
        return float(row["probability_B"])
    raise ValueError(f"Invalid action {action!r}")


def _validate_acquisition_summary(
    config: Mapping[str, Any], summary: Any, *, arm: str, checkpoint: int
) -> dict[str, Any]:
    if not isinstance(summary, Mapping):
        raise ValueError("Prediction lacks checkpoint acquisition diagnostics")
    cues = list(config["bridge"]["cue_regimes"])
    if (
        summary.get("schema_version") != "1.0"
        or summary.get("diagnostic_scope") != "cumulative_all_optimizer_updates"
        or summary.get("optimized_arm") != arm
        or summary.get("cue_regimes") != cues
        or summary.get("success_definition") != "optimized_realized_reward_equals_1"
    ):
        raise ValueError("Acquisition diagnostics identity mismatch")
    cells = summary.get("cells")
    if not isinstance(cells, Mapping) or set(cells) != set(cues):
        raise ValueError("Acquisition diagnostics have the wrong cue cells")

    def validate_cell(value: Any) -> dict[str, Any]:
        required = {
            "sample_count", "optimized_reward_sum", "optimal_action_count",
            "optimized_reward_mean", "optimal_action_accuracy",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("Malformed acquisition diagnostic summary cell")
        count = value["sample_count"]
        successes = value["optimal_action_count"]
        reward_sum = value["optimized_reward_sum"]
        if type(count) is not int or type(successes) is not int or not 0 <= successes <= count:
            raise ValueError("Invalid acquisition diagnostic counts")
        if not isinstance(reward_sum, (int, float)) or not math.isfinite(float(reward_sum)):
            raise ValueError("Invalid acquisition diagnostic reward sum")
        if not math.isclose(float(reward_sum), float(successes), abs_tol=1e-8):
            raise ValueError("Binary acquisition reward and success count disagree")
        expected_mean = float(successes) / count if count else None
        for key in ("optimized_reward_mean", "optimal_action_accuracy"):
            observed = value[key]
            if expected_mean is None:
                if observed is not None:
                    raise ValueError("Empty acquisition cell claims a mean")
            elif observed is None or not math.isclose(float(observed), expected_mean, abs_tol=1e-12):
                raise ValueError("Acquisition diagnostic mean is inconsistent")
        return {
            "sample_count": count,
            "optimized_reward_sum": float(reward_sum),
            "optimal_action_count": successes,
            "optimized_reward_mean": expected_mean,
            "optimal_action_accuracy": expected_mean,
        }

    checked_cells: dict[str, dict[str, dict[str, Any]]] = {}
    for cue in cues:
        raw_conditions = cells[cue]
        if not isinstance(raw_conditions, Mapping) or set(raw_conditions) != set(ACQUISITION_CONDITIONS):
            raise ValueError("Acquisition diagnostics have the wrong condition cells")
        checked_cells[cue] = {
            condition: validate_cell(raw_conditions[condition])
            for condition in ACQUISITION_CONDITIONS
        }

    def aggregate(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        count = sum(int(value["sample_count"]) for value in values)
        successes = sum(int(value["optimal_action_count"]) for value in values)
        mean = successes / count if count else None
        return {
            "sample_count": count,
            "optimized_reward_sum": float(successes),
            "optimal_action_count": successes,
            "optimized_reward_mean": mean,
            "optimal_action_accuracy": mean,
        }

    expected = {
        "schema_version": "1.0",
        "diagnostic_scope": "cumulative_all_optimizer_updates",
        "optimized_arm": arm,
        "cue_regimes": cues,
        "success_definition": "optimized_realized_reward_equals_1",
        "cells": checked_cells,
        "by_cue_regime": {
            cue: aggregate(list(checked_cells[cue].values())) for cue in cues
        },
        "by_condition": {
            condition: aggregate([checked_cells[cue][condition] for cue in cues])
            for condition in ACQUISITION_CONDITIONS
        },
        "overall": aggregate([
            checked_cells[cue][condition]
            for cue in cues for condition in ACQUISITION_CONDITIONS
        ]),
    }
    if canonical_json(dict(summary)) != canonical_json(expected):
        raise ValueError("Acquisition diagnostic aggregates are inconsistent")
    expected_samples = checkpoint * int(config["training"]["rollout_batch_size"])
    if expected["overall"]["sample_count"] != expected_samples:
        raise ValueError("Acquisition diagnostics do not cover every completed rollout")
    return expected


def _validate_acquisition_gate_window_summary(
    config: Mapping[str, Any], summary: Any, *, arm: str, checkpoint: int
) -> dict[str, Any]:
    """Validate the exact preregistered trailing window and all derived aggregates."""
    if not isinstance(summary, Mapping):
        raise ValueError("Prediction lacks checkpoint acquisition gate window diagnostics")
    window_updates = int(config["training"]["acquisition_gate_window_updates"])
    samples_per_update = int(config["training"]["rollout_batch_size"])
    covered_count = min(checkpoint, window_updates)
    first_update = checkpoint - covered_count + 1 if covered_count else None
    last_update = checkpoint if covered_count else None
    header = {
        "schema_version": "1.0",
        "diagnostic_scope": "trailing_optimizer_updates",
        "optimized_arm": arm,
        "cue_regimes": list(config["bridge"]["cue_regimes"]),
        "success_definition": "optimized_realized_reward_equals_1",
        "window_updates": window_updates,
        "samples_per_update": samples_per_update,
        "completed_updates": checkpoint,
        "covered_updates": {
            "first_completed_update": first_update,
            "last_completed_update": last_update,
            "update_count": covered_count,
        },
    }
    if any(summary.get(key) != value for key, value in header.items()):
        raise ValueError("Acquisition gate window identity or coverage mismatch")
    # Reuse the strict cumulative cell/aggregate validator on exactly the retained
    # updates.  Its checkpoint argument becomes the number of covered updates, so
    # the sample-count proof remains exact rather than merely bounded.
    cumulative_view = {
        "schema_version": "1.0",
        "diagnostic_scope": "cumulative_all_optimizer_updates",
        "optimized_arm": arm,
        "cue_regimes": header["cue_regimes"],
        "success_definition": header["success_definition"],
        "cells": summary.get("cells"),
        "by_cue_regime": summary.get("by_cue_regime"),
        "by_condition": summary.get("by_condition"),
        "overall": summary.get("overall"),
    }
    validated_rates = _validate_acquisition_summary(
        config,
        cumulative_view,
        arm=arm,
        checkpoint=covered_count,
    )
    expected = {
        **header,
        "cells": validated_rates["cells"],
        "by_cue_regime": validated_rates["by_cue_regime"],
        "by_condition": validated_rates["by_condition"],
        "overall": validated_rates["overall"],
    }
    if canonical_json(dict(summary)) != canonical_json(expected):
        raise ValueError("Acquisition gate window aggregates are inconsistent")
    return expected


def _cell_max_mean(rows: Sequence[dict[str, Any]], value_key: str, cell_keys: tuple[str, ...]) -> float:
    cells: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        cells[tuple(str(row.get(key)) for key in cell_keys)].append(abs(float(row[value_key])))
    return max((_mean(values) for values in cells.values()), default=float("nan"))


def _load_attested_prediction_file(
    config: Mapping[str, Any], path: Path
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any], dict[str, Any]]:
    """Load one JSONL only through its hash-bound full-runtime sidecar."""
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [dict(row) for row in read_jsonl(path)]
    if not rows:
        raise ValueError(f"Bridge prediction input is empty: {path}")
    summary_path = path.with_suffix(".summary.json")
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Prediction runtime sidecar is required for formal evidence: {summary_path}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, Mapping):
        raise ValueError(f"Malformed prediction summary: {summary_path}")
    prediction_hash = sha256_file(path)
    recorded_path = summary.get("predictions_path")
    if (
        summary.get("config_sha256") != config["_config_sha256"]
        or summary.get("predictions_sha256") != prediction_hash
        or not isinstance(recorded_path, str)
        or Path(recorded_path).name != path.name
        or int(summary.get("record_count", -1)) != len(rows)
    ):
        raise ValueError(f"Prediction summary is not bound to {path}")
    row_arms = {str(row.get("arm")) for row in rows}
    row_seeds = {int(row.get("pair_seed", -1)) for row in rows}
    row_splits = {str(row.get("split")) for row in rows}
    row_checkpoints = sorted({int(row.get("checkpoint_update", -1)) for row in rows})
    if (
        len(row_arms) != 1
        or len(row_seeds) != 1
        or len(row_splits) != 1
        or summary.get("arm") != next(iter(row_arms))
        or int(summary.get("pair_seed", -1)) != next(iter(row_seeds))
        or summary.get("split") != next(iter(row_splits))
    ):
        raise ValueError(f"Prediction summary arm/seed/split mismatch for {path}")
    kind = summary.get("kind")
    if kind == "bridge_fixed_checkpoint_series":
        if summary.get("checkpoint_updates") != row_checkpoints:
            raise ValueError(f"Prediction summary checkpoint series mismatch for {path}")
    elif kind == "bridge_extinction_evaluation":
        if row_checkpoints != [int(summary.get("checkpoint_update", -1))]:
            raise ValueError(f"Prediction summary checkpoint mismatch for {path}")
    else:
        raise ValueError(f"Unknown prediction summary kind for {path}: {kind!r}")
    raw_attestation = summary.get("model_runtime_attestation")
    if not isinstance(raw_attestation, Mapping):
        raise ValueError(f"Prediction summary lacks a full runtime attestation: {path}")
    full_attestation = verify_model_runtime_attestation(config, raw_attestation)
    full_hash = full_attestation["attestation_sha256"]
    compact = compact_model_runtime_contract(config, full_attestation)
    if (
        summary.get("model_runtime_attestation_sha256") != full_hash
        or summary.get("model_runtime_contract") != compact
    ):
        raise ValueError(f"Prediction summary runtime proof mismatch for {path}")
    for row in rows:
        if (
            row.get("model_runtime_attestation_sha256") != full_hash
            or row.get("model_runtime_contract") != compact
        ):
            raise ValueError(f"Prediction row is not bound to its runtime sidecar: {path}")
    hashes = {
        str(path): prediction_hash,
        str(summary_path): sha256_file(summary_path),
    }
    return rows, hashes, full_attestation, compact


def load_bridge_predictions(
    config: Mapping[str, Any], paths: Sequence[str | Path]
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any], dict[str, Any]]:
    resolved = [Path(path).resolve() for path in paths]
    if not resolved:
        raise ValueError("No bridge prediction inputs were supplied")
    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    attestations: list[dict[str, Any]] = []
    contracts: list[dict[str, Any]] = []
    for path in resolved:
        file_rows, file_hashes, attestation, contract = _load_attested_prediction_file(
            config, path
        )
        rows.extend(file_rows)
        hashes.update(file_hashes)
        attestations.append(attestation)
        contracts.append(contract)
    if not rows:
        raise ValueError("Bridge prediction inputs are empty")
    if any(value != attestations[0] for value in attestations[1:]):
        raise ValueError("Prediction inputs were produced under different model runtimes")
    if any(value != contracts[0] for value in contracts[1:]):
        raise ValueError("Prediction inputs carry different compact model contracts")
    return rows, hashes, attestations[0], contracts[0]


def _validate_unchanged_base_control(
    config: Mapping[str, Any],
    path: str | Path | None,
    primary_rows: Sequence[dict[str, Any]],
    *,
    split: str,
    expected_runtime_attestation: Mapping[str, Any] | None = None,
    require_runtime_sidecar: bool = False,
) -> tuple[dict[str, Any], dict[str, str]]:
    if path is None:
        return {"available": False, "pass": False, "reason": "not supplied"}, {}
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if require_runtime_sidecar:
        rows, hashes, full_runtime, _ = _load_attested_prediction_file(config, source)
        if (
            expected_runtime_attestation is None
            or full_runtime != dict(expected_runtime_attestation)
        ):
            raise ValueError("Unchanged-base control used a different model runtime")
    else:
        rows = [dict(row) for row in read_jsonl(source)]
        hashes = {str(source): sha256_file(source)}
    if not rows:
        raise ValueError("Unchanged-base control is empty")
    if split != config["bridge"]["splits"]["development"]:
        raise ValueError("The unchanged-base negative control is development-only")
    seeds = {int(row.get("pair_seed", -1)) for row in rows}
    if len(seeds) != 1:
        raise ValueError("Unchanged-base control must contain exactly one paired seed")
    seed = next(iter(seeds))
    reference = {
        str(row["case_id"]): row
        for row in primary_rows
        if row["arm"] == "genuine"
        and int(row["pair_seed"]) == seed
        and int(row["checkpoint_update"]) == 0
    }
    if not reference or {str(row.get("case_id", "")) for row in rows} != set(reference):
        raise ValueError("Unchanged-base control does not match the checkpoint-zero case set")
    probability_deltas: list[float] = []
    legal_mass_deltas: list[float] = []
    disagreements: list[float] = []
    seen_base_cases: set[str] = set()
    for row in rows:
        case_id = str(row["case_id"])
        if not case_id or case_id in seen_base_cases:
            raise ValueError(f"Duplicate or empty unchanged-base case ID: {case_id!r}")
        seen_base_cases.add(case_id)
        paired = reference[case_id]
        if (
            row.get("arm") != "base"
            or row.get("policy_condition") != "unchanged_base"
            or int(row.get("checkpoint_update", -1)) != 0
            or row.get("evidence_kind") != "environment_grounded_bridge"
            or row.get("config_sha256") != config["_config_sha256"]
            or row.get("split") != split
        ):
            raise ValueError(f"Invalid unchanged-base provenance for {case_id}")
        _validate_policy_artifact(
            config, row, base_policy=True, context=f"unchanged-base case {case_id}"
        )
        _validate_bridge_spec_provenance(
            config, row, context=f"unchanged-base case {case_id}"
        )
        _validate_bridge_evaluation_spec_provenance(
            config, row, context=f"unchanged-base case {case_id}"
        )
        runtime_contract = _validate_model_runtime_contract(
            config, row, context=f"unchanged-base case {case_id}"
        )
        paired_runtime_contract = _validate_model_runtime_contract(
            config, paired, context=f"checkpoint-zero case {case_id}"
        )
        if runtime_contract != paired_runtime_contract:
            raise ValueError(f"Unchanged-base runtime differs for {case_id}")
        if row.get("messages_sha256") != paired.get("messages_sha256"):
            raise ValueError(f"Unchanged-base prompt differs for {case_id}")
        probability_a = float(row["probability_A"])
        probability_b = float(row["probability_B"])
        logp_a = float(row["logp_A"])
        logp_b = float(row["logp_B"])
        legal_mass = float(row["legal_choice_mass"])
        if not all(
            math.isfinite(value)
            for value in (probability_a, probability_b, logp_a, logp_b, legal_mass)
        ) or not math.isclose(probability_a + probability_b, 1.0, abs_tol=1e-7):
            raise ValueError(f"Invalid unchanged-base score for {case_id}")
        maximum = max(logp_a, logp_b)
        expected_a = math.exp(logp_a - maximum) / (
            math.exp(logp_a - maximum) + math.exp(logp_b - maximum)
        )
        expected_mass = math.exp(maximum) * (
            math.exp(logp_a - maximum) + math.exp(logp_b - maximum)
        )
        if not math.isclose(probability_a, expected_a, abs_tol=1e-6) or not math.isclose(
            legal_mass, expected_mass, rel_tol=1e-5, abs_tol=1e-8
        ):
            raise ValueError(f"Inconsistent unchanged-base likelihoods for {case_id}")
        probability_deltas.append(abs(float(row["probability_A"]) - float(paired["probability_A"])))
        legal_mass_deltas.append(abs(float(row["legal_choice_mass"]) - float(paired["legal_choice_mass"])))
        disagreements.append(float(row["predicted_action"] != paired["predicted_action"]))
    _validate_generation_subset_group(
        config, rows, context=f"unchanged-base seed {seed} split {split}"
    )
    maximum_probability_delta = max(probability_deltas)
    limit = float(config["gates"]["smoke"]["adapter_reload_max_probability_delta"])
    by_case = {str(row["case_id"]): row for row in rows}
    fingerprint_cells: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        intervention = row.get("intervention")
        if not isinstance(intervention, Mapping):
            raise ValueError("Unchanged-base row lacks intervention metadata")
        family = str(
            intervention.get("base_family", intervention.get("family", ""))
        )
        channel = str(
            intervention.get("objective", intervention.get("channel", ""))
        )
        if (
            intervention.get("active") is not True
            or intervention.get("mode") != "switch"
            or family not in AUDIT_FAMILIES
        ):
            continue
        paired_id = row.get("paired_control_id")
        if not isinstance(paired_id, str) or paired_id not in by_case:
            raise ValueError("Unchanged-base active case lacks its matched sham")
        control = by_case[paired_id]
        control_intervention = control.get("intervention")
        if (
            not isinstance(control_intervention, Mapping)
            or control_intervention.get("active") is not False
            or control_intervention.get("mode") != "sham"
        ):
            raise ValueError("Unchanged-base paired control is not a sham")
        if channel not in ARMS:
            raise ValueError("Unchanged-base intervention has an invalid reward channel")
        cue = str(row.get("cue_regime", ""))
        fingerprint_cells[(cue, family, channel)].append(
            _probability(control, str(control["pre_target_action"]))
            - _probability(row, str(row["pre_target_action"]))
        )
    expected_fingerprint_cells = {
        (cue, family, channel)
        for cue in config["bridge"]["cue_regimes"]
        for family in AUDIT_FAMILIES
        for channel in ARMS
    }
    fingerprint_complete = expected_fingerprint_cells <= set(fingerprint_cells)
    fingerprint = {
        cue: {
            family: {
                channel: _mean(fingerprint_cells.get((cue, family, channel), []))
                for channel in ARMS
            }
            for family in AUDIT_FAMILIES
        }
        for cue in config["bridge"]["cue_regimes"]
    }
    channel_selectivity_gaps = {
        cue: {
            family: abs(
                float(fingerprint[cue][family]["genuine"])
                - float(fingerprint[cue][family]["proxy"])
            )
            if all(
                math.isfinite(float(fingerprint[cue][family][channel]))
                for channel in ARMS
            )
            else float("nan")
            for family in AUDIT_FAMILIES
        }
        for cue in config["bridge"]["cue_regimes"]
    }
    maximum_selectivity_gap = _maximum_finite(
        channel_selectivity_gaps[cue][family]
        for cue in config["bridge"]["cue_regimes"]
        for family in AUDIT_FAMILIES
    )
    # Semantic channel names (for example, "beneficiary condition") can elicit
    # a pretrained preference before any reward learning.  That prior is useful
    # to report and is removed by the checkpoint-zero learning-induced contrast,
    # but it must not veto the experiment.  The neutral nonce-channel condition
    # is the hard negative control for an already-present causal fingerprint.
    hard_gate_cues = [
        cue for cue in config["bridge"]["cue_regimes"] if cue == "neutral"
    ]
    maximum_hard_gate_selectivity_gap = (
        _maximum_finite(
            channel_selectivity_gaps[cue][family]
            for cue in hard_gate_cues
            for family in AUDIT_FAMILIES
        )
        if hard_gate_cues
        else float("nan")
    )
    scientific_limit = float(
        config["gates"]["stage1"].get(
            "base_control_neutral_channel_selectivity_gap_max", 1.0
        )
    )
    integrity_pass = (
        maximum_probability_delta <= limit
        and max(legal_mass_deltas) <= limit
        and max(disagreements) == 0.0
    )
    scientific_fingerprint_pass = (
        fingerprint_complete
        and (
            not hard_gate_cues
            or (
                math.isfinite(maximum_hard_gate_selectivity_gap)
                and maximum_hard_gate_selectivity_gap <= scientific_limit
            )
        )
    )
    result = {
        "available": True,
        # The unchanged base is both an integrity control (checkpoint-zero LoRA
        # really is the base policy) and a scientific negative control.  A base
        # model that already shows a channel-selective fingerprint under neutral
        # channel identities cannot support an attribution of that same pattern
        # to reward learning. Semantic priors are descriptive and are subtracted
        # by the checkpoint-zero learning-induced estimand.
        "pass": integrity_pass and scientific_fingerprint_pass,
        "integrity_pass": integrity_pass,
        "fingerprint_complete": fingerprint_complete,
        "causal_fingerprint_by_cue_family_channel": fingerprint,
        "channel_selectivity_gap_by_cue_family": channel_selectivity_gaps,
        "maximum_channel_selectivity_gap": maximum_selectivity_gap,
        "hard_gate_cue_regimes": hard_gate_cues,
        "hard_gate_applicable": bool(hard_gate_cues),
        "maximum_hard_gate_channel_selectivity_gap": (
            maximum_hard_gate_selectivity_gap if hard_gate_cues else None
        ),
        "neutral_channel_selectivity_gap_limit": scientific_limit,
        "scientific_fingerprint_pass": scientific_fingerprint_pass,
        "seed": seed,
        "record_count": len(rows),
        "max_probability_A_delta_vs_untrained_lora": maximum_probability_delta,
        "max_legal_choice_mass_delta_vs_untrained_lora": max(legal_mass_deltas),
        "action_disagreement_rate_vs_untrained_lora": _mean(disagreements),
        "probability_tolerance": limit,
    }
    return result, hashes


def validate_bridge_predictions(
    config: Mapping[str, Any], rows: Sequence[dict[str, Any]], *, split: str
) -> dict[str, Any]:
    expected_config_hash = str(config["_config_sha256"])
    seen: set[tuple[str, int, int, str]] = set()
    provenance_strings: set[str] = set()
    run_cases: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    evaluation_groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    initial_state_hashes: dict[tuple[int, str], set[str]] = defaultdict(set)
    message_hashes: dict[tuple[int, str], set[str]] = defaultdict(set)
    adapter_hashes: dict[tuple[int, str, int], set[str]] = defaultdict(set)
    acquisition_summaries: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    acquisition_gate_window_summaries: dict[
        tuple[str, int, int], set[str]
    ] = defaultdict(set)
    model_runtime_contracts: set[str] = set()
    for row in rows:
        arm = str(row.get("arm"))
        seed = int(row.get("pair_seed", -1))
        checkpoint = int(row.get("checkpoint_update", -1))
        case_id = str(row.get("case_id", ""))
        key = (arm, seed, checkpoint, case_id)
        if arm not in ARMS:
            raise ValueError(f"Unknown bridge arm in predictions: {arm!r}")
        if seed not in config["bridge"]["seeds"]:
            raise ValueError(f"Unregistered bridge seed: {seed}")
        if checkpoint < 0 or not case_id:
            raise ValueError("Bridge prediction lacks checkpoint_update or case_id")
        if key in seen:
            raise ValueError(f"Duplicate bridge prediction: {key}")
        seen.add(key)
        run_cases[(arm, seed, checkpoint)].add(case_id)
        evaluation_groups[(arm, seed, checkpoint)].append(row)
        if row.get("evidence_kind") != "environment_grounded_bridge":
            raise ValueError(f"Invalid bridge evidence kind in {key}")
        if row.get("config_sha256") != expected_config_hash:
            raise ValueError(f"Bridge config hash mismatch in {key}")
        _validate_bridge_spec_provenance(config, row, context=str(key))
        _validate_bridge_evaluation_spec_provenance(config, row, context=str(key))
        model_runtime_contracts.add(
            canonical_json(_validate_model_runtime_contract(config, row, context=str(key)))
        )
        if row.get("split") != split:
            raise ValueError(f"Requested {split!r} but row {key} claims {row.get('split')!r}")
        if row.get("cue_regime") not in set(config["bridge"]["cue_regimes"]):
            raise ValueError(f"Invalid or unregistered cue regime in {key}")
        provenance = row.get("environment_provenance")
        if not isinstance(provenance, Mapping) or not provenance:
            raise ValueError(f"Missing environment provenance in {key}")
        provenance_strings.add(json.dumps(provenance, sort_keys=True, separators=(",", ":")))
        initial_state_hash = row.get("initial_environment_state_sha256")
        if not _is_sha256(initial_state_hash):
            raise ValueError(f"Missing initial environment-state hash in {key}")
        initial_state_hashes[(seed, arm)].add(initial_state_hash)
        messages_sha256 = row.get("messages_sha256")
        if not _is_sha256(messages_sha256):
            raise ValueError(f"Missing exact prompt hash in {key}")
        message_hashes[(seed, case_id)].add(messages_sha256)
        raw_adapter_hashes = row.get("checkpoint_adapter_file_sha256")
        if not isinstance(raw_adapter_hashes, Mapping) or not raw_adapter_hashes:
            raise ValueError(f"Missing adapter file hashes in {key}")
        if not all(
            isinstance(name, str) and bool(name) and _is_sha256(digest)
            for name, digest in raw_adapter_hashes.items()
        ):
            raise ValueError(f"Invalid adapter file hashes in {key}")
        _validate_policy_artifact(config, row, base_policy=False, context=str(key))
        adapter_hashes[(seed, arm, checkpoint)].add(
            json.dumps(raw_adapter_hashes, sort_keys=True, separators=(",", ":"))
        )
        acquisition_summary = _validate_acquisition_summary(
            config,
            row.get("checkpoint_acquisition_diagnostics"),
            arm=arm,
            checkpoint=checkpoint,
        )
        acquisition_summaries[(arm, seed, checkpoint)].add(canonical_json(acquisition_summary))
        acquisition_gate_window_summary = _validate_acquisition_gate_window_summary(
            config,
            row.get("checkpoint_acquisition_gate_window_diagnostics"),
            arm=arm,
            checkpoint=checkpoint,
        )
        acquisition_gate_window_summaries[(arm, seed, checkpoint)].add(
            canonical_json(acquisition_gate_window_summary)
        )
        for cue in config["bridge"]["cue_regimes"]:
            for condition in ACQUISITION_CONDITIONS:
                cumulative_cell = acquisition_summary["cells"][cue][condition]
                window_cell = acquisition_gate_window_summary["cells"][cue][condition]
                if (
                    int(window_cell["sample_count"])
                    > int(cumulative_cell["sample_count"])
                    or int(window_cell["optimal_action_count"])
                    > int(cumulative_cell["optimal_action_count"])
                ):
                    raise ValueError(
                        "Acquisition gate window cannot exceed its cumulative diagnostic cell"
                    )
        if checkpoint <= int(config["training"]["acquisition_gate_window_updates"]):
            cumulative_rates = {
                key: acquisition_summary[key]
                for key in ("cells", "by_cue_regime", "by_condition", "overall")
            }
            window_rates = {
                key: acquisition_gate_window_summary[key]
                for key in ("cells", "by_cue_regime", "by_condition", "overall")
            }
            if canonical_json(cumulative_rates) != canonical_json(window_rates):
                raise ValueError(
                    "Acquisition gate window must equal cumulative diagnostics before the window fills"
                )
        expected_actions = row.get("expected_actions")
        if not isinstance(expected_actions, Mapping) or not set(ARMS) <= set(expected_actions):
            raise ValueError(f"Expected-action mapping lacks the two reward arms in {key}")
        if not set(expected_actions.values()) <= {"A", "B"}:
            raise ValueError(f"Illegal expected action in {key}")
        pre_target = row.get("pre_target_action")
        if pre_target not in {"A", "B"}:
            raise ValueError(f"Missing pre-target action in {key}")
        intervention = row.get("intervention")
        if not isinstance(intervention, Mapping):
            raise ValueError(f"Missing intervention metadata in {key}")
        probability_a = float(row["probability_A"])
        probability_b = float(row["probability_B"])
        logp_a = float(row["logp_A"])
        logp_b = float(row["logp_B"])
        legal_mass = float(row["legal_choice_mass"])
        if not all(math.isfinite(value) for value in (probability_a, probability_b, logp_a, logp_b, legal_mass)):
            raise ValueError(f"Non-finite bridge score in {key}")
        if not (0 <= probability_a <= 1 and 0 <= probability_b <= 1 and 0 <= legal_mass <= 1):
            raise ValueError(f"Out-of-range bridge score in {key}")
        if not math.isclose(probability_a + probability_b, 1.0, abs_tol=1e-7):
            raise ValueError(f"Bridge probabilities do not normalize in {key}")
        maximum = max(logp_a, logp_b)
        expected_a = math.exp(logp_a - maximum) / (
            math.exp(logp_a - maximum) + math.exp(logp_b - maximum)
        )
        if not math.isclose(probability_a, expected_a, abs_tol=1e-6):
            raise ValueError(f"Bridge probability/log-likelihood mismatch in {key}")
        expected_mass = math.exp(maximum) * (
            math.exp(logp_a - maximum) + math.exp(logp_b - maximum)
        )
        if not math.isclose(legal_mass, expected_mass, rel_tol=1e-5, abs_tol=1e-8):
            raise ValueError(f"Bridge legal-choice mass mismatch in {key}")
        predicted = "A" if probability_a >= 0.5 else "B"
        if row.get("predicted_action") != predicted:
            raise ValueError(f"Bridge predicted action mismatch in {key}")
        reload_available = row.get("adapter_reload_probe_available")
        reload_delta = row.get("adapter_reload_max_probability_delta")
        reload_check = row.get("adapter_reload_probability_check")
        if checkpoint > 0:
            if reload_available is not True or reload_check is not True:
                raise ValueError(f"Trained checkpoint failed or lacks adapter-reload validation in {key}")
            if reload_delta is None or not math.isfinite(float(reload_delta)):
                raise ValueError(f"Invalid adapter-reload delta in {key}")
        elif reload_available is not False or reload_delta is not None or reload_check is not None:
            raise ValueError(f"Checkpoint zero must mark adapter-reload probe N/A in {key}")
        optimizer_metrics = row.get("checkpoint_optimizer_metrics")
        if checkpoint > 0:
            if not isinstance(optimizer_metrics, Mapping) or not optimizer_metrics:
                raise ValueError(f"Trained checkpoint lacks optimizer metrics in {key}")
            required_optimizer_metrics = {
                "loss", "policy_loss", "gradient_norm", "rollout_reward_mean", "learning_rate",
            }
            if not required_optimizer_metrics <= set(optimizer_metrics):
                raise ValueError(f"Trained checkpoint lacks required optimizer metrics in {key}")
            if not all(math.isfinite(float(value)) for value in optimizer_metrics.values()):
                raise ValueError(f"Non-finite optimizer metric in {key}")
        elif optimizer_metrics is not None:
            raise ValueError(f"Checkpoint zero must not claim optimizer metrics in {key}")
    for group, group_rows in evaluation_groups.items():
        _validate_generation_subset_group(config, group_rows, context=str(group))
    if len(provenance_strings) != 1:
        raise ValueError("Bridge inputs mix different environment/data provenance")
    if len(model_runtime_contracts) != 1:
        raise ValueError("Bridge inputs mix different model runtimes or LoRA contracts")
    reference_cases: set[str] | None = None
    for run, cases in run_cases.items():
        if reference_cases is None:
            reference_cases = cases
        elif cases != reference_cases:
            raise ValueError(f"Bridge checkpoint {run} does not contain the identical frozen case set")
    expected_checkpoints = set(BridgeTrainingSpec.from_config(config).checkpoint_updates)
    by_arm_seed: dict[tuple[str, int], set[int]] = defaultdict(set)
    for arm, seed, checkpoint in run_cases:
        by_arm_seed[(arm, seed)].add(checkpoint)
    for run, checkpoints in by_arm_seed.items():
        if checkpoints != expected_checkpoints:
            raise ValueError(
                f"Bridge run {run} has checkpoint set {sorted(checkpoints)}; "
                f"expected {sorted(expected_checkpoints)}"
            )
    for seed in sorted({item[0] for item in initial_state_hashes}):
        genuine = initial_state_hashes.get((seed, "genuine"), set())
        proxy = initial_state_hashes.get((seed, "proxy"), set())
        if len(genuine) != 1 or genuine != proxy:
            raise ValueError(f"G/P arms do not share an identical initial environment state for seed {seed}")
        genuine_adapter = adapter_hashes.get((seed, "genuine", 0), set())
        proxy_adapter = adapter_hashes.get((seed, "proxy", 0), set())
        if len(genuine_adapter) != 1 or genuine_adapter != proxy_adapter:
            raise ValueError(f"G/P arms do not share identical checkpoint-zero adapter files for seed {seed}")
    changed_prompts = [key for key, hashes in message_hashes.items() if len(hashes) != 1]
    if changed_prompts:
        raise ValueError(
            f"Bridge reused case IDs with different prompts across arms/checkpoints: {changed_prompts[:3]}"
        )
    inconsistent_diagnostics = [
        key for key, values in acquisition_summaries.items() if len(values) != 1
    ]
    if inconsistent_diagnostics:
        raise ValueError(
            f"Bridge rows disagree on checkpoint acquisition diagnostics: {inconsistent_diagnostics[:3]}"
        )
    inconsistent_gate_windows = [
        key for key, values in acquisition_gate_window_summaries.items() if len(values) != 1
    ]
    if inconsistent_gate_windows:
        raise ValueError(
            "Bridge rows disagree on checkpoint acquisition gate windows: "
            f"{inconsistent_gate_windows[:3]}"
        )
    return {
        "environment_provenance": json.loads(next(iter(provenance_strings))),
        "model_runtime_contract": json.loads(next(iter(model_runtime_contracts))),
        "paired_initial_environment_state": True,
        "case_count_per_checkpoint": len(reference_cases or ()),
        "run_checkpoint_count": len(run_cases),
    }


def _probability_by_key(rows: Sequence[dict[str, Any]]) -> dict[tuple[str, int, int, str], float]:
    return {
        (str(row["arm"]), int(row["pair_seed"]), int(row["checkpoint_update"]), str(row["case_id"])):
        _probability(row, str(row["pre_target_action"]))
        for row in rows
    }


def paired_effects(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    probabilities = _probability_by_key(rows)
    by_key = {
        (str(row["arm"]), int(row["pair_seed"]), int(row["checkpoint_update"]), str(row["case_id"])): row
        for row in rows
    }
    output: list[dict[str, Any]] = []
    for row in rows:
        intervention = row["intervention"]
        mode = str(intervention.get("mode", ""))
        # ``family`` names the full channel-specific manipulation (for example,
        # genuine_value); ``base_family`` is the value/transition factor used by
        # the factorial analysis.  Accept the short form for older fixtures.
        family = str(intervention.get("base_family", intervention.get("family", "")))
        if not bool(intervention.get("active")) or family not in AUDIT_FAMILIES or mode not in {"switch", "no_switch"}:
            continue
        prefix = (str(row["arm"]), int(row["pair_seed"]), int(row["checkpoint_update"]))
        paired_id = row.get("paired_control_id")
        baseline_id = row.get("baseline_id")
        if not isinstance(paired_id, str) or prefix + (paired_id,) not in by_key:
            raise ValueError(f"Active bridge case {row['case_id']} lacks its matched sham")
        if not isinstance(baseline_id, str) or prefix + (baseline_id,) not in by_key:
            raise ValueError(f"Active bridge case {row['case_id']} lacks its baseline")
        active_probability = probabilities[prefix + (str(row["case_id"]),)]
        control_probability = probabilities[prefix + (paired_id,)]
        baseline_probability = probabilities[prefix + (baseline_id,)]
        control_intervention = by_key[prefix + (paired_id,)]["intervention"]
        if bool(control_intervention.get("active")) or str(control_intervention.get("mode")) != "sham":
            raise ValueError(f"Paired control for {row['case_id']} is not an inactive sham")
        channel = str(intervention.get("objective", intervention.get("channel", "")))
        if channel not in ARMS:
            raise ValueError(f"Active bridge intervention has invalid channel: {channel!r}")
        value_update_type = str(intervention.get("value_update_type", ""))
        if family == "value":
            allowed_value_types = (
                set(VALUE_SWITCH_DIRECTIONS)
                if mode == "switch"
                else {"upvalue_preferred"}
            )
            if value_update_type not in allowed_value_types:
                raise ValueError(
                    f"Active value intervention has invalid update type: {value_update_type!r}"
                )
        elif value_update_type != "not_applicable":
            raise ValueError("Transition interventions must mark value_update_type not_applicable")
        if str(control_intervention.get("value_update_type", "")) != value_update_type:
            raise ValueError(
                f"Paired control for {row['case_id']} does not match value-update direction"
            )
        output.append({
            "arm": str(row["arm"]),
            "pair_seed": int(row["pair_seed"]),
            "checkpoint_update": int(row["checkpoint_update"]),
            "case_id": str(row["case_id"]),
            "world_id": str(row.get("world_id")),
            "pre_target_action": str(row["pre_target_action"]),
            "renderer_id": str(row.get("renderer_id")),
            "role_assignment": str(intervention.get("role_assignment", "unknown")),
            "cue_regime": str(row.get("cue_regime", "unknown")),
            "family": family,
            "channel": channel,
            "mode": mode,
            "value_update_type": value_update_type,
            "relevant": channel == str(row["arm"]),
            "shift": control_probability - active_probability,
            "sham_shift": baseline_probability - control_probability,
            "active_accuracy": row["predicted_action"] == row["expected_actions"][row["arm"]],
        })
    if not output:
        raise ValueError("Bridge predictions contain no analyzable active causal interventions")
    return output


def _checkpoint_metrics(
    config: Mapping[str, Any],
    rows: Sequence[dict[str, Any]],
    effects: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped_rows: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    grouped_effects: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_rows[(str(row["arm"]), int(row["pair_seed"]), int(row["checkpoint_update"]))].append(row)
    for row in effects:
        grouped_effects[(row["arm"], row["pair_seed"], row["checkpoint_update"])].append(row)
    output: list[dict[str, Any]] = []
    for key, run_rows in sorted(grouped_rows.items()):
        arm, seed, checkpoint = key
        run_effects = grouped_effects.get(key, [])
        ordinary = [row for row in run_rows if row.get("condition") == "ordinary"]
        comprehension = [
            row for row in run_rows
            if row.get("condition") in {"update_comprehension", "value_comprehension", "transition_comprehension"}
        ]
        generation = [row for row in run_rows if row.get("parse_status") != "not_sampled"]
        switch = [row for row in run_effects if row["mode"] == "switch"]
        no_switch = [row for row in run_effects if row["mode"] == "no_switch"]
        relevant = [row for row in switch if row["relevant"]]
        irrelevant = [row for row in switch if not row["relevant"]]
        family_relevant = {
            family: _mean(row["shift"] for row in relevant if row["family"] == family)
            for family in AUDIT_FAMILIES
        }
        active_accuracy_by_family = {
            family: _mean(
                row["active_accuracy"] for row in relevant if row["family"] == family
            )
            for family in AUDIT_FAMILIES
        }
        no_switch_accuracy_by_family = {
            family: _mean(
                row["active_accuracy"] for row in no_switch if row["family"] == family
            )
            for family in AUDIT_FAMILIES
        }
        comprehension_by_family = {
            family: _mean(
                row["predicted_action"] == row["expected_actions"][arm]
                for row in comprehension
                if row["intervention"].get(
                    "base_family", row["intervention"].get("family")
                ) == family
            )
            for family in AUDIT_FAMILIES
        }
        ordinary_accuracy_by_cue = {
            cue: _mean(
                row["predicted_action"] == row["expected_actions"][arm]
                for row in ordinary if row["cue_regime"] == cue
            )
            for cue in config["bridge"]["cue_regimes"]
        }
        comprehension_by_cue_family = {
            cue: {
                family: _mean(
                    row["predicted_action"] == row["expected_actions"][arm]
                    for row in comprehension
                    if row["cue_regime"] == cue
                    and row["intervention"].get(
                        "base_family", row["intervention"].get("family")
                    ) == family
                )
                for family in AUDIT_FAMILIES
            }
            for cue in config["bridge"]["cue_regimes"]
        }
        role_cells: dict[str, list[float]] = defaultdict(list)
        renderer_cells: dict[str, list[float]] = defaultdict(list)
        cue_cells: dict[str, list[float]] = defaultdict(list)
        for row in relevant:
            role_cells[row["role_assignment"]].append(row["shift"])
            renderer_cells[row["renderer_id"]].append(row["shift"])
            cue_cells[row["cue_regime"]].append(row["shift"])
        role_means = [_mean(values) for values in role_cells.values()]
        renderer_means = [_mean(values) for values in renderer_cells.values()]
        cue_means = [_mean(values) for values in cue_cells.values()]
        reference_effect = max(_mean(row["shift"] for row in relevant), 1e-12)
        relevant_by_cue_family = {
            cue: {
                family: _mean(
                    row["shift"] for row in relevant
                    if row["cue_regime"] == cue and row["family"] == family
                )
                for family in AUDIT_FAMILIES
            }
            for cue in config["bridge"]["cue_regimes"]
        }
        value_relevant_by_cue_direction = {
            cue: {
                direction: _mean(
                    row["shift"] for row in relevant
                    if row["cue_regime"] == cue
                    and row["family"] == "value"
                    and row["value_update_type"] == direction
                )
                for direction in VALUE_SWITCH_DIRECTIONS
            }
            for cue in config["bridge"]["cue_regimes"]
        }
        value_accuracy_by_cue_direction = {
            cue: {
                direction: _mean(
                    row["active_accuracy"] for row in relevant
                    if row["cue_regime"] == cue
                    and row["family"] == "value"
                    and row["value_update_type"] == direction
                )
                for direction in VALUE_SWITCH_DIRECTIONS
            }
            for cue in config["bridge"]["cue_regimes"]
        }
        value_count_by_cue_direction = {
            cue: {
                direction: sum(
                    row["cue_regime"] == cue
                    and row["family"] == "value"
                    and row["value_update_type"] == direction
                    for row in relevant
                )
                for direction in VALUE_SWITCH_DIRECTIONS
            }
            for cue in config["bridge"]["cue_regimes"]
        }
        value_role_count_by_cue_direction = {
            cue: {
                direction: len({
                    row["role_assignment"] for row in relevant
                    if row["cue_regime"] == cue
                    and row["family"] == "value"
                    and row["value_update_type"] == direction
                })
                for direction in VALUE_SWITCH_DIRECTIONS
            }
            for cue in config["bridge"]["cue_regimes"]
        }
        value_renderer_count_by_cue_direction = {
            cue: {
                direction: len({
                    row["renderer_id"] for row in relevant
                    if row["cue_regime"] == cue
                    and row["family"] == "value"
                    and row["value_update_type"] == direction
                })
                for direction in VALUE_SWITCH_DIRECTIONS
            }
            for cue in config["bridge"]["cue_regimes"]
        }
        value_action_count_by_cue_direction = {
            cue: {
                direction: len({
                    row["pre_target_action"] for row in relevant
                    if row["cue_regime"] == cue
                    and row["family"] == "value"
                    and row["value_update_type"] == direction
                })
                for direction in VALUE_SWITCH_DIRECTIONS
            }
            for cue in config["bridge"]["cue_regimes"]
        }
        active_accuracy_by_cue_family = {
            cue: {
                family: _mean(
                    row["active_accuracy"] for row in relevant
                    if row["cue_regime"] == cue and row["family"] == family
                )
                for family in AUDIT_FAMILIES
            }
            for cue in config["bridge"]["cue_regimes"]
        }
        no_switch_accuracy_by_cue_family = {
            cue: {
                family: _mean(
                    row["active_accuracy"]
                    for row in no_switch
                    if row["cue_regime"] == cue and row["family"] == family
                )
                for family in AUDIT_FAMILIES
            }
            for cue in config["bridge"]["cue_regimes"]
        }
        irrelevant_by_cue_family = {
            cue: {
                family: _mean(
                    abs(row["shift"])
                    for row in irrelevant
                    if row["cue_regime"] == cue and row["family"] == family
                )
                for family in AUDIT_FAMILIES
            }
            for cue in config["bridge"]["cue_regimes"]
        }
        sham_by_cue_family = {
            cue: {
                family: _mean(
                    abs(row["sham_shift"])
                    for row in run_effects
                    if row["cue_regime"] == cue and row["family"] == family
                )
                for family in AUDIT_FAMILIES
            }
            for cue in config["bridge"]["cue_regimes"]
        }
        no_switch_by_cue_family = {
            cue: {
                family: _mean(
                    abs(row["shift"])
                    for row in no_switch
                    if row["cue_regime"] == cue and row["family"] == family
                )
                for family in AUDIT_FAMILIES
            }
            for cue in config["bridge"]["cue_regimes"]
        }

        def retention_by_cue_family(
            surface_key: str,
        ) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, int]]]:
            result: dict[str, dict[str, float]] = {}
            counts: dict[str, dict[str, int]] = {}
            for cue in config["bridge"]["cue_regimes"]:
                result[cue] = {}
                counts[cue] = {}
                for family in AUDIT_FAMILIES:
                    cell = [
                        row for row in relevant
                        if row["cue_regime"] == cue and row["family"] == family
                    ]
                    surfaces: dict[str, list[float]] = defaultdict(list)
                    for row in cell:
                        surfaces[str(row[surface_key])].append(float(row["shift"]))
                    counts[cue][family] = len(surfaces)
                    reference = _mean(row["shift"] for row in cell)
                    if len(surfaces) < 2 or not math.isfinite(reference) or reference <= 0:
                        result[cue][family] = float("nan")
                    else:
                        result[cue][family] = (
                            min(_mean(values) for values in surfaces.values()) / reference
                        )
            return result, counts

        (
            role_retention_by_cue_family,
            role_count_by_cue_family,
        ) = retention_by_cue_family("role_assignment")
        (
            renderer_retention_by_cue_family,
            renderer_count_by_cue_family,
        ) = retention_by_cue_family("renderer_id")
        output.append({
            "arm": arm,
            "pair_seed": seed,
            "checkpoint_update": checkpoint,
            "cumulative_acquisition_diagnostics": dict(
                run_rows[0]["checkpoint_acquisition_diagnostics"]
            ),
            "acquisition_gate_window_diagnostics": dict(
                run_rows[0]["checkpoint_acquisition_gate_window_diagnostics"]
            ),
            "record_count": len(run_rows),
            "ordinary_accuracy": _mean(
                row["predicted_action"] == row["expected_actions"][arm] for row in ordinary
            ),
            "ordinary_accuracy_by_cue_regime": ordinary_accuracy_by_cue,
            "comprehension_accuracy_by_family": comprehension_by_family,
            "comprehension_accuracy_by_cue_regime_family": comprehension_by_cue_family,
            "relevant_shift_by_family": family_relevant,
            "active_switch_accuracy_by_family": active_accuracy_by_family,
            "active_no_switch_accuracy_by_family": no_switch_accuracy_by_family,
            "irrelevant_channel_mean_abs_shift": _mean(abs(row["shift"]) for row in irrelevant),
            "max_renderer_family_abs_sham_shift": _cell_max_mean(
                run_effects, "sham_shift", ("renderer_id", "family", "channel")
            ),
            "max_renderer_family_abs_no_switch_shift": _cell_max_mean(
                no_switch, "shift", ("renderer_id", "family", "channel")
            ),
            "role_swap_effect_retention": min(role_means) / reference_effect if role_means else float("nan"),
            "renderer_effect_retention": min(renderer_means) / reference_effect if renderer_means else float("nan"),
            "cue_regime_effect_retention": min(cue_means) / reference_effect if cue_means else float("nan"),
            "relevant_shift_by_cue_regime_family": relevant_by_cue_family,
            "value_relevant_shift_by_cue_regime_direction": value_relevant_by_cue_direction,
            "value_active_switch_accuracy_by_cue_regime_direction": value_accuracy_by_cue_direction,
            "value_switch_cell_count_by_cue_regime_direction": value_count_by_cue_direction,
            "value_switch_role_count_by_cue_regime_direction": value_role_count_by_cue_direction,
            "value_switch_renderer_count_by_cue_regime_direction": value_renderer_count_by_cue_direction,
            "value_switch_action_count_by_cue_regime_direction": value_action_count_by_cue_direction,
            "active_switch_accuracy_by_cue_regime_family": active_accuracy_by_cue_family,
            "active_no_switch_accuracy_by_cue_regime_family": no_switch_accuracy_by_cue_family,
            "irrelevant_channel_mean_abs_shift_by_cue_regime_family": irrelevant_by_cue_family,
            "sham_mean_abs_shift_by_cue_regime_family": sham_by_cue_family,
            "no_switch_mean_abs_shift_by_cue_regime_family": no_switch_by_cue_family,
            "role_swap_effect_retention_by_cue_regime_family": role_retention_by_cue_family,
            "renderer_effect_retention_by_cue_regime_family": renderer_retention_by_cue_family,
            "role_assignment_count_by_cue_regime_family": role_count_by_cue_family,
            "renderer_count_by_cue_regime_family": renderer_count_by_cue_family,
            "role_assignment_count": len(role_cells),
            "renderer_count": len(renderer_cells),
            "cue_regime_count": len(cue_cells),
            "minimum_legal_choice_mass": min(float(row["legal_choice_mass"]) for row in run_rows),
            "invalid_choice_rate": 1.0 - _mean(row["parse_status"] == "exact" for row in generation),
            "sampled_generation_count": len(generation),
            "adapter_reload_max_probability_delta": _maximum_finite(
                float(row["adapter_reload_max_probability_delta"])
                for row in run_rows
                if row["adapter_reload_max_probability_delta"] is not None
            ) if checkpoint > 0 else None,
        })
    return output


def _paired_ordinary_metrics(rows: Sequence[dict[str, Any]], checkpoint: int) -> dict[str, Any]:
    by_arm_seed_case = {
        (str(row["arm"]), int(row["pair_seed"]), str(row["case_id"])): row
        for row in rows
        if int(row["checkpoint_update"]) == checkpoint and row.get("condition") == "ordinary"
    }
    seeds = sorted({key[1] for key in by_arm_seed_case})
    disagreements: list[float] = []
    probability_gaps: list[float] = []
    by_seed: dict[str, dict[str, float | int]] = {}
    by_seed_cue: dict[str, dict[str, dict[str, float | int]]] = {}
    for seed in seeds:
        genuine_ids = {key[2] for key in by_arm_seed_case if key[:2] == ("genuine", seed)}
        proxy_ids = {key[2] for key in by_arm_seed_case if key[:2] == ("proxy", seed)}
        if genuine_ids != proxy_ids or not genuine_ids:
            raise ValueError(f"Ordinary G/P pairing is incomplete for seed {seed}")
        seed_disagreements = [
            float(
                by_arm_seed_case[("genuine", seed, case)]["predicted_action"]
                != by_arm_seed_case[("proxy", seed, case)]["predicted_action"]
            )
            for case in genuine_ids
        ]
        seed_probability_gaps = [
            abs(
                float(by_arm_seed_case[("genuine", seed, case)]["probability_A"])
                - float(by_arm_seed_case[("proxy", seed, case)]["probability_A"])
            )
            for case in genuine_ids
        ]
        disagreements.extend(seed_disagreements)
        probability_gaps.extend(seed_probability_gaps)
        by_seed[str(seed)] = {
            "ordinary_case_count": len(genuine_ids),
            "action_disagreement_rate": _mean(seed_disagreements),
            "mean_abs_probability_A_gap": _mean(seed_probability_gaps),
            "max_abs_probability_A_gap": max(seed_probability_gaps),
        }
        cues = sorted({
            str(by_arm_seed_case[("genuine", seed, case)].get("cue_regime", "unknown"))
            for case in genuine_ids
        })
        by_seed_cue[str(seed)] = {}
        for cue in cues:
            cue_ids = {
                case for case in genuine_ids
                if str(
                    by_arm_seed_case[("genuine", seed, case)].get(
                        "cue_regime", "unknown"
                    )
                ) == cue
            }
            proxy_cue_ids = {
                case for case in proxy_ids
                if str(
                    by_arm_seed_case[("proxy", seed, case)].get(
                        "cue_regime", "unknown"
                    )
                ) == cue
            }
            if cue_ids != proxy_cue_ids or not cue_ids:
                raise ValueError(
                    f"Ordinary G/P cue pairing is incomplete for seed {seed}, cue {cue!r}"
                )
            cue_disagreements = [
                float(
                    by_arm_seed_case[("genuine", seed, case)]["predicted_action"]
                    != by_arm_seed_case[("proxy", seed, case)]["predicted_action"]
                )
                for case in cue_ids
            ]
            cue_probability_gaps = [
                abs(
                    float(by_arm_seed_case[("genuine", seed, case)]["probability_A"])
                    - float(by_arm_seed_case[("proxy", seed, case)]["probability_A"])
                )
                for case in cue_ids
            ]
            by_seed_cue[str(seed)][cue] = {
                "ordinary_case_count": len(cue_ids),
                "action_disagreement_rate": _mean(cue_disagreements),
                "mean_abs_probability_A_gap": _mean(cue_probability_gaps),
                "max_abs_probability_A_gap": max(cue_probability_gaps),
            }
    seed_cue_cells = [
        cell for cues in by_seed_cue.values() for cell in cues.values()
    ]
    return {
        "ordinary_action_disagreement": _mean(disagreements),
        "ordinary_probability_gap": _mean(probability_gaps),
        "maximum_seed_action_disagreement": max(
            (float(value["action_disagreement_rate"]) for value in by_seed.values()),
            default=float("nan"),
        ),
        "maximum_seed_mean_probability_gap": max(
            (float(value["mean_abs_probability_A_gap"]) for value in by_seed.values()),
            default=float("nan"),
        ),
        "maximum_seed_cue_action_disagreement": max(
            (float(value["action_disagreement_rate"]) for value in seed_cue_cells),
            default=float("nan"),
        ),
        "maximum_seed_cue_mean_probability_gap": max(
            (float(value["mean_abs_probability_A_gap"]) for value in seed_cue_cells),
            default=float("nan"),
        ),
        "by_seed": by_seed,
        "by_seed_cue_regime": by_seed_cue,
    }


def _paired_all_case_probability_gap(
    rows: Sequence[dict[str, Any]], checkpoint: int
) -> float:
    by_arm_seed_case = {
        (str(row["arm"]), int(row["pair_seed"]), str(row["case_id"])): float(row["probability_A"])
        for row in rows
        if int(row["checkpoint_update"]) == checkpoint
    }
    gaps: list[float] = []
    for seed in sorted({key[1] for key in by_arm_seed_case}):
        genuine_ids = {key[2] for key in by_arm_seed_case if key[:2] == ("genuine", seed)}
        proxy_ids = {key[2] for key in by_arm_seed_case if key[:2] == ("proxy", seed)}
        if genuine_ids != proxy_ids or not genuine_ids:
            raise ValueError(f"Initial full-case G/P pairing is incomplete for seed {seed}")
        gaps.extend(
            abs(
                by_arm_seed_case[("genuine", seed, case)]
                - by_arm_seed_case[("proxy", seed, case)]
            )
            for case in genuine_ids
        )
    return max(gaps, default=float("nan"))


def _minimum_finite(values: Iterable[float]) -> float:
    collected = [float(value) for value in values]
    return min(collected) if collected and all(math.isfinite(value) for value in collected) else float("nan")


def _maximum_finite(values: Iterable[float]) -> float:
    collected = [float(value) for value in values]
    return max(collected) if collected and all(math.isfinite(value) for value in collected) else float("nan")


def _value_direction_gate_statistics(
    config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, float | bool]:
    """Take worst cells so opposite value-update directions can never average out."""
    cues = config["bridge"]["cue_regimes"]
    complete = all(
        int(row["value_switch_cell_count_by_cue_regime_direction"][cue][direction]) > 0
        and int(row["value_switch_role_count_by_cue_regime_direction"][cue][direction]) >= 2
        and int(row["value_switch_renderer_count_by_cue_regime_direction"][cue][direction]) >= 2
        and int(row["value_switch_action_count_by_cue_regime_direction"][cue][direction]) >= 2
        for row in rows
        for cue in cues
        for direction in VALUE_SWITCH_DIRECTIONS
    )
    return {
        "cells_complete": complete,
        "minimum_relevant_shift": _minimum_finite(
            row["value_relevant_shift_by_cue_regime_direction"][cue][direction]
            for row in rows
            for cue in cues
            for direction in VALUE_SWITCH_DIRECTIONS
        ),
        "minimum_learning_induced_shift": _minimum_finite(
            row["learning_induced_value_shift_by_cue_regime_direction"][cue][direction]
            for row in rows
            for cue in cues
            for direction in VALUE_SWITCH_DIRECTIONS
        ),
        "minimum_active_switch_accuracy": _minimum_finite(
            row["value_active_switch_accuracy_by_cue_regime_direction"][cue][direction]
            for row in rows
            for cue in cues
            for direction in VALUE_SWITCH_DIRECTIONS
        ),
    }


def _cluster_bootstrap_interval(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    replicates: int,
    random_seed: int,
) -> dict[str, float | int]:
    """Hierarchical seed/world bootstrap; checkpoints are never independent units."""
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    cluster_values: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        value = float(row[value_key])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite bootstrap value for {value_key}")
        cluster_values[int(row["pair_seed"])][str(row["world_id"])].append(value)
    if not cluster_values or any(not worlds for worlds in cluster_values.values()):
        return {
            "estimate": float("nan"), "lower_95": float("nan"),
            "upper_95": float("nan"), "seed_count": 0, "world_cluster_count": 0,
        }
    world_sets = {seed: set(worlds) for seed, worlds in cluster_values.items()}
    reference_worlds = set(next(iter(world_sets.values())))
    if not reference_worlds or any(worlds != reference_worlds for worlds in world_sets.values()):
        raise ValueError(
            "Crossed seed×world bootstrap requires the identical world set for every seed"
        )
    world_ids = np.asarray(sorted(reference_worlds), dtype=object)
    arrays = {
        seed: {
            world_id: _mean(cluster_values[seed][world_id]) for world_id in world_ids
        }
        for seed in cluster_values
    }
    seeds = np.asarray(sorted(arrays), dtype=int)
    rng = np.random.default_rng(random_seed)
    samples = np.empty(replicates, dtype=float)
    for index in range(replicates):
        selected_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        # Worlds are crossed with seeds in this assay. Draw one common multiset of
        # worlds and apply it to every selected seed so their shared difficulty is
        # preserved instead of pretending each seed saw unrelated worlds.
        selected_worlds = rng.choice(world_ids, size=len(world_ids), replace=True)
        sampled_seed_means: list[float] = []
        for seed in selected_seeds:
            sampled_seed_means.append(
                float(np.mean([arrays[int(seed)][str(world)] for world in selected_worlds]))
            )
        samples[index] = float(np.mean(sampled_seed_means))
    observed_seed_means = [
        float(np.mean(list(values.values()))) for values in arrays.values()
    ]
    return {
        "estimate": float(np.mean(observed_seed_means)),
        "lower_95": float(np.quantile(samples, 0.025)),
        "upper_95": float(np.quantile(samples, 0.975)),
        "seed_count": len(seeds),
        "world_cluster_count": len(world_ids),
        "crossed_seed_world_resampling": True,
    }


def _bootstrap_seed(config: Mapping[str, Any], label: str) -> int:
    digest = hashlib.sha256(
        f"{config['seed']}|bridge-bootstrap|{label}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _causal_uncertainty(
    config: Mapping[str, Any],
    effects: Sequence[dict[str, Any]],
    *,
    initial_checkpoint: int,
    final_checkpoint: int,
) -> dict[str, Any]:
    replicates = int(config["evaluation"]["bootstrap_replicates"])
    initial_by_key = {
        (row["arm"], row["pair_seed"], row["case_id"]): row
        for row in effects
        if int(row["checkpoint_update"]) == initial_checkpoint
    }
    final_rows = [row for row in effects if int(row["checkpoint_update"]) == final_checkpoint]
    induced_rows: list[dict[str, Any]] = []
    for row in final_rows:
        key = (row["arm"], row["pair_seed"], row["case_id"])
        if key not in initial_by_key:
            raise ValueError(f"Final causal effect lacks checkpoint-zero match: {key}")
        induced = dict(row)
        induced["induced_shift"] = float(row["shift"]) - float(initial_by_key[key]["shift"])
        induced_rows.append(induced)
    output: dict[str, Any] = {
        "method": "crossed_seed_and_shared_world_cluster_bootstrap",
        "replicates": replicates,
        "checkpoint_units_treated_as_independent": False,
        "final_relevant_shift": {},
        "learning_induced_relevant_shift": {},
        "by_cue_regime": {},
    }
    for arm in ARMS:
        output["final_relevant_shift"][arm] = {}
        output["learning_induced_relevant_shift"][arm] = {}
        for family in AUDIT_FAMILIES:
            selected_final = [
                row for row in final_rows
                if row["arm"] == arm and row["family"] == family
                and row["mode"] == "switch" and row["relevant"]
            ]
            selected_induced = [
                row for row in induced_rows
                if row["arm"] == arm and row["family"] == family
                and row["mode"] == "switch" and row["relevant"]
            ]
            label = f"{arm}|{family}"
            output["final_relevant_shift"][arm][family] = _cluster_bootstrap_interval(
                selected_final, value_key="shift", replicates=replicates,
                random_seed=_bootstrap_seed(config, f"final|{label}"),
            )
            output["learning_induced_relevant_shift"][arm][family] = _cluster_bootstrap_interval(
                selected_induced, value_key="induced_shift", replicates=replicates,
                random_seed=_bootstrap_seed(config, f"induced|{label}"),
            )
    for cue in config["bridge"]["cue_regimes"]:
        output["by_cue_regime"][cue] = {
            "final_relevant_shift": {},
            "learning_induced_relevant_shift": {},
        }
        for arm in ARMS:
            output["by_cue_regime"][cue]["final_relevant_shift"][arm] = {}
            output["by_cue_regime"][cue]["learning_induced_relevant_shift"][arm] = {}
            for family in AUDIT_FAMILIES:
                selected_final = [
                    row for row in final_rows
                    if row["arm"] == arm and row["family"] == family
                    and row["cue_regime"] == cue and row["mode"] == "switch"
                    and row["relevant"]
                ]
                selected_induced = [
                    row for row in induced_rows
                    if row["arm"] == arm and row["family"] == family
                    and row["cue_regime"] == cue and row["mode"] == "switch"
                    and row["relevant"]
                ]
                label = f"{cue}|{arm}|{family}"
                output["by_cue_regime"][cue]["final_relevant_shift"][arm][family] = (
                    _cluster_bootstrap_interval(
                        selected_final, value_key="shift", replicates=replicates,
                        random_seed=_bootstrap_seed(config, f"cue-final|{label}"),
                    )
                )
                output["by_cue_regime"][cue]["learning_induced_relevant_shift"][arm][family] = (
                    _cluster_bootstrap_interval(
                        selected_induced, value_key="induced_shift", replicates=replicates,
                        random_seed=_bootstrap_seed(config, f"cue-induced|{label}"),
                    )
                )
    return output


def analyze_bridge_predictions(
    config: Mapping[str, Any], prediction_paths: Sequence[str | Path], *, split: str,
    base_control_path: str | Path | None = None,
) -> dict[str, Any]:
    rows, input_hashes, model_runtime_attestation, model_runtime_contract = (
        load_bridge_predictions(config, prediction_paths)
    )
    provenance = validate_bridge_predictions(config, rows, split=split)
    base_control, base_hash = _validate_unchanged_base_control(
        config,
        base_control_path,
        rows,
        split=split,
        expected_runtime_attestation=model_runtime_attestation,
        require_runtime_sidecar=base_control_path is not None,
    )
    input_hashes.update(base_hash)
    effects = paired_effects(rows)
    metrics = _checkpoint_metrics(config, rows, effects)
    checkpoints = sorted({int(row["checkpoint_update"]) for row in rows})
    final_checkpoint = max(checkpoints)
    initial_checkpoint = min(checkpoints)
    initial = [row for row in metrics if row["checkpoint_update"] == initial_checkpoint]
    final = [row for row in metrics if row["checkpoint_update"] == final_checkpoint]
    initial_by_run = {(row["arm"], row["pair_seed"]): row for row in initial}
    for row in final:
        baseline = initial_by_run.get((row["arm"], row["pair_seed"]))
        if baseline is None:
            raise ValueError(f"Final bridge run lacks checkpoint-zero assay: {(row['arm'], row['pair_seed'])}")
        row["learning_induced_relevant_shift_by_family"] = {
            family: float(row["relevant_shift_by_family"][family])
            - float(baseline["relevant_shift_by_family"][family])
            for family in AUDIT_FAMILIES
        }
        row["learning_induced_relevant_shift_by_cue_regime_family"] = {
            cue: {
                family: float(row["relevant_shift_by_cue_regime_family"][cue][family])
                - float(baseline["relevant_shift_by_cue_regime_family"][cue][family])
                for family in AUDIT_FAMILIES
            }
            for cue in config["bridge"]["cue_regimes"]
        }
        row["learning_induced_value_shift_by_cue_regime_direction"] = {
            cue: {
                direction: float(
                    row["value_relevant_shift_by_cue_regime_direction"][cue][direction]
                ) - float(
                    baseline["value_relevant_shift_by_cue_regime_direction"][cue][direction]
                )
                for direction in VALUE_SWITCH_DIRECTIONS
            }
            for cue in config["bridge"]["cue_regimes"]
        }
    final_pairing = _paired_ordinary_metrics(rows, final_checkpoint)
    initial_pairing = _paired_ordinary_metrics(rows, initial_checkpoint)
    initial_all_case_probability_gap = _paired_all_case_probability_gap(rows, initial_checkpoint)
    causal_uncertainty = _causal_uncertainty(
        config, effects, initial_checkpoint=initial_checkpoint, final_checkpoint=final_checkpoint
    )
    available_pairs = {(row["arm"], row["pair_seed"]) for row in final}
    expected_stage1 = {(arm, int(config["bridge"]["seeds"][0])) for arm in ARMS}
    expected_replication = {
        (arm, int(seed)) for arm in ARMS for seed in config["bridge"]["seeds"]
    }
    stage_gate = config["gates"]["stage1"]
    replication_gate = config["gates"]["replication"]

    def scientific_checks(gate: Mapping[str, Any], expected_pairs: set[tuple[str, int]]) -> dict[str, bool]:
        required_keys = {
            "ordinary_accuracy_min", "ordinary_action_disagreement_max",
            "ordinary_probability_gap_each_seed_max",
            "ordinary_action_disagreement_each_seed_cue_max",
            "ordinary_probability_gap_each_seed_cue_max",
            "acquisition_aligned_accuracy_each_cue_min",
            "update_comprehension_each_family_min", "relevant_value_shift_min",
            "relevant_transition_shift_min", "irrelevant_channel_shift_max",
            "irrelevant_channel_shift_each_cue_family_max",
            "sham_shift_each_family_max", "sham_shift_each_cue_family_max",
            "no_switch_shift_each_family_max", "no_switch_shift_each_cue_family_max",
            "active_no_switch_accuracy_each_family_min",
            "role_swap_effect_retention_min", "locked_renderer_recovery_min",
            "role_swap_effect_retention_each_cue_family_min",
            "renderer_effect_retention_each_cue_family_min",
            "base_control_neutral_channel_selectivity_gap_max",
            "value_direction_relevant_shift_each_cue_min",
            "value_direction_learning_induced_shift_each_cue_min",
            "value_direction_active_switch_accuracy_each_cue_min",
            "invalid_choice_rate_max",
        }
        if not required_keys <= set(gate):
            return {"full_scientific_gate_not_defined_in_this_config": False}
        relevant_by_family = {
            family: _minimum_finite(row["relevant_shift_by_family"][family] for row in final)
            for family in AUDIT_FAMILIES
        }
        induced_by_family = {
            family: _minimum_finite(
                row["learning_induced_relevant_shift_by_family"][family] for row in final
            )
            for family in AUDIT_FAMILIES
        }
        cue_relevant_min = _minimum_finite(
            row["relevant_shift_by_cue_regime_family"][cue][family]
            for row in final
            for cue in config["bridge"]["cue_regimes"]
            for family in AUDIT_FAMILIES
        )
        cue_induced_min = _minimum_finite(
            row["learning_induced_relevant_shift_by_cue_regime_family"][cue][family]
            for row in final
            for cue in config["bridge"]["cue_regimes"]
            for family in AUDIT_FAMILIES
        )
        active_accuracy_by_family = {
            family: _minimum_finite(
                row["active_switch_accuracy_by_family"][family] for row in final
            )
            for family in AUDIT_FAMILIES
        }
        cue_active_accuracy_min = _minimum_finite(
            row["active_switch_accuracy_by_cue_regime_family"][cue][family]
            for row in final
            for cue in config["bridge"]["cue_regimes"]
            for family in AUDIT_FAMILIES
        )
        no_switch_accuracy_min = _minimum_finite(
            row["active_no_switch_accuracy_by_family"][family]
            for row in final for family in AUDIT_FAMILIES
        )
        cue_no_switch_accuracy_min = _minimum_finite(
            row["active_no_switch_accuracy_by_cue_regime_family"][cue][family]
            for row in final
            for cue in config["bridge"]["cue_regimes"]
            for family in AUDIT_FAMILIES
        )
        comprehension_by_family = {
            family: _minimum_finite(row["comprehension_accuracy_by_family"][family] for row in final)
            for family in AUDIT_FAMILIES
        }
        cue_ordinary_accuracy_min = _minimum_finite(
            row["ordinary_accuracy_by_cue_regime"][cue]
            for row in final for cue in config["bridge"]["cue_regimes"]
        )
        cue_comprehension_min = _minimum_finite(
            row["comprehension_accuracy_by_cue_regime_family"][cue][family]
            for row in final
            for cue in config["bridge"]["cue_regimes"]
            for family in AUDIT_FAMILIES
        )
        cue_irrelevant_max = _maximum_finite(
            row["irrelevant_channel_mean_abs_shift_by_cue_regime_family"][cue][family]
            for row in final
            for cue in config["bridge"]["cue_regimes"]
            for family in AUDIT_FAMILIES
        )
        cue_sham_max = _maximum_finite(
            row["sham_mean_abs_shift_by_cue_regime_family"][cue][family]
            for row in final
            for cue in config["bridge"]["cue_regimes"]
            for family in AUDIT_FAMILIES
        )
        cue_no_switch_max = _maximum_finite(
            row["no_switch_mean_abs_shift_by_cue_regime_family"][cue][family]
            for row in final
            for cue in config["bridge"]["cue_regimes"]
            for family in AUDIT_FAMILIES
        )
        cue_role_retention_min = _minimum_finite(
            row["role_swap_effect_retention_by_cue_regime_family"][cue][family]
            for row in final
            for cue in config["bridge"]["cue_regimes"]
            for family in AUDIT_FAMILIES
        )
        cue_renderer_retention_min = _minimum_finite(
            row["renderer_effect_retention_by_cue_regime_family"][cue][family]
            for row in final
            for cue in config["bridge"]["cue_regimes"]
            for family in AUDIT_FAMILIES
        )
        cue_role_cells_complete = all(
            int(row["role_assignment_count_by_cue_regime_family"][cue][family]) >= 2
            for row in final
            for cue in config["bridge"]["cue_regimes"]
            for family in AUDIT_FAMILIES
        )
        cue_renderer_cells_complete = all(
            int(row["renderer_count_by_cue_regime_family"][cue][family]) >= 2
            for row in final
            for cue in config["bridge"]["cue_regimes"]
            for family in AUDIT_FAMILIES
        )
        value_direction_statistics = _value_direction_gate_statistics(config, final)
        legal_minimum = float(gate.get("legal_choice_mass_min", 0.05))
        relevant_ci_lower = _minimum_finite(
            causal_uncertainty["final_relevant_shift"][arm][family]["lower_95"]
            for arm in ARMS for family in AUDIT_FAMILIES
        )
        induced_ci_lower = _minimum_finite(
            causal_uncertainty["learning_induced_relevant_shift"][arm][family]["lower_95"]
            for arm in ARMS for family in AUDIT_FAMILIES
        )
        cue_relevant_ci_lower = _minimum_finite(
            causal_uncertainty["by_cue_regime"][cue]["final_relevant_shift"][arm][family]["lower_95"]
            for cue in config["bridge"]["cue_regimes"]
            for arm in ARMS
            for family in AUDIT_FAMILIES
        )
        cue_induced_ci_lower = _minimum_finite(
            causal_uncertainty["by_cue_regime"][cue]["learning_induced_relevant_shift"][arm][family]["lower_95"]
            for cue in config["bridge"]["cue_regimes"]
            for arm in ARMS
            for family in AUDIT_FAMILIES
        )
        acquisition_aligned_accuracy = _minimum_finite(
            row["acquisition_gate_window_diagnostics"]["by_condition"]["aligned"][
                "optimal_action_accuracy"
            ]
            for row in final
        )
        acquisition_aligned_each_cue = _minimum_finite(
            row["acquisition_gate_window_diagnostics"]["cells"][cue]["aligned"][
                "optimal_action_accuracy"
            ]
            for row in final for cue in config["bridge"]["cue_regimes"]
        )
        acquisition_conflict_accuracy = _minimum_finite(
            row["acquisition_gate_window_diagnostics"]["by_condition"][
                "diagnostic_conflict"
            ]["optimal_action_accuracy"]
            for row in final
        )
        acquisition_conflict_each_cue = _minimum_finite(
            row["acquisition_gate_window_diagnostics"]["cells"][cue][
                "diagnostic_conflict"
            ]["optimal_action_accuracy"]
            for row in final for cue in config["bridge"]["cue_regimes"]
        )
        acquisition_cells_nonempty = all(
            int(
                row["acquisition_gate_window_diagnostics"]["cells"][cue][condition][
                    "sample_count"
                ]
            ) > 0
            for row in final
            for cue in config["bridge"]["cue_regimes"]
            for condition in ACQUISITION_CONDITIONS
        )
        return {
            "complete_paired_matrix": available_pairs == expected_pairs,
            "unchanged_base_loader_integrity": (
                base_control.get("integrity_pass") is True
                if split == config["bridge"]["splits"]["development"]
                else True
            ),
            "unchanged_base_not_channel_selective_under_neutral_cues": (
                base_control.get("scientific_fingerprint_pass") is True
                if split == config["bridge"]["splits"]["development"]
                else True
            ),
            "paired_initialization_behavior": initial_pairing[
                "maximum_seed_mean_probability_gap"
            ] <= 1e-6,
            "paired_initialization_all_cases": initial_all_case_probability_gap <= 1e-6,
            "terminal_acquisition_window_cells_complete": acquisition_cells_nonempty,
            "terminal_acquisition_window_aligned_objective_accuracy": acquisition_aligned_accuracy
            >= float(gate.get("acquisition_aligned_accuracy_min", 0.85)),
            "terminal_acquisition_window_aligned_accuracy_each_cue_regime": acquisition_aligned_each_cue
            >= float(gate["acquisition_aligned_accuracy_each_cue_min"]),
            "terminal_acquisition_window_conflict_objective_accuracy": acquisition_conflict_accuracy
            >= float(gate.get("acquisition_conflict_accuracy_min", 0.65)),
            "terminal_acquisition_window_conflict_accuracy_each_cue_regime": acquisition_conflict_each_cue
            >= float(gate.get("acquisition_conflict_accuracy_each_cue_min", 0.60)),
            "ordinary_accuracy": _minimum_finite(row["ordinary_accuracy"] for row in final)
            >= float(gate["ordinary_accuracy_min"]),
            "ordinary_accuracy_each_cue_regime": cue_ordinary_accuracy_min
            >= float(gate["ordinary_accuracy_min"]),
            "ordinary_action_equivalence": final_pairing["maximum_seed_action_disagreement"]
            <= float(gate["ordinary_action_disagreement_max"]),
            "ordinary_probability_equivalence": final_pairing[
                "maximum_seed_mean_probability_gap"
            ] <= float(gate["ordinary_probability_gap_each_seed_max"]),
            "ordinary_action_equivalence_each_seed_cue_regime": final_pairing[
                "maximum_seed_cue_action_disagreement"
            ] <= float(gate["ordinary_action_disagreement_each_seed_cue_max"]),
            "ordinary_probability_equivalence_each_seed_cue_regime": final_pairing[
                "maximum_seed_cue_mean_probability_gap"
            ] <= float(gate["ordinary_probability_gap_each_seed_cue_max"]),
            "value_update_comprehension": comprehension_by_family["value"]
            >= float(gate["update_comprehension_each_family_min"]),
            "transition_update_comprehension": comprehension_by_family["transition"]
            >= float(gate["update_comprehension_each_family_min"]),
            "update_comprehension_each_cue_regime": cue_comprehension_min
            >= float(gate["update_comprehension_each_family_min"]),
            "genuine_and_proxy_value_sensitivity": relevant_by_family["value"]
            >= float(gate["relevant_value_shift_min"]),
            "genuine_and_proxy_transition_sensitivity": relevant_by_family["transition"]
            >= float(gate["relevant_transition_shift_min"]),
            "value_switch_choice_reversal": active_accuracy_by_family["value"]
            >= float(gate.get("active_switch_accuracy_each_family_min", 0.80)),
            "transition_switch_choice_reversal": active_accuracy_by_family["transition"]
            >= float(gate.get("active_switch_accuracy_each_family_min", 0.80)),
            "learning_induced_value_sensitivity": induced_by_family["value"]
            >= float(gate.get("learning_induced_relevant_shift_min", 0.10)),
            "learning_induced_transition_sensitivity": induced_by_family["transition"]
            >= float(gate.get("learning_induced_relevant_shift_min", 0.10)),
            "relevant_effect_bootstrap_lower_bound": relevant_ci_lower
            > float(gate.get("relevant_shift_ci_lower_min", 0.0)),
            "learning_induced_bootstrap_lower_bound": induced_ci_lower
            > float(gate.get("learning_induced_shift_ci_lower_min", 0.0)),
            "each_cue_regime_relevant_effect": cue_relevant_min
            >= float(gate.get("cue_regime_relevant_shift_min", 0.20)),
            "each_cue_regime_learning_induced_effect": cue_induced_min
            >= float(gate.get("cue_regime_learning_induced_shift_min", 0.10)),
            "each_cue_regime_switch_choice_reversal": cue_active_accuracy_min
            >= float(gate.get("cue_regime_active_switch_accuracy_min", 0.80)),
            "both_value_switch_directions_complete_within_each_cue_regime": (
                value_direction_statistics["cells_complete"] is True
            ),
            "both_value_switch_directions_relevant_within_each_cue_regime": (
                float(value_direction_statistics["minimum_relevant_shift"])
                >= float(gate["value_direction_relevant_shift_each_cue_min"])
            ),
            "both_value_switch_directions_learning_induced_within_each_cue_regime": (
                float(value_direction_statistics["minimum_learning_induced_shift"])
                >= float(gate["value_direction_learning_induced_shift_each_cue_min"])
            ),
            "both_value_switch_directions_reverse_choice_within_each_cue_regime": (
                float(value_direction_statistics["minimum_active_switch_accuracy"])
                >= float(gate["value_direction_active_switch_accuracy_each_cue_min"])
            ),
            "each_cue_regime_relevant_bootstrap_lower_bound": cue_relevant_ci_lower
            > float(gate.get("relevant_shift_ci_lower_min", 0.0)),
            "each_cue_regime_learning_induced_bootstrap_lower_bound": cue_induced_ci_lower
            > float(gate.get("learning_induced_shift_ci_lower_min", 0.0)),
            "irrelevant_channel_specificity": _maximum_finite(
                row["irrelevant_channel_mean_abs_shift"] for row in final
            ) <= float(gate["irrelevant_channel_shift_max"]),
            "irrelevant_channel_specificity_each_cue_family": cue_irrelevant_max
            <= float(gate["irrelevant_channel_shift_each_cue_family_max"]),
            "sham_specificity_each_family": _maximum_finite(
                row["max_renderer_family_abs_sham_shift"] for row in final
            ) <= float(gate["sham_shift_each_family_max"]),
            "sham_specificity_each_cue_family": cue_sham_max
            <= float(gate["sham_shift_each_cue_family_max"]),
            "active_no_switch_control": _maximum_finite(
                row["max_renderer_family_abs_no_switch_shift"] for row in final
            ) <= float(gate["no_switch_shift_each_family_max"]),
            "active_no_switch_control_each_cue_family": cue_no_switch_max
            <= float(gate["no_switch_shift_each_cue_family_max"]),
            "active_no_switch_choice_accuracy": no_switch_accuracy_min
            >= float(gate["active_no_switch_accuracy_each_family_min"]),
            "active_no_switch_choice_accuracy_each_cue_regime": cue_no_switch_accuracy_min
            >= float(gate["active_no_switch_accuracy_each_family_min"]),
            "channel_role_counterbalance": _minimum_finite(
                row["role_swap_effect_retention"] for row in final
            ) >= float(gate["role_swap_effect_retention_min"])
            and all(int(row["role_assignment_count"]) >= 2 for row in final),
            "channel_role_counterbalance_within_each_cue_family": cue_role_retention_min
            >= float(gate["role_swap_effect_retention_each_cue_family_min"])
            and cue_role_cells_complete,
            "heldout_renderer_robustness": _minimum_finite(
                row["renderer_effect_retention"] for row in final
            ) >= float(gate["locked_renderer_recovery_min"])
            and all(int(row["renderer_count"]) >= 2 for row in final),
            "renderer_robustness_within_each_cue_family": cue_renderer_retention_min
            >= float(gate["renderer_effect_retention_each_cue_family_min"])
            and cue_renderer_cells_complete,
            "semantic_neutral_cue_robustness": _minimum_finite(
                row["cue_regime_effect_retention"] for row in final
            ) >= float(gate.get("cue_regime_effect_retention_min", 0.70))
            and all(
                int(row["cue_regime_count"]) == len(config["bridge"]["cue_regimes"])
                for row in final
            ),
            "legal_choice_mass": _minimum_finite(row["minimum_legal_choice_mass"] for row in final)
            >= legal_minimum,
            "unconstrained_first_action_parse": _maximum_finite(
                row["invalid_choice_rate"] for row in final
            ) <= float(gate["invalid_choice_rate_max"]),
        }

    stage1_checks = scientific_checks(stage_gate, expected_stage1)
    replication_checks = scientific_checks(replication_gate, expected_replication)
    smoke_gate = config["gates"]["smoke"]
    trained_rows = [row for row in rows if int(row["checkpoint_update"]) > 0]
    finite_optimizer_metrics = bool(trained_rows) and all(
        isinstance(row.get("checkpoint_optimizer_metrics"), Mapping)
        and bool(row["checkpoint_optimizer_metrics"])
        and all(
            math.isfinite(float(value))
            for value in row["checkpoint_optimizer_metrics"].values()
        )
        for row in trained_rows
    )
    smoke_acquisition_complete = all(
        int(row["cumulative_acquisition_diagnostics"]["overall"]["sample_count"])
        == int(row["checkpoint_update"]) * int(config["training"]["rollout_batch_size"])
        for row in final
    )
    smoke_acquisition_gate_window_complete = all(
        int(row["acquisition_gate_window_diagnostics"]["overall"]["sample_count"])
        == min(
            int(row["checkpoint_update"]),
            int(config["training"]["acquisition_gate_window_updates"]),
        ) * int(config["training"]["rollout_batch_size"])
        for row in final
    )
    smoke_checks = {
        "finite_scores": all(
            math.isfinite(float(row[key]))
            for row in metrics
            for key in ("minimum_legal_choice_mass", "invalid_choice_rate")
        ),
        # Smoke is an engineering check, so it verifies that checkpoint-zero LoRA
        # exactly reproduces the adapter-free base.  Channel selectivity is a
        # scientific gate on the full formal DEV assay below, not on the tiny
        # semantic-only smoke corpus.
        "unchanged_base_loader_integrity": base_control.get("integrity_pass") is True,
        "paired_run_count": len(available_pairs) == int(smoke_gate["paired_run_count"]),
        "finite_training_loss": (
            finite_optimizer_metrics
            if bool(smoke_gate.get("finite_training_loss", True))
            else True
        ),
        "cumulative_acquisition_diagnostics_complete": smoke_acquisition_complete,
        "acquisition_gate_window_diagnostics_complete": (
            smoke_acquisition_gate_window_complete
        ),
        "paired_initialization_behavior": initial_pairing[
            "maximum_seed_mean_probability_gap"
        ] <= 1e-6,
        "paired_initialization_all_cases": initial_all_case_probability_gap <= 1e-6,
        "invalid_choice_rate": _maximum_finite(row["invalid_choice_rate"] for row in final)
        <= float(smoke_gate["invalid_choice_rate_max"]),
        "adapter_reload_probability": _maximum_finite(
            row["adapter_reload_max_probability_delta"] for row in final
        ) <= float(smoke_gate["adapter_reload_max_probability_delta"]),
    }
    return {
        "schema_version": "1.0",
        "kind": "same_environment_bridge_report",
        "config_sha256": config["_config_sha256"],
        "split": split,
        "input_sha256": input_hashes,
        "model_runtime_attestation": model_runtime_attestation,
        "model_runtime_attestation_sha256": model_runtime_attestation[
            "attestation_sha256"
        ],
        "model_runtime_contract": model_runtime_contract,
        **provenance,
        "checkpoint_updates": checkpoints,
        "final_checkpoint_update": final_checkpoint,
        "available_arm_seed_pairs": sorted([list(pair) for pair in available_pairs]),
        "paired_initial_ordinary": initial_pairing,
        "paired_initial_all_case_max_probability_gap": initial_all_case_probability_gap,
        "paired_final_ordinary": final_pairing,
        "acquisition_diagnostic_policy": {
            "continuation_gate_basis": "trailing_optimizer_updates",
            "window_updates": int(
                config["training"]["acquisition_gate_window_updates"]
            ),
            "samples_per_update": int(config["training"]["rollout_batch_size"]),
            "cumulative_diagnostics_role": "learning_curve_evidence_only",
        },
        "checkpoint_metrics": metrics,
        "paired_effects": effects,
        "causal_uncertainty": causal_uncertainty,
        "unchanged_base_control": base_control,
        "direct_conflict_incremental_status": "NOT_ESTIMABLE_IN_PURE_TWO_ARM_CONSTRUCT_BRIDGE",
        "gates": {
            "smoke": {"pass": all(smoke_checks.values()), "checks": smoke_checks},
            "stage1": {"pass": all(stage1_checks.values()), "checks": stage1_checks},
            "replication": {"pass": all(replication_checks.values()), "checks": replication_checks},
        },
        "decision": (
            "LOCKED_REPLICATION_PASSED_PLAN_EXTERNAL_VALIDATION"
            if split == config["bridge"]["splits"]["locked"] and all(replication_checks.values())
            else "DEV_BRIDGE_PASSED_HUMAN_REVIEW_REQUIRED"
            if split == config["bridge"]["splits"]["development"] and all(stage1_checks.values())
            else "STOP_OR_DEBUG_WITHOUT_OPENING_LOCKED_TEST"
        ),
    }


def write_bridge_analysis(
    config: Mapping[str, Any],
    prediction_paths: Sequence[str | Path],
    *,
    split: str,
    destination: str | Path,
    base_control_path: str | Path | None = None,
) -> Path:
    target = Path(destination).resolve()
    report = analyze_bridge_predictions(
        config, prediction_paths, split=split, base_control_path=base_control_path
    )
    write_json(target, report)
    return target


def verify_bridge_gate_report(
    config: Mapping[str, Any], report_path: str | Path, *, required: str
) -> dict[str, Any]:
    if required not in {"smoke", "stage1", "replication"}:
        raise ValueError(f"Unknown bridge gate {required!r}")
    path = Path(report_path).resolve()
    report = json.loads(path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if report.get("kind") != "same_environment_bridge_report":
        failures.append("wrong report kind")
    if report.get("config_sha256") != config["_config_sha256"]:
        failures.append("config hash mismatch")
    raw_runtime = report.get("model_runtime_attestation")
    if not isinstance(raw_runtime, Mapping):
        failures.append("full model runtime attestation is missing")
    else:
        try:
            verified_runtime = verify_model_runtime_attestation(config, raw_runtime)
            if (
                report.get("model_runtime_attestation_sha256")
                != verified_runtime["attestation_sha256"]
                or report.get("model_runtime_contract")
                != compact_model_runtime_contract(config, verified_runtime)
            ):
                failures.append("model runtime attestation/contract mismatch")
        except (TypeError, ValueError) as exc:
            failures.append(f"model runtime attestation failed: {exc}")
    expected_split = (
        config["bridge"]["splits"]["locked"]
        if required == "replication"
        else config["bridge"]["splits"]["development"]
    )
    if report.get("split") != expected_split:
        failures.append(
            f"gate {required} requires split {expected_split!r}, not {report.get('split')!r}"
        )
    for raw_path, expected_hash in (report.get("input_sha256") or {}).items():
        source = Path(raw_path)
        if not source.is_file() or sha256_file(source) != expected_hash:
            failures.append(f"prediction input missing or changed: {source}")
    gate = (report.get("gates") or {}).get(required)
    if not isinstance(gate, Mapping) or gate.get("pass") is not True:
        failures.append(f"gate {required} did not pass")
    return {"required": required, "pass": not failures, "failures": failures}
