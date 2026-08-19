"""Extinction evaluation for environment-grounded bridge checkpoints.

Evaluation restores the environment state saved with an acquisition checkpoint,
applies passive exposure through the environment, and scores exactly the first test
choice.  It never calls an environment transition after that choice.  Development is
the default and test access requires an explicit unlock flag.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from .bridge_training import (
    CHOICES,
    BridgeEnvironment,
    BridgeTrainingSpec,
    _environment_provenance,
    _environment_state_hash,
    _spec_hash,
    _verify_checkpoint,
    acquisition_gate_window_diagnostics_summary,
    acquisition_diagnostics_summary,
    canonical_arm,
    differentiable_choice_log_probs,
    load_bridge_state,
    validate_acquisition_gate_window_state,
    validate_acquisition_diagnostics_state,
)
from .config import config_hash, output_root
from .io import canonical_json, read_jsonl, sha256_file, write_json, write_jsonl
from .manifest import environment_snapshot, project_hash
from .modeling import (
    chat_prompt_text,
    load_adapter_model,
    load_base_model,
    load_tokenizer,
    validate_loaded_base_runtime,
    validate_loaded_lora_runtime,
    verify_model_runtime_attestation,
)


EXTINCTION_INVARIANTS = {
    "policy_frozen_at_checkpoint": True,
    "passive_revaluation": True,
    "target_action_taken_during_revaluation": False,
    "parameter_update_during_revaluation": False,
    "reward_delivered_at_test": False,
    "feedback_delivered_at_test": False,
    "first_choice_only": True,
    "max_test_choices": 1,
    "same_environment_and_causal_schema_as_acquisition": True,
    "heldout_world": True,
}


@runtime_checkable
class BridgeEvaluationEnvironment(BridgeEnvironment, Protocol):
    """Acquisition environment extended with frozen passive-exposure assays."""

    def extinction_cases(
        self, *, split: str, trajectory_seed: int, checkpoint_update: int
    ) -> Sequence[Mapping[str, Any]]: ...


@dataclass(frozen=True)
class BridgeEvaluationSpec:
    batch_size: int = 16
    generation_subset_size: int = 256
    generation_batch_size: int = 8
    max_new_tokens: int = 1
    minimum_legal_choice_mass: float = 0.05
    minimum_exact_parse_rate: float = 0.95

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.generation_batch_size <= 0:
            raise ValueError("Evaluation and generation batch sizes must be positive")
        if self.max_new_tokens != 1:
            raise ValueError("Extinction is a first-action assay, so max_new_tokens must equal one")
        if self.generation_subset_size <= 0:
            raise ValueError("generation_subset_size must be positive")
        for name in ("minimum_legal_choice_mass", "minimum_exact_parse_rate"):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "BridgeEvaluationSpec":
        if values is None:
            return cls()
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown bridge-evaluation settings: {sorted(unknown)}")
        return cls(**dict(values))

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        overrides: Mapping[str, Any] | None = None,
    ) -> "BridgeEvaluationSpec":
        evaluation = config.get("evaluation")
        if not isinstance(evaluation, Mapping):
            raise ValueError("Bridge config requires an evaluation mapping")
        required = set(cls.__dataclass_fields__)
        missing = required - set(evaluation)
        if missing:
            raise ValueError(
                f"Bridge evaluation config lacks frozen settings: {sorted(missing)}"
            )
        values: dict[str, Any] = {
            key: evaluation[key] for key in cls.__dataclass_fields__
        }
        if overrides:
            unknown = set(overrides) - set(cls.__dataclass_fields__)
            if unknown:
                raise ValueError(f"Unknown bridge-evaluation settings: {sorted(unknown)}")
            values.update(overrides)
        return cls.from_mapping(values)


def _evaluation_spec_hash(spec: BridgeEvaluationSpec) -> str:
    return hashlib.sha256(canonical_json(asdict(spec)).encode("utf-8")).hexdigest()


def config_bound_bridge_evaluation_spec(
    config: Mapping[str, Any], requested_settings: Mapping[str, Any] | None = None
) -> BridgeEvaluationSpec:
    """Return the YAML-defined evaluator and reject evidence-changing overrides."""
    configured = BridgeEvaluationSpec.from_config(config)
    if requested_settings is not None:
        requested = BridgeEvaluationSpec.from_config(config, requested_settings)
        if requested != configured:
            raise ValueError(
                "Evidence-producing bridge evaluation forbids settings that differ "
                "from the loaded config"
            )
    return configured


def configured_bridge_evaluation_spec_sha256(config: Mapping[str, Any]) -> str:
    return _evaluation_spec_hash(config_bound_bridge_evaluation_spec(config))


def _valid_messages(messages: Any) -> bool:
    return (
        isinstance(messages, list)
        and bool(messages)
        and all(
            isinstance(message, dict)
            and message.get("role") in {"system", "user", "assistant"}
            and isinstance(message.get("content"), str)
            for message in messages
        )
        and messages[-1].get("role") == "user"
    )


def validate_extinction_cases(
    cases: Sequence[Mapping[str, Any]], *, split: str
) -> list[Mapping[str, Any]]:
    """Enforce acquisition/exposure/test separation before model evaluation."""
    collected = list(cases)
    if not collected:
        raise ValueError("The bridge environment returned no extinction cases")
    seen: set[str] = set()
    for case in collected:
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("Every extinction case needs a non-empty case_id")
        if case_id in seen:
            raise ValueError(f"Duplicate extinction case_id: {case_id}")
        seen.add(case_id)
        if case.get("split") != split:
            raise ValueError(f"Extinction case {case_id} is not in requested split {split!r}")
        for key in (
            "world_id",
            "renderer_id",
            "base_template_id",
            "intervention_template_id",
            "pair_id",
            "condition",
        ):
            if not isinstance(case.get(key), str) or not case[key]:
                raise ValueError(f"Extinction case {case_id} lacks non-empty {key}")
        if case.get("cue_regime") not in {"semantic", "neutral"}:
            raise ValueError(f"Extinction case {case_id} has invalid cue_regime")
        if not _valid_messages(case.get("messages")):
            raise ValueError(f"Extinction case {case_id} has invalid chat messages")
        protocol = case.get("extinction_protocol")
        if not isinstance(protocol, Mapping):
            raise ValueError(f"Extinction case {case_id} lacks protocol metadata")
        for key, expected in EXTINCTION_INVARIANTS.items():
            if protocol.get(key) != expected or type(protocol.get(key)) is not type(expected):
                raise ValueError(f"Extinction invariant {key} failed for {case_id}")
        pre_target = case.get("pre_target_action")
        if pre_target not in CHOICES:
            raise ValueError(f"Extinction case {case_id} lacks a legal pre_target_action")
        intervention = case.get("intervention")
        if not isinstance(intervention, Mapping) or not isinstance(intervention.get("family"), str):
            raise ValueError(f"Extinction case {case_id} lacks intervention metadata")
        if intervention.get("cue_regime") != case["cue_regime"]:
            raise ValueError(f"Extinction case {case_id} has inconsistent cue-regime metadata")
        value_update_type = intervention.get("value_update_type")
        family = intervention.get("base_family")
        mode = intervention.get("mode")
        if family == "value" and mode in {"switch", "sham", "comprehension"}:
            if value_update_type not in {
                "devalue_preferred",
                "upvalue_nonpreferred",
                "upvalue_preferred",
            }:
                raise ValueError(f"Extinction case {case_id} has invalid value-update provenance")
        elif family == "value" and mode == "no_switch":
            if value_update_type != "upvalue_preferred":
                raise ValueError(f"Extinction case {case_id} has invalid no-switch value update")
        elif value_update_type != "not_applicable":
            raise ValueError(f"Extinction case {case_id} has spurious value-update provenance")
        expected_actions = case.get("expected_actions")
        if (
            not isinstance(expected_actions, Mapping)
            or not expected_actions
            or not set(expected_actions.values()) <= set(CHOICES)
        ):
            raise ValueError(f"Invalid or missing expected_actions for {case_id}")
    by_id = {str(case["case_id"]): case for case in collected}
    active = {
        case["case_id"]: case
        for case in collected
        if bool(case["intervention"].get("active"))
    }
    for case_id, case in active.items():
        paired = case.get("paired_control_id")
        if not isinstance(paired, str) or paired not in by_id:
            raise ValueError(f"Active extinction case {case_id} lacks its paired passive control")
        control = by_id[paired]
        control_intervention = control["intervention"]
        if bool(control_intervention.get("active")) or control_intervention.get("mode") != "sham":
            raise ValueError(f"Paired control for {case_id} is not an inactive sham")
        for key in (
            "world_id",
            "renderer_id",
            "base_template_id",
            "intervention_template_id",
            "pre_target_action",
            "cue_regime",
        ):
            if case.get(key) != control.get(key):
                raise ValueError(f"Paired control for {case_id} differs on {key}")
        for key in ("base_family", "objective", "value_update_type"):
            if case["intervention"].get(key) != control_intervention.get(key):
                raise ValueError(f"Paired control for {case_id} differs on intervention {key}")
    return collected


def legal_choice_diagnostics(logp_a: float, logp_b: float) -> dict[str, float]:
    """Keep normalized preference separate from absolute legal-answer mass."""
    values = (float(logp_a), float(logp_b))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Choice log-likelihoods must be finite")
    maximum = max(values)
    exp_a = math.exp(values[0] - maximum)
    exp_b = math.exp(values[1] - maximum)
    probability_a = exp_a / (exp_a + exp_b)
    log_legal_mass = maximum + math.log(exp_a + exp_b)
    # A and B are verified prefix-disjoint completions, so their mass cannot exceed 1.
    if log_legal_mass > 1e-5:
        raise ValueError(f"Legal choice sequence mass exceeds one: exp({log_legal_mass})")
    legal_mass = math.exp(log_legal_mass) if log_legal_mass > -745 else 0.0
    return {
        "probability_A": probability_a,
        "probability_B": 1.0 - probability_a,
        "log_legal_choice_mass": log_legal_mass,
        "legal_choice_mass": legal_mass,
    }


def parse_unconstrained_choice(text: str) -> tuple[str | None, str]:
    """Only an exact, whitespace-trimmed A or B is protocol compliant."""
    stripped = text.strip()
    if stripped in CHOICES:
        return stripped, "exact"
    if not stripped:
        return None, "empty"
    if stripped[0] in CHOICES:
        return None, "extra_text"
    return None, "non_choice"


def _batches(values: Sequence[Any], size: int) -> Sequence[Sequence[Any]]:
    return [values[start : start + size] for start in range(0, len(values), size)]


def _generation_subset(cases: Sequence[Mapping[str, Any]], limit: int) -> list[Mapping[str, Any]]:
    """Deterministic round-robin sample over every assay-design cell."""
    groups: dict[tuple[str, ...], deque[Mapping[str, Any]]] = defaultdict(deque)
    for case in cases:
        intervention = case.get("intervention") or {}
        key = (
            str(case.get("renderer_id", "unknown")),
            str(case.get("cue_regime", "unknown")),
            str(case.get("condition", "unknown")),
            str(intervention.get("family", "unknown")),
            str(intervention.get("base_family", "unknown")),
            str(intervention.get("objective", "unknown")),
            str(intervention.get("mode", "unknown")),
            str(bool(intervention.get("active"))),
        )
        groups[key].append(case)
    for key, values in groups.items():
        ordered = sorted(
            values,
            key=lambda case: hashlib.sha256(str(case["case_id"]).encode("utf-8")).hexdigest(),
        )
        groups[key] = deque(ordered)
    selected: list[Mapping[str, Any]] = []
    keys = sorted(groups)
    while len(selected) < min(limit, len(cases)):
        progressed = False
        for key in keys:
            if groups[key] and len(selected) < limit:
                selected.append(groups[key].popleft())
                progressed = True
        if not progressed:
            break
    return selected


GENERATION_SUBSET_METHOD = "deterministic_round_robin_design_cells_sha256_v1"
GENERATION_DESIGN_FIELDS = (
    "renderer_id",
    "cue_regime",
    "condition",
    "family",
    "base_family",
    "objective",
    "mode",
    "active",
)


def generation_subset_attestation(
    cases: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    *,
    requested_size: int,
) -> dict[str, Any]:
    """Hash the full case universe and exact ordered deterministic generation subset."""
    all_ids = [str(case["case_id"]) for case in cases]
    selected_ids = [str(case["case_id"]) for case in selected]
    if len(all_ids) != len(set(all_ids)) or len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Generation-subset attestation requires unique case IDs")
    if not set(selected_ids) <= set(all_ids):
        raise ValueError("Generation subset contains a case outside the extinction assay")
    expected_count = min(int(requested_size), len(all_ids))
    if int(requested_size) <= 0 or len(selected_ids) != expected_count:
        raise ValueError("Generation subset does not have its config-defined size")
    return {
        "schema_version": "1.0",
        "method": GENERATION_SUBSET_METHOD,
        "design_fields": list(GENERATION_DESIGN_FIELDS),
        "requested_size": int(requested_size),
        "available_case_count": len(all_ids),
        "selected_case_count": len(selected_ids),
        "all_case_ids_sha256": hashlib.sha256(
            canonical_json(sorted(all_ids)).encode("utf-8")
        ).hexdigest(),
        "ordered_selected_case_ids_sha256": hashlib.sha256(
            canonical_json(selected_ids).encode("utf-8")
        ).hexdigest(),
        "selected_case_ids_sha256": hashlib.sha256(
            canonical_json(sorted(selected_ids)).encode("utf-8")
        ).hexdigest(),
    }


def _generate_unconstrained(
    model: Any,
    tokenizer: Any,
    cases: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
) -> dict[str, dict[str, Any]]:
    import torch

    output: dict[str, dict[str, Any]] = {}
    original_padding_side = tokenizer.padding_side
    original_truncation_side = tokenizer.truncation_side
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    try:
        for batch in _batches(list(cases), batch_size):
            prompts = [chat_prompt_text(tokenizer, list(case["messages"])) for case in batch]
            encoded = tokenizer(
                prompts,
                add_special_tokens=False,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(next(model.parameters()).device) for key, value in encoded.items()}
            input_width = int(encoded["input_ids"].shape[1])
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=int(tokenizer.pad_token_id),
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            completions = tokenizer.batch_decode(
                generated[:, input_width:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for case, completion in zip(batch, completions, strict=True):
                parsed, status = parse_unconstrained_choice(completion)
                output[str(case["case_id"])] = {
                    "generated_output": completion,
                    "parsed_action": parsed,
                    "parse_status": status,
                }
    finally:
        tokenizer.padding_side = original_padding_side
        tokenizer.truncation_side = original_truncation_side
    return output


def _verify_checkpoint_for_evaluation(
    config: Mapping[str, Any],
    environment: BridgeEvaluationEnvironment,
    checkpoint: Path,
    *,
    arm: str,
    pair_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = checkpoint / "checkpoint_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing bridge checkpoint manifest: {manifest_path}")
    configured_spec = BridgeTrainingSpec.from_config(config)
    configured_spec_sha256 = _spec_hash(configured_spec)
    # The expected spec is independently reconstructed from the loaded config. Never
    # accept a checkpoint's own training-spec hash as its expected value.
    verified = _verify_checkpoint(
        checkpoint,
        arm=arm,
        seed=pair_seed,
        config_sha256=config.get("_config_sha256") or config_hash(dict(config)),
        spec_sha256=configured_spec_sha256,
        environment_provenance=_environment_provenance(environment),
        initial_environment_state_sha256=_environment_state_hash(environment),
    )
    verify_model_runtime_attestation(
        config, verified.get("model_runtime_attestation") or {}
    )
    state = load_bridge_state(checkpoint)
    if int(state.get("completed_updates", -1)) != int(verified.get("completed_updates", -2)):
        raise ValueError("Checkpoint state and checkpoint manifest disagree on update index")
    diagnostics_state = validate_acquisition_diagnostics_state(
        state.get("acquisition_diagnostics_state") or {},
        arm=arm,
        cue_regimes=list(config["bridge"]["cue_regimes"]),
    )
    diagnostics = acquisition_diagnostics_summary(diagnostics_state)
    if verified.get("acquisition_diagnostics") != diagnostics:
        raise ValueError("Checkpoint state and manifest disagree on acquisition diagnostics")
    expected_samples = int(verified["completed_updates"]) * int(
        config["training"]["rollout_batch_size"]
    )
    if int(diagnostics["overall"]["sample_count"]) != expected_samples:
        raise ValueError(
            "Checkpoint acquisition diagnostics do not cover every completed rollout"
        )
    gate_window_state = validate_acquisition_gate_window_state(
        state.get("acquisition_gate_window_diagnostics_state") or {},
        arm=arm,
        cue_regimes=list(config["bridge"]["cue_regimes"]),
        window_updates=int(config["training"]["acquisition_gate_window_updates"]),
        samples_per_update=int(config["training"]["rollout_batch_size"]),
    )
    gate_window = acquisition_gate_window_diagnostics_summary(gate_window_state)
    if verified.get("acquisition_gate_window_diagnostics") != gate_window:
        raise ValueError(
            "Checkpoint state and manifest disagree on acquisition gate window"
        )
    expected_window_samples = min(
        int(verified["completed_updates"]),
        int(config["training"]["acquisition_gate_window_updates"]),
    ) * int(config["training"]["rollout_batch_size"])
    if (
        int(gate_window["completed_updates"]) != int(verified["completed_updates"])
        or int(gate_window["overall"]["sample_count"]) != expected_window_samples
    ):
        raise ValueError(
            "Checkpoint acquisition gate window does not cover the exact trailing updates"
        )
    initial_state_hash = verified.get("initial_environment_state_sha256")
    if (
        not isinstance(initial_state_hash, str)
        or len(initial_state_hash) != 64
        or any(character not in "0123456789abcdef" for character in initial_state_hash)
    ):
        raise ValueError("Checkpoint lacks a valid initial environment-state hash")
    return verified, state


def adapter_reload_diagnostic(
    model: Any,
    tokenizer: Any,
    checkpoint: str | Path,
    *,
    max_length: int,
) -> dict[str, Any]:
    """Compare reloaded probabilities with same-prompt values captured pre-save."""
    import torch

    path = Path(checkpoint) / "reload_probe.json"
    if not path.is_file():
        return {"available": False, "probe_count": 0, "max_probability_delta": None}
    probes = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(probes, list) or not probes:
        raise ValueError(f"Invalid bridge reload probe: {path}")
    cases: list[dict[str, Any]] = []
    expected: list[float] = []
    expected_raw: list[tuple[float, float]] = []
    for probe in probes:
        if not isinstance(probe, Mapping) or not _valid_messages(probe.get("messages")):
            raise ValueError(f"Invalid bridge reload probe row: {path}")
        probability = float(probe["probability_A_before_save"])
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"Invalid pre-save probability in {path}")
        cases.append({"case_id": str(probe["case_id"]), "messages": probe["messages"]})
        expected.append(probability)
        before_raw = (
            float(probe["logp_A_before_save"]),
            float(probe["logp_B_before_save"]),
        )
        if not all(math.isfinite(value) and value <= 0.0 for value in before_raw):
            raise ValueError(f"Invalid pre-save raw log probability in {path}")
        expected_raw.append(before_raw)
    with torch.inference_mode():
        legal, raw = differentiable_choice_log_probs(
            model,
            tokenizer,
            cases,
            max_length=max_length,
        )
    observed = legal.exp()[:, 0].detach().float().cpu().tolist()
    observed_raw = raw.detach().float().cpu().tolist()
    normalized_deltas = [
        abs(float(before) - float(after))
        for before, after in zip(expected, observed, strict=True)
    ]
    token_probability_deltas: list[float] = []
    legal_mass_deltas: list[float] = []
    for before, after in zip(expected_raw, observed_raw, strict=True):
        before_probabilities = [math.exp(value) for value in before]
        after_probabilities = [math.exp(float(value)) for value in after]
        token_probability_deltas.extend(
            abs(left - right)
            for left, right in zip(before_probabilities, after_probabilities, strict=True)
        )
        legal_mass_deltas.append(abs(sum(before_probabilities) - sum(after_probabilities)))
    maximum = max([*normalized_deltas, *token_probability_deltas, *legal_mass_deltas])
    return {
        "available": True,
        "probe_count": len(cases),
        "max_probability_delta": maximum,
        "max_normalized_probability_delta": max(normalized_deltas),
        "max_token_probability_delta": max(token_probability_deltas),
        "max_legal_choice_mass_delta": max(legal_mass_deltas),
        "mean_normalized_probability_delta": float(np.mean(normalized_deltas)),
    }


def evaluate_bridge_checkpoint(
    config: dict[str, Any],
    environment: BridgeEvaluationEnvironment,
    *,
    checkpoint: str | Path,
    arm: str,
    pair_seed: int,
    split: str = "dev",
    unlock_test: bool = False,
    base_policy: bool = False,
    evaluation_settings: Mapping[str, Any] | None = None,
    destination: str | Path | None = None,
) -> Path:
    """Evaluate one fixed checkpoint without reinforcement or post-choice feedback."""
    import gc
    import torch

    evaluation_started = time.monotonic()
    selected_arm = canonical_arm(arm)
    requested_split = split.strip().lower()
    if requested_split not in {"dev", "test"}:
        raise ValueError("Bridge evaluation split must be dev or test")
    if requested_split == "test" and not unlock_test:
        raise PermissionError("Locked bridge test assays require unlock_test=True")
    spec = config_bound_bridge_evaluation_spec(config, evaluation_settings)
    configured_evaluation_spec = json.loads(canonical_json(asdict(spec)))
    configured_evaluation_spec_sha256 = _evaluation_spec_hash(spec)
    checkpoint_path = Path(checkpoint).resolve()
    checkpoint_manifest, bridge_state = _verify_checkpoint_for_evaluation(
        config,
        environment,
        checkpoint_path,
        arm=selected_arm,
        pair_seed=pair_seed,
    )
    environment.load_state_dict(bridge_state["environment_state"])
    checkpoint_update = int(checkpoint_manifest["completed_updates"])
    checkpoint_adapter_hashes = {
        str(name): str(digest)
        for name, digest in checkpoint_manifest["file_sha256"].items()
        if str(name).startswith("adapter_") or str(name).endswith(".safetensors")
    }
    if not checkpoint_adapter_hashes:
        raise ValueError("Bridge checkpoint lacks adapter weight hashes")
    if base_policy and checkpoint_update != 0:
        raise ValueError("The unchanged-base negative control must anchor to checkpoint zero")
    policy_condition = (
        "unchanged_base"
        if base_policy
        else ("untrained_lora" if checkpoint_update == 0 else f"{selected_arm}_trained_lora")
    )
    policy_artifact = {
        "kind": "unchanged_base_model" if base_policy else "lora_adapter_checkpoint",
        "base_model_id": str(config["model"]["id"]),
        "base_model_revision": str(config["model"]["revision"]),
        "adapter_loaded": not base_policy,
        "loaded_adapter_file_sha256": None if base_policy else checkpoint_adapter_hashes,
        "anchor_checkpoint": str(checkpoint_path),
        "anchor_checkpoint_manifest_sha256": sha256_file(
            checkpoint_path / "checkpoint_manifest.json"
        ),
    }
    optimizer_metrics = checkpoint_manifest.get("optimizer_metrics")
    model_runtime_attestation = verify_model_runtime_attestation(
        config, checkpoint_manifest["model_runtime_attestation"]
    )
    acquisition_diagnostics = checkpoint_manifest["acquisition_diagnostics"]
    acquisition_gate_window_diagnostics = checkpoint_manifest[
        "acquisition_gate_window_diagnostics"
    ]
    configured_spec = BridgeTrainingSpec.from_config(config)
    configured_training_spec = json.loads(canonical_json(asdict(configured_spec)))
    configured_training_spec_sha256 = _spec_hash(configured_spec)
    if checkpoint_update > 0:
        if not isinstance(optimizer_metrics, Mapping) or not optimizer_metrics:
            raise ValueError("A trained bridge checkpoint lacks optimizer metrics")
        if not all(math.isfinite(float(value)) for value in optimizer_metrics.values()):
            raise ValueError("A trained bridge checkpoint has non-finite optimizer metrics")
    cases = validate_extinction_cases(
        environment.extinction_cases(
            split=requested_split,
            trajectory_seed=int(pair_seed),
            checkpoint_update=checkpoint_update,
        ),
        split=requested_split,
    )
    cfg_sha256 = config.get("_config_sha256") or config_hash(config)
    target = (
        Path(destination).resolve()
        if destination
        else output_root(config)
        / "bridge"
        / "evaluations"
        / f"{policy_condition}_seed{pair_seed}_step{checkpoint_update:06d}_{requested_split}.jsonl"
    )
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite bridge evaluation {target}")
    target.parent.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        # Bound the peak to this checkpoint's complete inference lifecycle. Reset
        # immediately before any tokenizer/model loading so the cost preflight sees
        # base weights, adapter loading, reload probes, scoring, and generation.
        torch.cuda.reset_peak_memory_stats()
    tokenizer_started = time.monotonic()
    tokenizer = load_tokenizer(config)
    tokenizer_wall_seconds = time.monotonic() - tokenizer_started
    model_load_started = time.monotonic()
    model = (
        load_base_model(config, training=False)
        if base_policy
        else load_adapter_model(config, checkpoint_path)
    )
    if base_policy:
        model_and_tokenizer_load_wall_seconds = (
            tokenizer_wall_seconds + time.monotonic() - model_load_started
        )
        base_validation_started = time.monotonic()
        model_runtime_contract = validate_loaded_base_runtime(
            config, model, tokenizer, model_runtime_attestation
        )
        model_and_tokenizer_load_wall_seconds += (
            time.monotonic() - base_validation_started
        )
        adapter_load_and_attestation_wall_seconds = 0.0
    else:
        model_and_tokenizer_load_wall_seconds = tokenizer_wall_seconds + float(
            getattr(model, "_ue_base_model_load_wall_seconds", 0.0)
        )
        adapter_attestation_started = time.monotonic()
        model_runtime_contract = validate_loaded_lora_runtime(
            config,
            model,
            tokenizer,
            getattr(model, "_ue_lora_target_inventory", None),
            model_runtime_attestation,
        )
        adapter_load_and_attestation_wall_seconds = float(
            getattr(model, "_ue_adapter_load_wall_seconds", 0.0)
        ) + (time.monotonic() - adapter_attestation_started)
    model.eval()
    reload_probe_started = time.monotonic()
    reload_diagnostic = (
        {"available": False, "probe_count": 0, "max_probability_delta": None}
        if base_policy
        else adapter_reload_diagnostic(
            model,
            tokenizer,
            checkpoint_path,
            max_length=int(config["model"]["max_length"]),
        )
    )
    adapter_reload_probe_wall_seconds = (
        0.0 if base_policy else time.monotonic() - reload_probe_started
    )
    if not base_policy and checkpoint_update > 0 and not reload_diagnostic["available"]:
        raise ValueError("A non-initial bridge checkpoint lacks its pre-save reload probe")
    gates = config.get("gates") or {}
    smoke_gates = gates.get("smoke", gates) if isinstance(gates, Mapping) else {}
    reload_delta_limit = (
        float(smoke_gates.get("adapter_reload_max_probability_delta", 0.001))
        if isinstance(smoke_gates, Mapping)
        else 0.001
    )
    reload_check = (
        None
        if base_policy or not reload_diagnostic["available"]
        else float(reload_diagnostic["max_probability_delta"]) <= reload_delta_limit
    )
    forced_scoring_started = time.monotonic()
    forced: dict[str, dict[str, float]] = {}
    for batch in _batches(cases, spec.batch_size):
        with torch.inference_mode():
            _, raw_scores = differentiable_choice_log_probs(
                model,
                tokenizer,
                list(batch),
                max_length=int(config["model"]["max_length"]),
            )
        for case, raw in zip(batch, raw_scores.detach().float().cpu(), strict=True):
            logp_a, logp_b = map(float, raw.tolist())
            diagnostics = legal_choice_diagnostics(logp_a, logp_b)
            forced[str(case["case_id"])] = {
                "logp_A": logp_a,
                "logp_B": logp_b,
                **diagnostics,
            }
    forced_scoring_wall_seconds = time.monotonic() - forced_scoring_started

    generation_cases = _generation_subset(cases, spec.generation_subset_size)
    generation_attestation = generation_subset_attestation(
        cases,
        generation_cases,
        requested_size=spec.generation_subset_size,
    )
    generation_case_ids = {str(case["case_id"]) for case in generation_cases}
    generation_started = time.monotonic()
    generated = _generate_unconstrained(
        model,
        tokenizer,
        generation_cases,
        batch_size=spec.generation_batch_size,
        max_length=int(config["model"]["max_length"]),
        max_new_tokens=spec.max_new_tokens,
    )
    generation_wall_seconds = time.monotonic() - generation_started
    peak_vram_bytes = (
        int(
            max(
                torch.cuda.max_memory_allocated(),
                torch.cuda.max_memory_reserved(),
            )
        )
        if torch.cuda.is_available()
        else 0
    )
    peak_vram_allocated_bytes = (
        int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    )
    peak_vram_reserved_bytes = (
        int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0
    )
    if set(generated) != generation_case_ids:
        raise RuntimeError("Unconstrained generation did not cover the attested case subset")
    rows: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["case_id"])
        forced_row = forced[case_id]
        predicted = "A" if forced_row["probability_A"] >= 0.5 else "B"
        generation = generated.get(case_id, {
            "generated_output": None,
            "parsed_action": None,
            "parse_status": "not_sampled",
        })
        row = {
            "schema_version": "1.0",
            "evidence_kind": "environment_grounded_bridge",
            "config_sha256": cfg_sha256,
            "environment_provenance": dict(checkpoint_manifest["environment_provenance"]),
            "initial_environment_state_sha256": checkpoint_manifest[
                "initial_environment_state_sha256"
            ],
            "checkpoint": str(checkpoint_path),
            "checkpoint_manifest_sha256": sha256_file(checkpoint_path / "checkpoint_manifest.json"),
            "checkpoint_adapter_file_sha256": checkpoint_adapter_hashes,
            "bridge_spec": configured_training_spec,
            "bridge_spec_sha256": configured_training_spec_sha256,
            "bridge_spec_source": "loaded_config_exact",
            "bridge_evaluation_spec": configured_evaluation_spec,
            "bridge_evaluation_spec_sha256": configured_evaluation_spec_sha256,
            "bridge_evaluation_spec_source": "loaded_config_exact",
            "generation_subset_attestation": generation_attestation,
            "generation_subset_selected": case_id in generation_case_ids,
            "checkpoint_update": checkpoint_update,
            "arm": "base" if base_policy else selected_arm,
            "checkpoint_arm": selected_arm,
            "policy_condition": policy_condition,
            "policy_artifact": policy_artifact,
            "model_runtime_attestation_sha256": model_runtime_attestation[
                "attestation_sha256"
            ],
            "model_runtime_contract": model_runtime_contract,
            "adapter_reload_probe_available": bool(reload_diagnostic["available"]),
            "adapter_reload_max_probability_delta": reload_diagnostic["max_probability_delta"],
            "adapter_reload_probability_check": reload_check,
            "checkpoint_optimizer_metrics": (
                dict(optimizer_metrics) if optimizer_metrics is not None else None
            ),
            "checkpoint_acquisition_diagnostics": acquisition_diagnostics,
            "checkpoint_acquisition_gate_window_diagnostics": (
                acquisition_gate_window_diagnostics
            ),
            "pair_seed": int(pair_seed),
            "case_id": case_id,
            "messages_sha256": hashlib.sha256(
                canonical_json(case["messages"]).encode("utf-8")
            ).hexdigest(),
            "split": requested_split,
            "condition": case.get("condition"),
            "cue_regime": case.get("cue_regime"),
            "renderer_id": case.get("renderer_id"),
            "base_template_id": case.get("base_template_id"),
            "intervention_template_id": case.get("intervention_template_id"),
            "world_id": case.get("world_id"),
            "pair_id": case.get("pair_id"),
            "paired_control_id": case.get("paired_control_id"),
            "baseline_id": case.get("baseline_id"),
            "pre_target_action": case["pre_target_action"],
            "intervention": dict(case["intervention"]),
            "expected_actions": dict(case.get("expected_actions") or {}),
            "predicted_action": predicted,
            **forced_row,
            **generation,
        }
        for objective, expected_action in row["expected_actions"].items():
            row[f"correct_{objective}"] = predicted == expected_action
        rows.append(row)
    generated_rows = [row for row in rows if row["generation_subset_selected"]]
    if any(row["parse_status"] == "not_sampled" for row in generated_rows) or any(
        row["parse_status"] != "not_sampled"
        or row["generated_output"] is not None
        or row["parsed_action"] is not None
        for row in rows
        if not row["generation_subset_selected"]
    ):
        raise RuntimeError("Generation outputs do not match the attested deterministic subset")
    parse_rate = (
        float(np.mean([row["parse_status"] == "exact" for row in generated_rows]))
        if generated_rows
        else float("nan")
    )
    legal_masses = [float(row["legal_choice_mass"]) for row in rows]
    per_condition: dict[str, dict[str, float | int]] = {}
    for condition in sorted({str(row["condition"]) for row in rows}):
        subset = [row for row in rows if str(row["condition"]) == condition]
        per_condition[condition] = {
            "count": len(subset),
            "mean_legal_choice_mass": float(np.mean([row["legal_choice_mass"] for row in subset])),
        }
    summary = {
        "schema_version": "1.0",
        "kind": "bridge_extinction_evaluation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": cfg_sha256,
        "bridge_evaluation_spec": configured_evaluation_spec,
        "bridge_evaluation_spec_sha256": configured_evaluation_spec_sha256,
        "bridge_evaluation_spec_source": "loaded_config_exact",
        "generation_subset_attestation": generation_attestation,
        "environment_provenance": dict(checkpoint_manifest["environment_provenance"]),
        "initial_environment_state_sha256": checkpoint_manifest[
            "initial_environment_state_sha256"
        ],
        "checkpoint": str(checkpoint_path),
        "checkpoint_manifest_sha256": sha256_file(checkpoint_path / "checkpoint_manifest.json"),
        "checkpoint_adapter_file_sha256": checkpoint_adapter_hashes,
        "bridge_spec": configured_training_spec,
        "bridge_spec_sha256": configured_training_spec_sha256,
        "bridge_spec_source": "loaded_config_exact",
        "checkpoint_update": checkpoint_update,
        "arm": "base" if base_policy else selected_arm,
        "checkpoint_arm": selected_arm,
        "policy_condition": policy_condition,
        "policy_artifact": policy_artifact,
        "model_runtime_attestation": model_runtime_attestation,
        "model_runtime_attestation_sha256": model_runtime_attestation[
            "attestation_sha256"
        ],
        "model_runtime_contract": model_runtime_contract,
        "pair_seed": int(pair_seed),
        "split": requested_split,
        "test_unlocked": bool(unlock_test),
        "record_count": len(rows),
        "generation_sample_count": len(generated_rows),
        "exact_parse_rate": parse_rate,
        "mean_legal_choice_mass": float(np.mean(legal_masses)),
        "minimum_legal_choice_mass": min(legal_masses),
        "adapter_reload_diagnostic": reload_diagnostic,
        "checkpoint_optimizer_metrics": (
            dict(optimizer_metrics) if optimizer_metrics is not None else None
        ),
        "checkpoint_acquisition_diagnostics": acquisition_diagnostics,
        "checkpoint_acquisition_gate_window_diagnostics": (
            acquisition_gate_window_diagnostics
        ),
        "adapter_reload_max_probability_delta": reload_delta_limit,
        "per_condition": per_condition,
        "checks": {
            "legal_choice_mass": min(legal_masses) >= spec.minimum_legal_choice_mass,
            "unconstrained_exact_parse": parse_rate >= spec.minimum_exact_parse_rate,
            "extinction_invariants": True,
            "adapter_reload_probability": reload_check,
        },
        "predictions_path": str(target),
        "predictions_sha256": None,
        "project_tree_sha256": project_hash(Path(config["_config_path"]).parent.parent),
        "runtime_environment": environment_snapshot(),
        "timing": None,
        "wall_seconds": None,
    }
    summary_target = target.with_suffix(".summary.json")
    nonce = f"{os.getpid()}.{time.time_ns()}"
    temporary_target = target.with_name(f".{target.name}.{nonce}.tmp")
    temporary_summary = summary_target.with_name(f".{summary_target.name}.{nonce}.tmp")
    try:
        write_finalize_started = time.monotonic()
        write_jsonl(temporary_target, rows)
        summary["predictions_sha256"] = sha256_file(temporary_target)
        write_finalize_wall_seconds = time.monotonic() - write_finalize_started
        timing = {
            "schema_version": "1.0",
            "model_and_tokenizer_load_wall_seconds": (
                model_and_tokenizer_load_wall_seconds
            ),
            "adapter_load_and_attestation_wall_seconds": (
                adapter_load_and_attestation_wall_seconds
            ),
            "adapter_reload_probe_wall_seconds": adapter_reload_probe_wall_seconds,
            "forced_scoring_wall_seconds": forced_scoring_wall_seconds,
            "forced_record_count": len(cases),
            "generation_wall_seconds": generation_wall_seconds,
            "generated_record_count": len(generation_cases),
            "peak_vram_bytes": peak_vram_bytes,
            "peak_vram_allocated_bytes": peak_vram_allocated_bytes,
            "peak_vram_reserved_bytes": peak_vram_reserved_bytes,
            "write_finalize_wall_seconds": write_finalize_wall_seconds,
            "total_wall_seconds": time.monotonic() - evaluation_started,
        }
        summary["timing"] = timing
        summary["wall_seconds"] = timing["total_wall_seconds"]
        write_json(temporary_summary, summary)
        # Commit the JSONL last: paid-run scripts use its existence as completion.
        os.replace(temporary_summary, summary_target)
        os.replace(temporary_target, target)
    except BaseException:
        temporary_target.unlink(missing_ok=True)
        temporary_summary.unlink(missing_ok=True)
        raise
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return target


def fixed_bridge_checkpoints(
    run_dir: str | Path, *, config: Mapping[str, Any] | None = None
) -> list[Path]:
    """Return every preregistered fixed checkpoint, failing on a partial series."""
    root = Path(run_dir).resolve()
    manifest_path = root / "bridge_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing bridge run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("state") != "COMPLETE":
        raise ValueError("Checkpoint-dynamics evaluation requires a COMPLETE bridge run")
    if config is not None:
        configured_spec = BridgeTrainingSpec.from_config(config)
        expected_spec = json.loads(canonical_json(asdict(configured_spec)))
        expected_spec_sha256 = _spec_hash(configured_spec)
        if (
            manifest.get("bridge_spec") != expected_spec
            or manifest.get("bridge_spec_sha256") != expected_spec_sha256
            or manifest.get("bridge_spec_source") != "loaded_config_exact"
        ):
            raise ValueError("Bridge run training spec is not exactly bound to the loaded config")
    semantics = manifest.get("optimizer_update_semantics")
    if not isinstance(semantics, Mapping):
        raise ValueError("Bridge manifest lacks optimizer-update semantics")
    updates = semantics.get("checkpoint_updates")
    if not isinstance(updates, list) or not updates:
        raise ValueError("Bridge manifest lacks a fixed checkpoint schedule")
    if config is not None and updates != list(configured_spec.checkpoint_updates):
        raise ValueError("Bridge run checkpoint schedule differs from the config-bound training spec")
    paths = [root / "checkpoints" / f"checkpoint-{int(update):06d}" for update in updates]
    missing = [str(path) for path in paths if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Fixed bridge checkpoints are missing: " + ", ".join(missing))
    return paths


def evaluate_unchanged_base_control(
    config: dict[str, Any],
    environment: BridgeEvaluationEnvironment,
    *,
    anchor_run_dir: str | Path,
    anchor_arm: str,
    pair_seed: int,
    split: str = "dev",
    dev_only: bool = True,
    unlock_test: bool = False,
    evaluation_settings: Mapping[str, Any] | None = None,
    destination: str | Path | None = None,
) -> Path:
    """Score the unchanged base on the exact checkpoint-zero extinction assay.

    Checkpoint zero evaluated normally is the untrained-LoRA control. This function
    bypasses PEFT entirely, supplying the distinct unchanged-base negative control.
    """
    requested_split = split.strip().lower()
    if dev_only and requested_split != "dev":
        raise PermissionError("This negative-control invocation is explicitly DEV-only")
    checkpoints = fixed_bridge_checkpoints(anchor_run_dir, config=config)
    initial = next(
        (path for path in checkpoints if path.name == "checkpoint-000000"),
        None,
    )
    if initial is None:
        raise FileNotFoundError("Bridge run lacks the checkpoint-zero control")
    return evaluate_bridge_checkpoint(
        config,
        environment,
        checkpoint=initial,
        arm=anchor_arm,
        pair_seed=pair_seed,
        split=requested_split,
        unlock_test=unlock_test,
        base_policy=True,
        evaluation_settings=evaluation_settings,
        destination=destination,
    )


def evaluate_bridge_run(
    config: dict[str, Any],
    environment: BridgeEvaluationEnvironment,
    *,
    run_dir: str | Path,
    arm: str,
    pair_seed: int,
    split: str = "dev",
    dev_only: bool = True,
    unlock_test: bool = False,
    evaluation_settings: Mapping[str, Any] | None = None,
    destination: str | Path | None = None,
) -> list[Path] | Path:
    """Evaluate the complete fixed-checkpoint series for learning dynamics.

    ``dev_only=True`` is the safe default used during iteration. Locked test access
    requires setting it false *and* passing ``unlock_test=True``.
    """
    requested_split = split.strip().lower()
    if dev_only and requested_split != "dev":
        raise PermissionError("This evaluation invocation is explicitly DEV-only")
    checkpoints = fixed_bridge_checkpoints(run_dir, config=config)
    fresh_environment_state = json.loads(canonical_json(dict(environment.state_dict())))
    if destination is None:
        outputs: list[Path] = []
        for checkpoint in checkpoints:
            environment.load_state_dict(fresh_environment_state)
            outputs.append(
                evaluate_bridge_checkpoint(
                    config,
                    environment,
                    checkpoint=checkpoint,
                    arm=arm,
                    pair_seed=pair_seed,
                    split=requested_split,
                    unlock_test=unlock_test,
                    evaluation_settings=evaluation_settings,
                )
            )
        return outputs

    target = Path(destination).resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite combined bridge evaluation {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".bridge-eval-", dir=target.parent))
    temporary_target = target.with_name(f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp")
    summary_target = target.with_suffix(".summary.json")
    temporary_summary = summary_target.with_name(
        f".{summary_target.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        rows: list[dict[str, Any]] = []
        checkpoint_summaries: list[dict[str, Any]] = []
        runtime_attestations: list[dict[str, Any]] = []
        runtime_contracts: list[dict[str, Any]] = []
        for checkpoint in checkpoints:
            environment.load_state_dict(fresh_environment_state)
            update = int(checkpoint.name.rsplit("-", 1)[1])
            part = staging / f"checkpoint-{update:06d}.jsonl"
            evaluate_bridge_checkpoint(
                config,
                environment,
                checkpoint=checkpoint,
                arm=arm,
                pair_seed=pair_seed,
                split=requested_split,
                unlock_test=unlock_test,
                evaluation_settings=evaluation_settings,
                destination=part,
            )
            rows.extend(dict(row) for row in read_jsonl(part))
            part_summary = json.loads(
                part.with_suffix(".summary.json").read_text(encoding="utf-8")
            )
            runtime_attestations.append(
                verify_model_runtime_attestation(
                    config, part_summary.pop("model_runtime_attestation")
                )
            )
            runtime_contracts.append(dict(part_summary["model_runtime_contract"]))
            part_summary["standalone_predictions_path"] = None
            part_summary["predictions_path"] = str(target)
            checkpoint_summaries.append(part_summary)
        if not runtime_attestations or any(
            value != runtime_attestations[0] for value in runtime_attestations[1:]
        ):
            raise ValueError("Checkpoint evaluations used different model runtimes")
        if any(value != runtime_contracts[0] for value in runtime_contracts[1:]):
            raise ValueError("Checkpoint evaluations used different compact model contracts")
        combined_write_started = time.monotonic()
        write_jsonl(temporary_target, rows)
        combined_write_wall_seconds = time.monotonic() - combined_write_started
        timing_fields = (
            "model_and_tokenizer_load_wall_seconds",
            "adapter_load_and_attestation_wall_seconds",
            "adapter_reload_probe_wall_seconds",
            "forced_scoring_wall_seconds",
            "forced_record_count",
            "generation_wall_seconds",
            "generated_record_count",
            "write_finalize_wall_seconds",
            "total_wall_seconds",
        )
        combined_timing: dict[str, Any] = {
            "schema_version": "1.0",
            **{
                field: sum(float(item["timing"][field]) for item in checkpoint_summaries)
                for field in timing_fields
                if not field.endswith("_count")
            },
            **{
                field: sum(int(item["timing"][field]) for item in checkpoint_summaries)
                for field in timing_fields
                if field.endswith("_count")
            },
            "per_checkpoint": [
                {
                    "checkpoint_update": int(item["checkpoint_update"]),
                    "timing": dict(item["timing"]),
                }
                for item in checkpoint_summaries
            ],
        }
        combined_timing["peak_vram_bytes"] = max(
            int(item["timing"]["peak_vram_bytes"])
            for item in checkpoint_summaries
        )
        combined_timing["peak_vram_allocated_bytes"] = max(
            int(item["timing"].get("peak_vram_allocated_bytes", 0))
            for item in checkpoint_summaries
        )
        combined_timing["peak_vram_reserved_bytes"] = max(
            int(item["timing"].get("peak_vram_reserved_bytes", 0))
            for item in checkpoint_summaries
        )
        combined_timing["write_finalize_wall_seconds"] += combined_write_wall_seconds
        combined_timing["total_wall_seconds"] += combined_write_wall_seconds
        combined_summary = {
            "schema_version": "1.0",
            "kind": "bridge_fixed_checkpoint_series",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": config.get("_config_sha256") or config_hash(config),
            "bridge_spec": json.loads(
                canonical_json(asdict(BridgeTrainingSpec.from_config(config)))
            ),
            "bridge_spec_sha256": _spec_hash(BridgeTrainingSpec.from_config(config)),
            "bridge_spec_source": "loaded_config_exact",
            "bridge_evaluation_spec": json.loads(
                canonical_json(asdict(config_bound_bridge_evaluation_spec(config)))
            ),
            "bridge_evaluation_spec_sha256": configured_bridge_evaluation_spec_sha256(
                config
            ),
            "bridge_evaluation_spec_source": "loaded_config_exact",
            "arm": canonical_arm(arm),
            "pair_seed": int(pair_seed),
            "split": requested_split,
            "test_unlocked": bool(unlock_test),
            "checkpoint_updates": [int(item["checkpoint_update"]) for item in checkpoint_summaries],
            "checkpoint_summaries": checkpoint_summaries,
            "model_runtime_attestation": runtime_attestations[0],
            "model_runtime_attestation_sha256": runtime_attestations[0][
                "attestation_sha256"
            ],
            "model_runtime_contract": runtime_contracts[0],
            "timing": combined_timing,
            "record_count": len(rows),
            "predictions_path": str(target),
            "predictions_sha256": sha256_file(temporary_target),
        }
        write_json(temporary_summary, combined_summary)
        os.replace(temporary_summary, summary_target)
        os.replace(temporary_target, target)
    except BaseException:
        temporary_target.unlink(missing_ok=True)
        temporary_summary.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return target
