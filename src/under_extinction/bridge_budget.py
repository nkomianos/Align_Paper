"""Fail-closed exact-model smoke-to-Stage-1 time and cost projection."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .bridge_evaluation import (
    BridgeEvaluationSpec,
    configured_bridge_evaluation_spec_sha256,
)
from .bridge_training import BridgeTrainingSpec, configured_bridge_spec_sha256
from .config import output_root
from .io import canonical_json, sha256_file
from .modeling import (
    compact_model_runtime_contract,
    verify_model_runtime_attestation,
)


PROJECTION_FORMULA_VERSION = "qwen35_componentwise_max_v2"
WORKLOAD_PROFILE_KIND = "frozen_qwen35_token_workload_profile"


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def _number(
    value: Any, *, label: str, positive: bool = False, integer: bool = False
) -> float | int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(number) or (positive and number <= 0.0) or (
        not positive and number < 0.0
    ):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{label} must be finite and {qualifier}")
    if integer:
        if not number.is_integer():
            raise ValueError(f"{label} must be an integer")
        return int(number)
    return number


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label}: {path}") from exc
    return _mapping(value, label=label)


def _json_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _canonical(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _assert_throughput_contract(
    smoke_config: Mapping[str, Any], stage1_config: Mapping[str, Any]
) -> None:
    if smoke_config["model"] != stage1_config["model"]:
        raise ValueError("Smoke and Stage 1 must use the exact same model contract")
    if list(smoke_config["bridge"]["objectives"]) != ["genuine", "proxy"] or list(
        stage1_config["bridge"]["objectives"]
    ) != ["genuine", "proxy"]:
        raise ValueError("Projection requires exactly the paired genuine/proxy arms")
    if int(smoke_config["bridge"]["seeds"][0]) != int(
        stage1_config["bridge"]["seeds"][0]
    ):
        raise ValueError("Smoke and Stage 1 must use the same paired Stage-1 seed")
    if int(smoke_config["training"]["updates"]) != 1:
        raise ValueError("Projection requires exactly one smoke optimizer update")
    training_keys = (
        "algorithm",
        "rollout_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "entropy_coefficient",
        "kl_coefficient",
        "normalize_advantages",
        "max_grad_norm",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "lora_targets",
        "expected_lora_target_counts",
        "expected_lora_module_count",
        "expected_lora_trainable_parameter_count",
    )
    for key in training_keys:
        if smoke_config["training"].get(key) != stage1_config["training"].get(key):
            raise ValueError(f"Smoke/Stage-1 throughput contract differs for training.{key}")
    evaluation_keys = (
        "batch_size",
        "generation_subset_size",
        "generation_batch_size",
        "max_new_tokens",
    )
    for key in evaluation_keys:
        if smoke_config["evaluation"].get(key) != stage1_config["evaluation"].get(key):
            raise ValueError(f"Smoke/Stage-1 throughput contract differs for evaluation.{key}")


def _frozen_manifest(config: Mapping[str, Any], *, label: str) -> tuple[Path, dict[str, Any]]:
    path = output_root(dict(config)) / "data" / "MANIFEST.json"
    manifest = _read_json(path, label=f"{label} frozen-data manifest")
    if manifest.get("config_sha256") != config["_config_sha256"]:
        raise ValueError(f"{label} frozen data does not match its config")
    return path, manifest


def _load_workload_profile(
    smoke_config: Mapping[str, Any], stage1_config: Mapping[str, Any]
) -> tuple[dict[str, Any], Path, Path, Path]:
    """Load a locally computed profile without parsing formal DEV on the GPU."""
    project_root = Path(str(stage1_config["_config_path"])).resolve().parent.parent
    relative = Path(str(stage1_config["budget"]["preflight_workload_profile_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("The workload profile path must remain inside the project")
    profile_path = (project_root / relative).resolve()
    if not profile_path.is_relative_to(project_root):
        raise ValueError("The workload profile escapes the project root")
    profile = _read_json(profile_path, label="frozen token-workload profile")
    unsigned = dict(profile)
    claimed_hash = unsigned.pop("profile_sha256", None)
    if claimed_hash != _json_hash(unsigned):
        raise ValueError("Frozen token-workload profile self-hash mismatch")
    smoke_manifest_path, smoke_manifest = _frozen_manifest(
        smoke_config, label="Smoke"
    )
    stage1_manifest_path, stage1_manifest = _frozen_manifest(
        stage1_config, label="Stage 1"
    )
    expected_identity = {
        "schema_version": "1.0",
        "kind": WORKLOAD_PROFILE_KIND,
        "method": "exact_pinned_tokenizer_actual_microbatch_padding_v1",
        "model_id": stage1_config["model"]["id"],
        "model_revision": stage1_config["model"]["revision"],
        "smoke_config_sha256": smoke_config["_config_sha256"],
        "stage1_config_sha256": stage1_config["_config_sha256"],
        "smoke_data_manifest_sha256": sha256_file(smoke_manifest_path),
        "stage1_data_manifest_sha256": sha256_file(stage1_manifest_path),
    }
    for key, expected in expected_identity.items():
        if profile.get(key) != expected:
            raise ValueError(f"Frozen token-workload profile mismatch for {key}")
    chat_hash = profile.get("chat_template_sha256")
    if not isinstance(chat_hash, str) or len(chat_hash) != 64:
        raise ValueError("Frozen token-workload profile lacks a chat-template hash")

    smoke_spec = BridgeTrainingSpec.from_config(smoke_config)
    stage1_spec = BridgeTrainingSpec.from_config(stage1_config)
    cells = {
        "smoke_training": (1, smoke_spec.batch_size, smoke_spec.microbatch_size),
        "stage1_training": (
            stage1_spec.updates,
            stage1_spec.batch_size,
            stage1_spec.microbatch_size,
        ),
    }
    for name, expected in cells.items():
        cell = _mapping(profile.get(name), label=f"workload profile {name}")
        observed = (
            int(cell.get("updates_profiled", -1)),
            int(cell.get("rollout_batch_size", -1)),
            int(cell.get("microbatch_size", -1)),
        )
        if observed != expected:
            raise ValueError(f"Frozen token-workload profile schedule mismatch for {name}")
        for key in (
            "maximum_padded_prompt_token_positions_per_update",
            "maximum_prompt_tokens",
            "mean_prompt_tokens",
        ):
            _number(cell.get(key), label=f"{name}.{key}", positive=True)
    smoke_dev = _mapping(
        profile.get("smoke_development"), label="smoke development workload"
    )
    stage1_dev = _mapping(
        profile.get("stage1_development"), label="Stage-1 development workload"
    )
    expected_smoke_cases = int(smoke_manifest["assay_case_counts"]["dev"])
    expected_stage1_cases = int(stage1_manifest["assay_case_counts"]["dev"])
    if int(smoke_dev.get("case_count", -1)) != expected_smoke_cases or int(
        stage1_dev.get("case_count", -1)
    ) != expected_stage1_cases:
        raise ValueError("Frozen token-workload profile has stale DEV case counts")
    for name, cell in (("smoke_development", smoke_dev), ("stage1_development", stage1_dev)):
        for key in ("maximum_prompt_tokens", "mean_prompt_tokens"):
            _number(cell.get(key), label=f"{name}.{key}", positive=True)
    if max(
        int(profile["smoke_training"]["maximum_prompt_tokens"]),
        int(profile["stage1_training"]["maximum_prompt_tokens"]),
        int(smoke_dev["maximum_prompt_tokens"]),
        int(stage1_dev["maximum_prompt_tokens"]),
    ) + 1 > int(stage1_config["model"]["max_length"]):
        raise ValueError("Frozen prompts exceed model.max_length")
    profile["training_update_scale"] = max(
        1.0,
        float(
            profile["stage1_training"][
                "maximum_padded_prompt_token_positions_per_update"
            ]
        )
        / float(
            profile["smoke_training"][
                "maximum_padded_prompt_token_positions_per_update"
            ]
        ),
    )
    profile["evaluation_per_record_scale"] = max(
        1.0,
        float(stage1_dev["maximum_prompt_tokens"])
        / float(smoke_dev["maximum_prompt_tokens"]),
    )
    return _canonical(profile), profile_path, smoke_manifest_path, stage1_manifest_path


def _expected_runtime(
    config: Mapping[str, Any], report: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = report.get("model_runtime_attestation")
    if not isinstance(raw, Mapping):
        raise ValueError("Smoke report lacks the full model-runtime attestation")
    attestation = verify_model_runtime_attestation(config, raw)
    contract = compact_model_runtime_contract(config, attestation)
    if (
        report.get("model_runtime_attestation_sha256")
        != attestation["attestation_sha256"]
        or report.get("model_runtime_contract") != contract
    ):
        raise ValueError("Smoke report runtime attestation/contract mismatch")
    return attestation, contract


def _training_timing(
    path: Path,
    *,
    smoke_config: Mapping[str, Any],
    arm: str,
    seed: int,
    runtime_attestation: Mapping[str, Any],
) -> tuple[dict[str, float], int]:
    manifest = _read_json(path, label=f"{arm} smoke training manifest")
    spec = BridgeTrainingSpec.from_config(smoke_config)
    expected_spec = _canonical(asdict(spec))
    checkpoints = list(spec.checkpoint_updates)
    if (
        manifest.get("state") != "COMPLETE"
        or manifest.get("arm") != arm
        or int(manifest.get("pair_seed", -1)) != seed
        or manifest.get("config_sha256") != smoke_config["_config_sha256"]
        or manifest.get("bridge_spec") != expected_spec
        or manifest.get("bridge_spec_sha256")
        != configured_bridge_spec_sha256(smoke_config)
        or manifest.get("bridge_spec_source") != "loaded_config_exact"
        or manifest.get("model") != smoke_config["model"]
        or manifest.get("model_runtime_attestation") != dict(runtime_attestation)
        or int(manifest.get("completed_updates", -1)) != 1
    ):
        raise ValueError(f"Invalid {arm} smoke training provenance")
    semantics = _mapping(
        manifest.get("optimizer_update_semantics"), label=f"{arm} optimizer semantics"
    )
    if semantics.get("checkpoint_updates") != checkpoints:
        raise ValueError(f"{arm} smoke checkpoint schedule mismatch")
    timing = _mapping(manifest.get("timing"), label=f"{arm} smoke timing")
    result = _mapping(manifest.get("result"), label=f"{arm} smoke result")
    if result.get("timing") != timing:
        raise ValueError(f"{arm} result and manifest timings differ")
    required = {
        "schema_version",
        "model_and_tokenizer_load_wall_seconds",
        "adapter_setup_and_attestation_wall_seconds",
        "update_compute_wall_seconds_total",
        "rollout_wall_seconds_total",
        "gradient_forward_backward_wall_seconds_total",
        "optimizer_step_wall_seconds_total",
        "diagnostics_wall_seconds_total",
        "checkpoint_write_wall_seconds_total",
        "maximum_checkpoint_write_wall_seconds",
        "checkpoint_write_count",
        "reload_probe_wall_seconds_total",
        "maximum_reload_probe_wall_seconds",
        "reload_probe_count",
        "finalize_wall_seconds",
        "completed_update_count",
        "resume_count",
        "mean_update_compute_wall_seconds",
    }
    if not required <= set(timing) or timing.get("schema_version") != "1.0":
        raise ValueError(f"{arm} smoke timing schema is incomplete")
    if int(_number(timing["completed_update_count"], label="completed updates", integer=True)) != 1:
        raise ValueError("Cost projection requires one completed smoke update")
    checkpoint_count = int(
        _number(timing["checkpoint_write_count"], label="checkpoint count", integer=True)
    )
    reload_count = int(
        _number(timing["reload_probe_count"], label="reload count", integer=True)
    )
    if checkpoint_count != len(checkpoints) or reload_count != len(checkpoints) - 1:
        raise ValueError(f"{arm} smoke checkpoint/reload timing coverage is incomplete")
    values = {
        "model_load": float(
            _number(timing["model_and_tokenizer_load_wall_seconds"], label="model load", positive=True)
        ),
        "adapter_setup": float(
            _number(timing["adapter_setup_and_attestation_wall_seconds"], label="adapter setup", positive=True)
        ),
        "update": float(
            _number(timing["mean_update_compute_wall_seconds"], label="mean update", positive=True)
        ),
        "checkpoint": float(
            _number(timing["maximum_checkpoint_write_wall_seconds"], label="max checkpoint", positive=True)
        ),
        "reload_probe": float(
            _number(timing["maximum_reload_probe_wall_seconds"], label="max reload probe", positive=True)
        ),
        "finalize": float(
            _number(timing["finalize_wall_seconds"], label="finalize", positive=True)
        ),
    }
    update_total = float(
        _number(timing["update_compute_wall_seconds_total"], label="update total", positive=True)
    )
    if not math.isclose(values["update"], update_total, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"{arm} mean-update timing is inconsistent")
    checkpoint_total = float(
        _number(timing["checkpoint_write_wall_seconds_total"], label="checkpoint total", positive=True)
    )
    reload_total = float(
        _number(timing["reload_probe_wall_seconds_total"], label="reload total", positive=True)
    )
    if values["checkpoint"] > checkpoint_total * 1.000001 or values["reload_probe"] > reload_total * 1.000001:
        raise ValueError(f"{arm} timing maximum exceeds its total")
    sub_update_total = sum(
        float(_number(timing[key], label=key))
        for key in (
            "rollout_wall_seconds_total",
            "gradient_forward_backward_wall_seconds_total",
            "optimizer_step_wall_seconds_total",
            "diagnostics_wall_seconds_total",
        )
    )
    if sub_update_total > update_total * 1.02:
        raise ValueError(f"{arm} nested update timers exceed update wall time")
    wall = float(_number(result.get("wall_seconds"), label=f"{arm} wall time", positive=True))
    timed_total = (
        values["model_load"]
        + values["adapter_setup"]
        + update_total
        + checkpoint_total
        + reload_total
        + values["finalize"]
    )
    if timed_total > wall * 1.02:
        raise ValueError(f"{arm} training component timers exceed wall time")
    values["unattributed_residual"] = max(0.0, wall - timed_total)
    peak_vram = int(
        _number(result.get("peak_vram_bytes"), label=f"{arm} training peak VRAM", positive=True, integer=True)
    )
    return values, peak_vram


def _evaluation_part_timing(
    part: Mapping[str, Any], *, label: str, expected_records: int, expected_generated: int
) -> tuple[dict[str, float], int]:
    timing = _mapping(part.get("timing"), label=f"{label} timing")
    required = {
        "schema_version",
        "model_and_tokenizer_load_wall_seconds",
        "adapter_load_and_attestation_wall_seconds",
        "adapter_reload_probe_wall_seconds",
        "forced_scoring_wall_seconds",
        "forced_record_count",
        "generation_wall_seconds",
        "generated_record_count",
        "write_finalize_wall_seconds",
        "total_wall_seconds",
        "peak_vram_bytes",
    }
    if not required <= set(timing) or timing.get("schema_version") != "1.0":
        raise ValueError(f"{label} evaluation timing schema is incomplete")
    forced_count = int(
        _number(timing["forced_record_count"], label=f"{label} forced count", positive=True, integer=True)
    )
    generated_count = int(
        _number(timing["generated_record_count"], label=f"{label} generation count", positive=True, integer=True)
    )
    if forced_count != expected_records or generated_count != expected_generated:
        raise ValueError(f"{label} evaluation timing record counts are incomplete")
    model_load = float(
        _number(timing["model_and_tokenizer_load_wall_seconds"], label=f"{label} model load", positive=True)
    )
    adapter_setup = float(
        _number(timing["adapter_load_and_attestation_wall_seconds"], label=f"{label} adapter setup")
    )
    reload_probe = float(
        _number(timing["adapter_reload_probe_wall_seconds"], label=f"{label} reload probe")
    )
    forced = float(
        _number(timing["forced_scoring_wall_seconds"], label=f"{label} forced scoring", positive=True)
    )
    generation = float(
        _number(timing["generation_wall_seconds"], label=f"{label} generation", positive=True)
    )
    write = float(
        _number(timing["write_finalize_wall_seconds"], label=f"{label} write", positive=True)
    )
    total = float(
        _number(timing["total_wall_seconds"], label=f"{label} total", positive=True)
    )
    timed_total = model_load + adapter_setup + reload_probe + forced + generation + write
    if timed_total > total * 1.02:
        raise ValueError(f"{label} evaluation component timers exceed wall time")
    peak = int(
        _number(timing["peak_vram_bytes"], label=f"{label} evaluation peak VRAM", positive=True, integer=True)
    )
    return {
        "model_load": model_load,
        "adapter_setup": adapter_setup,
        "reload_probe": reload_probe,
        "forced_rate": forced / forced_count,
        "generation_rate": generation / generated_count,
        "write_rate": write / forced_count,
        "unattributed_residual": max(0.0, total - timed_total),
        "total": total,
    }, peak


def _validate_evaluation_summary(
    path: Path,
    *,
    predictions_path: Path,
    smoke_config: Mapping[str, Any],
    arm: str,
    seed: int,
    split: str,
    runtime_attestation: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
    base_policy: bool,
) -> tuple[list[dict[str, float]], float, list[int]]:
    summary = _read_json(path, label=f"{arm} evaluation summary")
    expected_spec = _canonical(asdict(BridgeEvaluationSpec.from_config(smoke_config)))
    expected_training_spec = _canonical(asdict(BridgeTrainingSpec.from_config(smoke_config)))
    expected_records = int(
        _read_json(
            output_root(dict(smoke_config)) / "data" / "MANIFEST.json",
            label="smoke data manifest",
        )["assay_case_counts"][split]
    )
    expected_generated = min(
        int(smoke_config["evaluation"]["generation_subset_size"]), expected_records
    )
    expected_arm = "base" if base_policy else arm
    if (
        summary.get("config_sha256") != smoke_config["_config_sha256"]
        or summary.get("arm") != expected_arm
        or int(summary.get("pair_seed", -1)) != seed
        or summary.get("split") != split
        or summary.get("predictions_sha256") != sha256_file(predictions_path)
        or not isinstance(summary.get("predictions_path"), str)
        or Path(str(summary["predictions_path"])).name != predictions_path.name
        or summary.get("bridge_evaluation_spec") != expected_spec
        or summary.get("bridge_evaluation_spec_sha256")
        != configured_bridge_evaluation_spec_sha256(smoke_config)
        or summary.get("bridge_evaluation_spec_source") != "loaded_config_exact"
        or summary.get("bridge_spec") != expected_training_spec
        or summary.get("bridge_spec_sha256") != configured_bridge_spec_sha256(smoke_config)
        or summary.get("bridge_spec_source") != "loaded_config_exact"
        or summary.get("model_runtime_attestation") != dict(runtime_attestation)
        or summary.get("model_runtime_attestation_sha256")
        != runtime_attestation["attestation_sha256"]
        or summary.get("model_runtime_contract") != dict(runtime_contract)
    ):
        raise ValueError(f"Invalid {arm} smoke evaluation provenance")
    expected_checkpoints = list(BridgeTrainingSpec.from_config(smoke_config).checkpoint_updates)
    parts: list[dict[str, Any]]
    combined_extra_write = 0.0
    if base_policy:
        if summary.get("kind") != "bridge_extinction_evaluation" or int(
            summary.get("checkpoint_update", -1)
        ) != 0:
            raise ValueError("Unchanged-base evaluation is not anchored at checkpoint zero")
        if int(summary.get("record_count", -1)) != expected_records:
            raise ValueError("Unchanged-base evaluation has the wrong record count")
        parts = [summary]
    else:
        if (
            summary.get("kind") != "bridge_fixed_checkpoint_series"
            or summary.get("checkpoint_updates") != expected_checkpoints
            or int(summary.get("record_count", -1))
            != expected_records * len(expected_checkpoints)
        ):
            raise ValueError(f"{arm} combined evaluation checkpoint series is incomplete")
        raw_parts = summary.get("checkpoint_summaries")
        if not isinstance(raw_parts, list) or len(raw_parts) != len(expected_checkpoints):
            raise ValueError(f"{arm} combined evaluation lacks checkpoint summaries")
        parts = [_mapping(value, label=f"{arm} checkpoint summary") for value in raw_parts]
        observed_updates = [int(part.get("checkpoint_update", -1)) for part in parts]
        if observed_updates != expected_checkpoints or len(set(observed_updates)) != len(observed_updates):
            raise ValueError(f"{arm} checkpoint summaries are duplicated or out of order")
        for part in parts:
            if (
                part.get("config_sha256") != smoke_config["_config_sha256"]
                or part.get("arm") != arm
                or int(part.get("pair_seed", -1)) != seed
                or part.get("split") != split
                or int(part.get("record_count", -1)) != expected_records
                or int(part.get("generation_sample_count", -1)) != expected_generated
                or part.get("model_runtime_attestation_sha256")
                != runtime_attestation["attestation_sha256"]
                or part.get("model_runtime_contract") != dict(runtime_contract)
            ):
                raise ValueError(f"{arm} checkpoint evaluation provenance is incomplete")
    timings: list[dict[str, float]] = []
    peaks: list[int] = []
    for part in parts:
        timing, peak = _evaluation_part_timing(
            part,
            label=f"{arm} checkpoint {part.get('checkpoint_update')}",
            expected_records=expected_records,
            expected_generated=expected_generated,
        )
        timings.append(timing)
        peaks.append(peak)
    if not base_policy:
        combined = _mapping(summary.get("timing"), label=f"{arm} combined timing")
        if int(combined.get("peak_vram_bytes", -1)) != max(peaks):
            raise ValueError(f"{arm} combined evaluation peak VRAM is not a maximum")
        for field in (
            "model_and_tokenizer_load_wall_seconds",
            "adapter_load_and_attestation_wall_seconds",
            "adapter_reload_probe_wall_seconds",
            "forced_scoring_wall_seconds",
            "forced_record_count",
            "generation_wall_seconds",
            "generated_record_count",
        ):
            expected_sum = sum(float(part["timing"][field]) for part in parts)
            observed_sum = float(_number(combined.get(field), label=f"combined {field}"))
            if not math.isclose(observed_sum, expected_sum, rel_tol=1e-9, abs_tol=1e-6):
                raise ValueError(f"{arm} combined timing is inconsistent for {field}")
        combined_write = float(
            _number(combined.get("write_finalize_wall_seconds"), label="combined write", positive=True)
        )
        part_write = sum(
            float(part["timing"]["write_finalize_wall_seconds"]) for part in parts
        )
        combined_extra_write = combined_write - part_write
        if combined_extra_write < -1e-6:
            raise ValueError(f"{arm} combined write timing is less than its parts")
        combined_extra_write = max(0.0, combined_extra_write)
        combined_total = float(
            _number(combined.get("total_wall_seconds"), label="combined total", positive=True)
        )
        expected_total = sum(timing["total"] for timing in timings) + combined_extra_write
        if not math.isclose(combined_total, expected_total, rel_tol=0.02, abs_tol=0.05):
            raise ValueError(f"{arm} combined evaluation total is inconsistent")
        per_checkpoint = combined.get("per_checkpoint")
        if not isinstance(per_checkpoint, list) or [
            int(value.get("checkpoint_update", -1)) for value in per_checkpoint
        ] != expected_checkpoints:
            raise ValueError(f"{arm} combined per-checkpoint timing is incomplete")
    combined_write_rate = combined_extra_write / max(1, expected_records * len(parts))
    return timings, combined_write_rate, peaks


def _maximum_components(values: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not values:
        raise ValueError("No measured timing components were supplied")
    return {key: max(float(value[key]) for value in values) for key in values[0]}


def _telemetry(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing preflight GPU telemetry: {path}")
    names: set[str] = set()
    totals: set[float] = set()
    used: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        columns = [column.strip() for column in line.split(",")]
        if len(columns) != 8:
            raise ValueError("Malformed preflight GPU telemetry row")
        names.add(columns[2])
        used.append(float(_number(columns[4], label="telemetry memory.used")))
        totals.add(float(_number(columns[5], label="telemetry memory.total", positive=True)))
    if not used or len(names) != 1 or len(totals) != 1 or "H100" not in next(iter(names)).upper():
        raise ValueError("GPU telemetry does not attest one H100 device")
    total_mib = next(iter(totals))
    return {
        "schema_version": "1.0",
        "sample_count": len(used),
        "gpu_name": next(iter(names)),
        "maximum_sampled_memory_used_mib": max(used),
        "device_memory_total_mib": total_mib,
        "device_memory_total_bytes": int(total_mib * 1024 * 1024),
    }


def _budget_environment(
    stage1_config: Mapping[str, Any], *, now_epoch: int, environment: Mapping[str, str]
) -> dict[str, float | int | str]:
    try:
        instance_id = str(environment["UE_INSTANCE_ID"]).strip()
        start = int(environment["UE_INSTANCE_START_EPOCH"])
        deadline = int(environment["UE_HARD_DEADLINE_EPOCH"])
        displayed_rate = float(environment["UE_HOURLY_USD"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "UE_INSTANCE_ID, UE_INSTANCE_START_EPOCH, UE_HARD_DEADLINE_EPOCH, "
            "and UE_HOURLY_USD are required for the paid cost projection"
        ) from exc
    if not instance_id:
        raise ValueError("UE_INSTANCE_ID must be nonempty")
    configured_rate = float(
        _number(stage1_config["budget"]["hourly_usd"], label="configured hourly rate", positive=True)
    )
    if not math.isclose(displayed_rate, configured_rate, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Displayed hourly price differs from the frozen config")
    reserve_seconds = int(stage1_config["budget"]["retrieval_reserve_minutes"]) * 60
    cutoff = deadline - reserve_seconds
    if start <= 0 or start > now_epoch or cutoff <= now_epoch:
        raise ValueError("Instance start/deadline leaves no valid paid compute window")
    return {
        "instance_id": instance_id,
        "instance_start_epoch": start,
        "provider_termination_deadline_epoch": deadline,
        "compute_cutoff_epoch": cutoff,
        "displayed_hourly_usd": displayed_rate,
        "configured_hourly_usd": configured_rate,
    }


def build_stage1_cost_projection(
    smoke_config: Mapping[str, Any],
    stage1_config: Mapping[str, Any],
    *,
    smoke_report_path: str | Path,
    now_epoch: int | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Project the exact formal workload from conservative measured components."""
    _assert_throughput_contract(smoke_config, stage1_config)
    now = int(time.time()) if now_epoch is None else int(now_epoch)
    environ = os.environ if environment is None else environment
    budget_environment = _budget_environment(
        stage1_config, now_epoch=now, environment=environ
    )
    report_path = Path(smoke_report_path).resolve()
    report = _read_json(report_path, label="smoke gate report")
    if (
        report.get("kind") != "same_environment_bridge_report"
        or report.get("config_sha256") != smoke_config["_config_sha256"]
        or report.get("split") != smoke_config["bridge"]["splits"]["development"]
        or (report.get("gates") or {}).get("smoke", {}).get("pass") is not True
    ):
        raise ValueError("Cost projection requires a passing smoke gate report")
    runtime_attestation, runtime_contract = _expected_runtime(smoke_config, report)
    if runtime_contract.get("deltanet_backend") != "torch_fallback":
        raise ValueError("Projection requires the frozen Transformers torch fallback")

    run_root = report_path.parent
    seed = int(smoke_config["bridge"]["seeds"][0])
    split = str(smoke_config["bridge"]["splits"]["development"])
    objectives = [str(value) for value in smoke_config["bridge"]["objectives"]]
    source_paths: list[Path] = [report_path]
    expected_report_inputs: dict[str, str] = {}
    training_components: list[dict[str, float]] = []
    training_peaks: list[int] = []
    evaluation_components: list[dict[str, float]] = []
    evaluation_peaks: list[int] = []
    combined_write_rates: list[float] = []
    for arm in objectives:
        manifest_path = run_root / "runs" / f"{arm}_seed{seed}" / "bridge_manifest.json"
        components, peak = _training_timing(
            manifest_path,
            smoke_config=smoke_config,
            arm=arm,
            seed=seed,
            runtime_attestation=runtime_attestation,
        )
        training_components.append(components)
        training_peaks.append(peak)
        source_paths.append(manifest_path)
        predictions = run_root / "predictions" / f"{arm}_seed{seed}_{split}.jsonl"
        summary_path = predictions.with_suffix(".summary.json")
        for source in (predictions, summary_path):
            if not source.is_file():
                raise FileNotFoundError(source)
            expected_report_inputs[str(source.resolve())] = sha256_file(source)
        timings, combined_rate, peaks = _validate_evaluation_summary(
            summary_path,
            predictions_path=predictions,
            smoke_config=smoke_config,
            arm=arm,
            seed=seed,
            split=split,
            runtime_attestation=runtime_attestation,
            runtime_contract=runtime_contract,
            base_policy=False,
        )
        evaluation_components.extend(timings)
        evaluation_peaks.extend(peaks)
        combined_write_rates.append(combined_rate)
        source_paths.extend((predictions, summary_path))

    base_predictions = run_root / "predictions" / f"unchanged_base_seed{seed}_{split}.jsonl"
    base_summary = base_predictions.with_suffix(".summary.json")
    for source in (base_predictions, base_summary):
        if not source.is_file():
            raise FileNotFoundError(source)
        expected_report_inputs[str(source.resolve())] = sha256_file(source)
    base_timings, _, base_peaks = _validate_evaluation_summary(
        base_summary,
        predictions_path=base_predictions,
        smoke_config=smoke_config,
        arm="base",
        seed=seed,
        split=split,
        runtime_attestation=runtime_attestation,
        runtime_contract=runtime_contract,
        base_policy=True,
    )
    if report.get("input_sha256") != expected_report_inputs:
        raise ValueError("Smoke report inputs do not exactly match the measured prediction evidence")
    source_paths.extend((base_predictions, base_summary))

    workload, profile_path, smoke_manifest_path, stage1_manifest_path = _load_workload_profile(
        smoke_config, stage1_config
    )
    if workload.get("chat_template_sha256") != runtime_contract.get(
        "chat_template_sha256"
    ):
        raise ValueError(
            "Frozen token-workload profile was measured with a different chat template"
        )
    source_paths.extend((profile_path, smoke_manifest_path, stage1_manifest_path))
    telemetry_path = run_root / "gpu_telemetry.csv"
    telemetry = _telemetry(telemetry_path)
    source_paths.append(telemetry_path)

    training = _maximum_components(training_components)
    adapter_evaluation = _maximum_components(evaluation_components)
    base_evaluation = base_timings[0]
    forced_rate = max(adapter_evaluation["forced_rate"], base_evaluation["forced_rate"])
    generation_rate = max(
        adapter_evaluation["generation_rate"], base_evaluation["generation_rate"]
    )
    write_rate = max(adapter_evaluation["write_rate"], base_evaluation["write_rate"])
    combined_write_rate = max(combined_write_rates)

    formal_spec = BridgeTrainingSpec.from_config(stage1_config)
    arm_count = len(stage1_config["bridge"]["objectives"])
    checkpoint_count = len(formal_spec.checkpoint_updates)
    formal_case_count = int(workload["stage1_development"]["case_count"])
    generation_count = min(
        int(stage1_config["evaluation"]["generation_subset_size"]), formal_case_count
    )
    train_scale = float(workload["training_update_scale"])
    eval_scale = float(workload["evaluation_per_record_scale"])
    training_per_arm = (
        training["model_load"]
        + training["adapter_setup"]
        + formal_spec.updates * training["update"] * train_scale
        + checkpoint_count * training["checkpoint"]
        + (checkpoint_count - 1) * training["reload_probe"]
        + training["finalize"]
        + training["unattributed_residual"]
    )
    evaluation_per_checkpoint = (
        adapter_evaluation["model_load"]
        + adapter_evaluation["adapter_setup"]
        + adapter_evaluation["reload_probe"]
        + formal_case_count * forced_rate * eval_scale
        + generation_count * generation_rate * eval_scale
        + formal_case_count * write_rate
        + adapter_evaluation["unattributed_residual"]
    )
    adapter_evaluation_per_arm = (
        checkpoint_count * evaluation_per_checkpoint
        + checkpoint_count * formal_case_count * combined_write_rate
    )
    base_evaluation_total = (
        base_evaluation["model_load"]
        + base_evaluation["adapter_setup"]
        + base_evaluation["reload_probe"]
        + formal_case_count * forced_rate * eval_scale
        + generation_count * generation_rate * eval_scale
        + formal_case_count * write_rate
        + base_evaluation["unattributed_residual"]
    )
    raw_training_seconds = arm_count * training_per_arm
    raw_evaluation_seconds = arm_count * adapter_evaluation_per_arm + base_evaluation_total
    control_plane_seconds = float(
        _number(
            stage1_config["budget"]["stage1_control_plane_minutes"],
            label="Stage-1 control-plane minutes",
            positive=True,
        )
    ) * 60.0
    raw_total_seconds = raw_training_seconds + raw_evaluation_seconds + control_plane_seconds
    margin = float(stage1_config["budget"]["preflight_projection_margin_fraction"])
    guarded_total_seconds = raw_total_seconds * (1.0 + margin)
    hourly = float(budget_environment["configured_hourly_usd"])
    guarded_cost = guarded_total_seconds * hourly / 3600.0
    usable_dollars = float(stage1_config["budget"]["nominal_usd"]) * (
        1.0 - float(stage1_config["budget"]["reserve_fraction"])
    )
    accrued_seconds = now - int(budget_environment["instance_start_epoch"])
    accrued_cost = accrued_seconds * hourly / 3600.0
    remaining_dollars = usable_dollars - accrued_cost
    remaining_seconds = int(budget_environment["compute_cutoff_epoch"]) - now
    train_ceiling = float(
        stage1_config["budget"]["stage1_train_minutes_per_objective"]
    ) * 60.0
    eval_ceiling = float(
        stage1_config["budget"]["stage1_eval_minutes_per_objective"]
    ) * 60.0
    raw_training_peak = max(training_peaks)
    raw_evaluation_peak = max(evaluation_peaks + base_peaks)
    projected_training_peak = int(math.ceil(raw_training_peak * train_scale))
    projected_evaluation_peak = int(math.ceil(raw_evaluation_peak * eval_scale))
    sampled_device_peak = int(
        math.ceil(float(telemetry["maximum_sampled_memory_used_mib"]) * 1024 * 1024)
    )
    maximum_peak = max(
        projected_training_peak,
        projected_evaluation_peak,
        sampled_device_peak,
    )
    memory_fraction = maximum_peak / int(telemetry["device_memory_total_bytes"])
    memory_limit = float(stage1_config["budget"]["preflight_max_peak_vram_fraction"])
    checks = {
        "within_remaining_compute_window": guarded_total_seconds <= remaining_seconds,
        "within_remaining_usable_dollars": guarded_cost <= remaining_dollars,
        "per_arm_training_within_command_ceiling": (
            training_per_arm * (1.0 + margin) <= train_ceiling
        ),
        "per_arm_evaluation_within_command_ceiling": (
            adapter_evaluation_per_arm * (1.0 + margin) <= eval_ceiling
        ),
        "base_evaluation_within_command_ceiling": (
            base_evaluation_total * (1.0 + margin) <= eval_ceiling
        ),
        "peak_vram_within_frozen_headroom": memory_fraction <= memory_limit,
    }
    source_hashes = {
        str(path.resolve()): sha256_file(path)
        for path in sorted(set(source_paths), key=lambda value: str(value.resolve()))
    }
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "formula_version": PROJECTION_FORMULA_VERSION,
        "created_at_epoch": now,
        "smoke_config_sha256": smoke_config["_config_sha256"],
        "stage1_config_sha256": stage1_config["_config_sha256"],
        "source_file_sha256": source_hashes,
        "runtime": {
            "model_runtime_attestation_sha256": runtime_attestation[
                "attestation_sha256"
            ],
            "deltanet_backend": runtime_contract["deltanet_backend"],
            "training_peak_vram_bytes": raw_training_peak,
            "evaluation_peak_vram_bytes": raw_evaluation_peak,
            "projected_training_peak_vram_bytes": projected_training_peak,
            "projected_evaluation_peak_vram_bytes": projected_evaluation_peak,
            "maximum_sampled_device_memory_used_bytes": sampled_device_peak,
            "maximum_peak_vram_bytes": maximum_peak,
            "peak_vram_fraction_of_device": memory_fraction,
            "peak_vram_fraction_limit": memory_limit,
            "gpu_telemetry": telemetry,
        },
        "formal_workload": {
            "arm_count": arm_count,
            "updates_per_arm": formal_spec.updates,
            "checkpoint_updates": list(formal_spec.checkpoint_updates),
            "adapter_evaluation_count": arm_count * checkpoint_count,
            "base_evaluation_count": 1,
            "development_case_count_per_evaluation": formal_case_count,
            "generation_case_count_per_evaluation": generation_count,
            "token_workload_profile": workload,
        },
        "observed_componentwise_max_seconds": {
            "training": training,
            "adapter_evaluation": adapter_evaluation,
            "base_evaluation": base_evaluation,
            "forced_seconds_per_record": forced_rate,
            "generation_seconds_per_record": generation_rate,
            "write_seconds_per_record": write_rate,
            "combined_write_seconds_per_record": combined_write_rate,
        },
        "projection": {
            "training_seconds_per_arm": training_per_arm,
            "evaluation_seconds_per_checkpoint": evaluation_per_checkpoint,
            "adapter_evaluation_seconds_per_arm": adapter_evaluation_per_arm,
            "base_evaluation_seconds": base_evaluation_total,
            "raw_training_seconds": raw_training_seconds,
            "raw_evaluation_seconds": raw_evaluation_seconds,
            "control_plane_seconds": control_plane_seconds,
            "raw_total_seconds": raw_total_seconds,
            "margin_fraction": margin,
            "guarded_total_seconds": guarded_total_seconds,
            "guarded_cost_usd": guarded_cost,
        },
        "budget_state": {
            **budget_environment,
            "evaluated_at_epoch": now,
            "usable_dollars_after_reserve": usable_dollars,
            "accrued_paid_seconds": accrued_seconds,
            "accrued_paid_cost_usd": accrued_cost,
            "remaining_compute_seconds": remaining_seconds,
            "remaining_usable_dollars": remaining_dollars,
            "train_command_ceiling_seconds_per_arm": train_ceiling,
            "eval_command_ceiling_seconds_per_call": eval_ceiling,
        },
        "checks": checks,
        "pass": all(checks.values()),
    }
    return _canonical(result)


def verify_stage1_cost_projection(
    stored: Mapping[str, Any],
    smoke_config: Mapping[str, Any],
    stage1_config: Mapping[str, Any],
    *,
    smoke_report_path: str | Path,
    now_epoch: int | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Recompute evidence/rates and re-check the shrinking live budget."""
    if stored.get("pass") is not True:
        raise ValueError("Stored Stage-1 cost projection was not a pass")
    current = build_stage1_cost_projection(
        smoke_config,
        stage1_config,
        smoke_report_path=smoke_report_path,
        now_epoch=now_epoch,
        environment=environment,
    )
    stored_budget = _mapping(
        stored.get("budget_state"), label="stored Stage-1 budget state"
    )
    current_budget = _mapping(
        current.get("budget_state"), label="current Stage-1 budget state"
    )
    for key in ("instance_id", "instance_start_epoch"):
        if stored_budget.get(key) != current_budget.get(key):
            raise ValueError(f"Stage-1 budget identity changed for {key}")
    for key in (
        "provider_termination_deadline_epoch",
        "compute_cutoff_epoch",
    ):
        if int(current_budget.get(key, 0)) > int(stored_budget.get(key, 0)):
            raise ValueError(f"Stage-1 budget deadline was expanded for {key}")
    immutable_keys = (
        "schema_version",
        "formula_version",
        "smoke_config_sha256",
        "stage1_config_sha256",
        "source_file_sha256",
        "runtime",
        "formal_workload",
        "observed_componentwise_max_seconds",
        "projection",
    )
    for key in immutable_keys:
        if stored.get(key) != current[key]:
            raise ValueError(f"Stored Stage-1 cost projection changed for {key}")
    if current["pass"] is not True:
        failures = [key for key, value in current["checks"].items() if not value]
        raise ValueError("Stage 1 no longer fits the paid budget: " + ", ".join(failures))
    return current
