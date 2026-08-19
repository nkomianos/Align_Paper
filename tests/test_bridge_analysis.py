from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

import under_extinction.bridge_analysis as bridge_analysis
from under_extinction.bridge_analysis import (
    _cluster_bootstrap_interval,
    _paired_ordinary_metrics,
    _validate_unchanged_base_control,
    _value_direction_gate_statistics,
    paired_effects,
    validate_bridge_predictions,
    verify_bridge_gate_report,
    write_bridge_analysis,
)
from under_extinction.bridge_env import build_bridge_data, load_bridge_environment
from under_extinction.bridge_evaluation import (
    LEGAL_CHOICE_LOG_MASS_TOLERANCE,
    BridgeEvaluationSpec,
    _generation_subset,
    configured_bridge_evaluation_spec_sha256,
    generation_subset_attestation,
)
from under_extinction.bridge_training import (
    BridgeTrainingSpec,
    acquisition_gate_window_diagnostics_summary,
    acquisition_diagnostics_summary,
    configured_bridge_spec_sha256,
    initialize_acquisition_gate_window_diagnostics,
    initialize_acquisition_diagnostics,
)
from under_extinction.config import load_config
from under_extinction.io import canonical_json, read_jsonl, sha256_file, write_json, write_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_ordinary_equivalence_reports_the_worst_seed_not_only_the_pool() -> None:
    rows = []
    for seed, genuine_probability, proxy_probability in (
        (11, 0.7, 0.7),
        (29, 0.9, 0.5),
    ):
        for arm, probability in (
            ("genuine", genuine_probability),
            ("proxy", proxy_probability),
        ):
            rows.append({
                "arm": arm,
                "pair_seed": seed,
                "checkpoint_update": 1,
                "case_id": "ordinary-1",
                "condition": "ordinary",
                "predicted_action": "A",
                "probability_A": probability,
            })
    result = _paired_ordinary_metrics(rows, 1)
    assert result["ordinary_probability_gap"] == pytest.approx(0.2)
    assert result["maximum_seed_mean_probability_gap"] == pytest.approx(0.4)
    assert result["by_seed"]["29"]["mean_abs_probability_A_gap"] == pytest.approx(0.4)


def test_ordinary_equivalence_reports_the_worst_seed_by_cue_cell() -> None:
    rows = []
    for cue, genuine_probability, proxy_probability in (
        ("semantic", 0.9, 0.5),
        ("neutral", 0.7, 0.7),
    ):
        for arm, probability in (
            ("genuine", genuine_probability),
            ("proxy", proxy_probability),
        ):
            rows.append({
                "arm": arm,
                "pair_seed": 11,
                "checkpoint_update": 1,
                "case_id": f"ordinary-{cue}",
                "condition": "ordinary",
                "cue_regime": cue,
                "predicted_action": "A",
                "probability_A": probability,
            })
    result = _paired_ordinary_metrics(rows, 1)
    assert result["maximum_seed_mean_probability_gap"] == pytest.approx(0.2)
    assert result["maximum_seed_cue_mean_probability_gap"] == pytest.approx(0.4)
    assert result["by_seed_cue_regime"]["11"]["semantic"][
        "mean_abs_probability_A_gap"
    ] == pytest.approx(0.4)


def test_value_direction_gate_uses_the_worst_direction_not_the_pooled_mean() -> None:
    config = {"bridge": {"cue_regimes": ["semantic", "neutral"]}}
    row = {
        "value_relevant_shift_by_cue_regime_direction": {
            cue: {"devalue_preferred": 0.40, "upvalue_nonpreferred": 0.00}
            for cue in ("semantic", "neutral")
        },
        "learning_induced_value_shift_by_cue_regime_direction": {
            cue: {"devalue_preferred": 0.20, "upvalue_nonpreferred": 0.00}
            for cue in ("semantic", "neutral")
        },
        "value_active_switch_accuracy_by_cue_regime_direction": {
            cue: {"devalue_preferred": 1.00, "upvalue_nonpreferred": 0.50}
            for cue in ("semantic", "neutral")
        },
        "value_switch_cell_count_by_cue_regime_direction": {
            cue: {"devalue_preferred": 8, "upvalue_nonpreferred": 8}
            for cue in ("semantic", "neutral")
        },
        "value_switch_role_count_by_cue_regime_direction": {
            cue: {"devalue_preferred": 2, "upvalue_nonpreferred": 2}
            for cue in ("semantic", "neutral")
        },
        "value_switch_renderer_count_by_cue_regime_direction": {
            cue: {"devalue_preferred": 2, "upvalue_nonpreferred": 2}
            for cue in ("semantic", "neutral")
        },
        "value_switch_action_count_by_cue_regime_direction": {
            cue: {"devalue_preferred": 2, "upvalue_nonpreferred": 2}
            for cue in ("semantic", "neutral")
        },
    }
    result = _value_direction_gate_statistics(config, [row])
    assert result["cells_complete"] is True
    assert result["minimum_relevant_shift"] == pytest.approx(0.0)
    assert result["minimum_learning_induced_shift"] == pytest.approx(0.0)
    assert result["minimum_active_switch_accuracy"] == pytest.approx(0.5)

    row["value_switch_action_count_by_cue_regime_direction"]["neutral"][
        "upvalue_nonpreferred"
    ] = 1
    assert _value_direction_gate_statistics(config, [row])["cells_complete"] is False


def test_paired_effects_preserve_value_direction_and_require_sign_matched_sham() -> None:
    common = {
        "arm": "genuine",
        "pair_seed": 11,
        "checkpoint_update": 1,
        "pre_target_action": "A",
        "world_id": "world-1",
        "renderer_id": "renderer-1",
        "cue_regime": "semantic",
        "expected_actions": {"genuine": "B", "proxy": "A"},
    }
    rows = [{
        **common,
        "case_id": "baseline",
        "probability_A": 0.90,
        "predicted_action": "A",
        "intervention": {
            "active": False,
            "mode": "none",
            "base_family": "baseline",
            "objective": "none",
            "role_assignment": "genuine_slot_1",
            "value_update_type": "not_applicable",
        },
    }]
    for direction in ("devalue_preferred", "upvalue_nonpreferred"):
        control_id = f"control-{direction}"
        rows.extend([
            {
                **common,
                "case_id": control_id,
                "probability_A": 0.90,
                "predicted_action": "A",
                "intervention": {
                    "active": False,
                    "mode": "sham",
                    "base_family": "value",
                    "objective": "genuine",
                    "role_assignment": "genuine_slot_1",
                    "value_update_type": direction,
                },
            },
            {
                **common,
                "case_id": f"active-{direction}",
                "paired_control_id": control_id,
                "baseline_id": "baseline",
                "probability_A": 0.20,
                "predicted_action": "B",
                "intervention": {
                    "active": True,
                    "mode": "switch",
                    "base_family": "value",
                    "objective": "genuine",
                    "role_assignment": "genuine_slot_1",
                    "value_update_type": direction,
                },
            },
        ])
    effects = paired_effects(rows)
    assert {row["value_update_type"] for row in effects} == {
        "devalue_preferred", "upvalue_nonpreferred"
    }

    mismatched = json.loads(json.dumps(rows))
    mismatched[1]["intervention"]["value_update_type"] = "upvalue_nonpreferred"
    with pytest.raises(ValueError, match="does not match value-update direction"):
        paired_effects(mismatched)


def test_causal_bootstrap_preserves_the_crossed_seed_world_design() -> None:
    rows = [
        {"pair_seed": seed, "world_id": world, "shift": value}
        for seed, values in ((11, (0.1, 0.2)), (29, (0.3, 0.4)))
        for world, value in zip(("world-a", "world-b"), values, strict=True)
    ]
    interval = _cluster_bootstrap_interval(
        rows, value_key="shift", replicates=100, random_seed=7
    )
    assert interval["crossed_seed_world_resampling"] is True
    assert interval["seed_count"] == 2
    assert interval["world_cluster_count"] == 2
    assert interval["estimate"] == pytest.approx(0.25)

    with pytest.raises(ValueError, match="identical world set"):
        _cluster_bootstrap_interval(
            rows[:-1], value_key="shift", replicates=10, random_seed=7
        )


def _score(action: str, legal_mass: float = 0.9) -> dict[str, float | str]:
    probability_a = 0.98 if action == "A" else 0.02
    probability_b = 1.0 - probability_a
    return {
        "probability_A": probability_a,
        "probability_B": probability_b,
        "logp_A": math.log(legal_mass * probability_a),
        "logp_B": math.log(legal_mass * probability_b),
        "legal_choice_mass": legal_mass,
        "predicted_action": action,
    }


def _score_target_probability(target_action: str, probability: float) -> dict[str, float | str]:
    probability_a = probability if target_action == "A" else 1.0 - probability
    probability_b = 1.0 - probability_a
    legal_mass = 0.9
    predicted = "A" if probability_a >= 0.5 else "B"
    return {
        "probability_A": probability_a,
        "probability_B": probability_b,
        "logp_A": math.log(legal_mass * probability_a),
        "logp_B": math.log(legal_mass * probability_b),
        "legal_choice_mass": legal_mass,
        "predicted_action": predicted,
    }


def _prediction_row(
    config: dict,
    provenance: dict,
    case: dict,
    *,
    arm: str,
    checkpoint: int,
) -> dict:
    action = str(case["pre_target_action"]) if checkpoint == 0 else str(case["expected_actions"][arm])
    initial_hashes = {"adapter_model.safetensors": "1" * 64}
    adapter_hashes = (
        initial_hashes
        if checkpoint == 0
        else {"adapter_model.safetensors": ("2" if arm == "genuine" else "3") * 64}
    )
    optimizer = None if checkpoint == 0 else {
        "loss": 0.1,
        "policy_loss": 0.1,
        "gradient_norm": 0.2,
        "rollout_reward_mean": 0.8,
        "learning_rate": 1e-5,
    }
    full_runtime_hash = "e" * 64
    runtime_contract = {
        "schema_version": "1.0",
        "full_attestation_sha256": full_runtime_hash,
        "model_id": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "loader_class": config["model"]["loader_class"],
        "model_type": config["model"]["expected_model_type"],
        "text_only": config["model"]["text_only"],
        "text_parameter_count": config["model"]["expected_text_parameter_count"],
        "layer_type_counts": config["model"]["expected_layer_type_counts"],
        "chat_template_kwargs": config["model"]["chat_template_kwargs"],
        "chat_template_kwargs_supported": True,
        "closed_reasoning_preamble_observed": True,
        "chat_template_sha256": "f" * 64,
        "deltanet_kernel_policy": config["model"]["delta_net_kernel_policy"],
        "deltanet_backend": "torch_fallback",
        "lora_per_target_module_count": config["training"][
            "expected_lora_target_counts"
        ],
        "lora_module_count": config["training"]["expected_lora_module_count"],
        "lora_trainable_parameter_count": config["training"][
            "expected_lora_trainable_parameter_count"
        ],
        "lora_inventory_sha256": "9" * 64,
    }
    runtime_contract["contract_sha256"] = hashlib.sha256(
        canonical_json(runtime_contract).encode("utf-8")
    ).hexdigest()
    diagnostics_state = initialize_acquisition_diagnostics(arm, config["bridge"]["cue_regimes"])
    if checkpoint > 0:
        # The smoke schedule has one 64-case update and a semantic-only corpus.
        diagnostics_state["cells"]["semantic"]["aligned"] = {
            "count": 48, "reward_sum": 48.0, "success_count": 48,
        }
        diagnostics_state["cells"]["semantic"]["diagnostic_conflict"] = {
            "count": 16, "reward_sum": 14.0, "success_count": 14,
        }
    gate_window_state = initialize_acquisition_gate_window_diagnostics(
        arm,
        config["bridge"]["cue_regimes"],
        window_updates=int(config["training"]["acquisition_gate_window_updates"]),
        samples_per_update=int(config["training"]["rollout_batch_size"]),
    )
    if checkpoint > 0:
        gate_window_state["completed_updates"] = checkpoint
        gate_window_state["updates"] = [{
            "completed_update": checkpoint,
            "cells": json.loads(json.dumps(diagnostics_state["cells"])),
        }]
    return {
        "schema_version": "1.0",
        "evidence_kind": "environment_grounded_bridge",
        "config_sha256": config["_config_sha256"],
        "bridge_spec": json.loads(
            canonical_json(BridgeTrainingSpec.from_config(config).__dict__)
        ),
        "bridge_spec_sha256": configured_bridge_spec_sha256(config),
        "bridge_spec_source": "loaded_config_exact",
        "bridge_evaluation_spec": json.loads(
            canonical_json(BridgeEvaluationSpec.from_config(config).__dict__)
        ),
        "bridge_evaluation_spec_sha256": (
            configured_bridge_evaluation_spec_sha256(config)
        ),
        "bridge_evaluation_spec_source": "loaded_config_exact",
        "model_runtime_attestation_sha256": full_runtime_hash,
        "model_runtime_contract": runtime_contract,
        "environment_provenance": provenance,
        "initial_environment_state_sha256": "a" * 64,
        "messages_sha256": hashlib.sha256(canonical_json(case["messages"]).encode()).hexdigest(),
        "checkpoint_adapter_file_sha256": adapter_hashes,
        "checkpoint": f"/synthetic/{arm}/checkpoint-{checkpoint:06d}",
        "checkpoint_manifest_sha256": "d" * 64,
        "checkpoint_update": checkpoint,
        "arm": arm,
        "checkpoint_arm": arm,
        "policy_condition": "untrained_lora" if checkpoint == 0 else f"{arm}_trained_lora",
        "policy_artifact": {
            "kind": "lora_adapter_checkpoint",
            "base_model_id": config["model"]["id"],
            "base_model_revision": config["model"]["revision"],
            "adapter_loaded": True,
            "loaded_adapter_file_sha256": adapter_hashes,
            "anchor_checkpoint": f"/synthetic/{arm}/checkpoint-{checkpoint:06d}",
            "anchor_checkpoint_manifest_sha256": "d" * 64,
        },
        "adapter_reload_probe_available": checkpoint > 0,
        "adapter_reload_max_probability_delta": 0.0 if checkpoint > 0 else None,
        "adapter_reload_probability_check": True if checkpoint > 0 else None,
        "checkpoint_optimizer_metrics": optimizer,
        "checkpoint_acquisition_diagnostics": acquisition_diagnostics_summary(
            diagnostics_state
        ),
        "checkpoint_acquisition_gate_window_diagnostics": (
            acquisition_gate_window_diagnostics_summary(gate_window_state)
        ),
        "pair_seed": 11,
        "case_id": case["case_id"],
        "split": "dev",
        "condition": case["condition"],
        "cue_regime": case["cue_regime"],
        "renderer_id": case["renderer_id"],
        "world_id": case["world_id"],
        "pair_id": case["pair_id"],
        "paired_control_id": case.get("paired_control_id"),
        "baseline_id": case.get("baseline_id"),
        "pre_target_action": case["pre_target_action"],
        "intervention": case["intervention"],
        "expected_actions": case["expected_actions"],
        "generated_output": action,
        "parsed_action": action,
        "parse_status": "exact",
        **_score(action),
    }


def _attest_generation(config: dict, rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int, int], list[dict]] = {}
    for row in rows:
        key = (row["arm"], row["pair_seed"], row["checkpoint_update"])
        groups.setdefault(key, []).append(row)
    requested = int(config["evaluation"]["generation_subset_size"])
    for group_rows in groups.values():
        selected = list(_generation_subset(group_rows, requested))
        selected_ids = {row["case_id"] for row in selected}
        attestation = generation_subset_attestation(
            group_rows, selected, requested_size=requested
        )
        for row in group_rows:
            is_selected = row["case_id"] in selected_ids
            row["generation_subset_attestation"] = attestation
            row["generation_subset_selected"] = is_selected
            if not is_selected:
                row.update({
                    "generated_output": None,
                    "parsed_action": None,
                    "parse_status": "not_sampled",
                })
    return rows


def test_unchanged_base_semantic_prior_is_reported_but_not_a_smoke_kill(
    tmp_path: Path,
) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "bridge_smoke.yaml")
    data_dir = tmp_path / "data"
    build_bridge_data(config, data_dir)
    environment = load_bridge_environment(config, data_dir)
    cases = list(
        environment.extinction_cases(
            split="dev", trajectory_seed=11, checkpoint_update=0
        )
    )
    provenance = dict(environment.provenance())
    primary_rows = [
        _prediction_row(config, provenance, case, arm="genuine", checkpoint=0)
        for case in cases
    ]
    _attest_generation(config, primary_rows)
    base_rows: list[dict] = []
    for primary, case in zip(primary_rows, cases, strict=True):
        intervention = case["intervention"]
        if (
            intervention.get("active") is True
            and intervention.get("mode") == "switch"
            and intervention.get("objective") == "genuine"
        ):
            primary.update(
                _score_target_probability(str(case["pre_target_action"]), 0.70)
            )
        base = json.loads(json.dumps(primary))
        base.update({"arm": "base", "policy_condition": "unchanged_base"})
        base["policy_artifact"].update({
            "kind": "unchanged_base_model",
            "adapter_loaded": False,
            "loaded_adapter_file_sha256": None,
        })
        base_rows.append(base)
    _attest_generation(config, base_rows)
    base_path = tmp_path / "selective_base.jsonl"
    write_jsonl(base_path, base_rows)

    result, _ = _validate_unchanged_base_control(
        config, base_path, primary_rows, split="dev"
    )
    assert result["integrity_pass"] is True
    assert result["fingerprint_complete"] is True
    assert result["maximum_channel_selectivity_gap"] > 0.05
    assert result["hard_gate_applicable"] is False
    assert result["maximum_hard_gate_channel_selectivity_gap"] is None
    assert result["scientific_fingerprint_pass"] is True
    assert result["pass"] is True


def test_unchanged_base_neutral_channel_selectivity_is_a_formal_kill(
    tmp_path: Path,
) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "bridge_pilot.yaml")
    data_dir = tmp_path / "data"
    build_bridge_data(config, data_dir)
    environment = load_bridge_environment(config, data_dir)
    cases = list(
        environment.extinction_cases(
            split="dev", trajectory_seed=11, checkpoint_update=0
        )
    )
    provenance = dict(environment.provenance())
    primary_rows = [
        _prediction_row(config, provenance, case, arm="genuine", checkpoint=0)
        for case in cases
    ]
    _attest_generation(config, primary_rows)
    base_rows: list[dict] = []
    for primary, case in zip(primary_rows, cases, strict=True):
        intervention = case["intervention"]
        if (
            case["cue_regime"] == "neutral"
            and intervention.get("active") is True
            and intervention.get("mode") == "switch"
            and intervention.get("objective") == "genuine"
        ):
            primary.update(
                _score_target_probability(str(case["pre_target_action"]), 0.70)
            )
        base = json.loads(json.dumps(primary))
        base.update({"arm": "base", "policy_condition": "unchanged_base"})
        base["policy_artifact"].update({
            "kind": "unchanged_base_model",
            "adapter_loaded": False,
            "loaded_adapter_file_sha256": None,
        })
        base_rows.append(base)
    _attest_generation(config, base_rows)
    base_path = tmp_path / "selective_neutral_base.jsonl"
    write_jsonl(base_path, base_rows)

    result, _ = _validate_unchanged_base_control(
        config, base_path, primary_rows, split="dev"
    )
    assert result["integrity_pass"] is True
    assert result["fingerprint_complete"] is True
    assert result["hard_gate_applicable"] is True
    assert result["hard_gate_cue_regimes"] == ["neutral"]
    assert result["maximum_hard_gate_channel_selectivity_gap"] > 0.05
    assert result["scientific_fingerprint_pass"] is False
    assert result["pass"] is False


def _write_runtime_sidecar(
    path: Path, rows: list[dict], config: dict, full_runtime: dict
) -> None:
    arms = {str(row["arm"]) for row in rows}
    seeds = {int(row["pair_seed"]) for row in rows}
    splits = {str(row["split"]) for row in rows}
    assert len(arms) == len(seeds) == len(splits) == 1
    checkpoints = sorted({int(row["checkpoint_update"]) for row in rows})
    combined = len(checkpoints) > 1
    summary = {
        "schema_version": "1.0",
        "kind": (
            "bridge_fixed_checkpoint_series"
            if combined
            else "bridge_extinction_evaluation"
        ),
        "config_sha256": config["_config_sha256"],
        "arm": next(iter(arms)),
        "pair_seed": next(iter(seeds)),
        "split": next(iter(splits)),
        "record_count": len(rows),
        "predictions_path": str(path),
        "predictions_sha256": sha256_file(path),
        "model_runtime_attestation": full_runtime,
        "model_runtime_attestation_sha256": full_runtime["attestation_sha256"],
        "model_runtime_contract": rows[0]["model_runtime_contract"],
    }
    if combined:
        summary["checkpoint_updates"] = checkpoints
    else:
        summary["checkpoint_update"] = checkpoints[0]
    write_json(path.with_suffix(".summary.json"), summary)


def test_bridge_analysis_accepts_only_complete_hash_bound_checkpoint_series(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "bridge_smoke.yaml")
    data_dir = tmp_path / "data"
    build_bridge_data(config, data_dir)
    environment = load_bridge_environment(config, data_dir)
    cases = list(environment.extinction_cases(split="dev", trajectory_seed=11, checkpoint_update=0))
    provenance = dict(environment.provenance())
    full_runtime = {"attestation_sha256": "e" * 64, "kind": "synthetic-test"}
    runtime_contract: dict | None = None

    def fake_verify_runtime(_config: dict, attestation: dict) -> dict:
        assert attestation == full_runtime
        return dict(attestation)

    def fake_compact_runtime(_config: dict, _attestation: dict) -> dict:
        assert runtime_contract is not None
        return dict(runtime_contract)

    monkeypatch.setattr(
        bridge_analysis, "verify_model_runtime_attestation", fake_verify_runtime
    )
    monkeypatch.setattr(
        bridge_analysis, "compact_model_runtime_contract", fake_compact_runtime
    )

    prediction_paths: list[Path] = []
    for arm in ("genuine", "proxy"):
        path = tmp_path / f"{arm}.jsonl"
        rows = _attest_generation(
            config,
            [
                _prediction_row(config, provenance, case, arm=arm, checkpoint=checkpoint)
                for checkpoint in (0, 1)
                for case in cases
            ],
        )
        write_jsonl(path, rows)
        runtime_contract = dict(rows[0]["model_runtime_contract"])
        _write_runtime_sidecar(path, rows, config, full_runtime)
        prediction_paths.append(path)

    base_path = tmp_path / "base.jsonl"
    base_rows: list[dict] = []
    for case in cases:
        row = _prediction_row(config, provenance, case, arm="genuine", checkpoint=0)
        row.update({"arm": "base", "policy_condition": "unchanged_base"})
        row["policy_artifact"] = {
            **row["policy_artifact"],
            "kind": "unchanged_base_model",
            "adapter_loaded": False,
            "loaded_adapter_file_sha256": None,
        }
        base_rows.append(row)
    _attest_generation(config, base_rows)
    write_jsonl(base_path, base_rows)
    _write_runtime_sidecar(base_path, base_rows, config, full_runtime)

    report_path = tmp_path / "report.json"
    write_bridge_analysis(
        config,
        prediction_paths,
        split="dev",
        destination=report_path,
        base_control_path=base_path,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["gates"]["smoke"]["pass"] is True
    assert report["unchanged_base_control"]["pass"] is True
    assert report["paired_initial_all_case_max_probability_gap"] == 0.0
    assert report["causal_uncertainty"]["replicates"] == 100
    assert report["acquisition_diagnostic_policy"] == {
        "continuation_gate_basis": "trailing_optimizer_updates",
        "window_updates": 1,
        "samples_per_update": 64,
        "cumulative_diagnostics_role": "learning_curve_evidence_only",
    }
    final_metrics = [
        row for row in report["checkpoint_metrics"] if row["checkpoint_update"] == 1
    ]
    assert all(
        row["cumulative_acquisition_diagnostics"]["overall"]["sample_count"] == 64
        and row["acquisition_gate_window_diagnostics"]["overall"]["sample_count"] == 64
        for row in final_metrics
    )
    assert verify_bridge_gate_report(config, report_path, required="smoke")["pass"] is True

    # Exact BF16 roundoff pattern from the first Qwen3.5-9B Stage 1 DEV run.
    # The scorer accepted this log-space mass under its frozen numerical
    # tolerance, so validation must accept the immutable serialized row too.
    tolerant_rows = [
        row
        for path in prediction_paths
        for row in read_jsonl(path)
    ]
    tolerant_row = next(
        row
        for row in tolerant_rows
        if row["arm"] == "genuine" and row["checkpoint_update"] == 1
    )
    tolerant_row.update(
        {
            "probability_A": 1.2098659719807352e-06,
            "probability_B": 0.999998790134028,
            "logp_A": -13.625000953674316,
            "logp_B": -1.1920922133867862e-06,
            "legal_choice_mass": 1.0000000177744908,
            "predicted_action": "B",
        }
    )
    validate_bridge_predictions(config, tolerant_rows, split="dev")

    # Stored mass alone is insufficient. This pair of likelihoods is outside
    # the scorer's log-space allowance, while its deliberately inconsistent
    # stored mass remains inside the allowance and within the old rel_tol=1e-5
    # comparison. The recomputed log mass must therefore fail closed.
    invalid_rows = json.loads(json.dumps(tolerant_rows))
    invalid_row = next(
        row
        for row in invalid_rows
        if row["arm"] == "genuine" and row["checkpoint_update"] == 1
    )
    invalid_log_mass = 1.49 * LEGAL_CHOICE_LOG_MASS_TOLERANCE
    invalid_row.update(
        {
            "probability_A": 0.5,
            "probability_B": 0.5,
            "logp_A": math.log(0.5) + invalid_log_mass,
            "logp_B": math.log(0.5) + invalid_log_mass,
            "legal_choice_mass": math.exp(
                0.5 * LEGAL_CHOICE_LOG_MASS_TOLERANCE
            ),
            "predicted_action": "A",
        }
    )
    with pytest.raises(ValueError, match="Legal choice sequence mass exceeds one"):
        validate_bridge_predictions(config, invalid_rows, split="dev")

    tampered_path = tmp_path / "genuine_tampered_spec.jsonl"
    tampered_rows = [dict(row) for row in read_jsonl(prediction_paths[0])]
    tampered_rows[0]["bridge_spec"]["learning_rate"] *= 2.0
    write_jsonl(tampered_path, tampered_rows)
    _write_runtime_sidecar(tampered_path, tampered_rows, config, full_runtime)
    with pytest.raises(ValueError, match="training spec is not exactly config-bound"):
        write_bridge_analysis(
            config,
            [tampered_path, prediction_paths[1]],
            split="dev",
            destination=tmp_path / "tampered_report.json",
            base_control_path=base_path,
        )

    tampered_evaluation_path = tmp_path / "genuine_tampered_evaluation.jsonl"
    tampered_evaluation_rows = [dict(row) for row in read_jsonl(prediction_paths[0])]
    tampered_evaluation_rows[0]["bridge_evaluation_spec"][
        "generation_subset_size"
    ] = 1
    write_jsonl(tampered_evaluation_path, tampered_evaluation_rows)
    _write_runtime_sidecar(
        tampered_evaluation_path, tampered_evaluation_rows, config, full_runtime
    )
    with pytest.raises(ValueError, match="evaluation spec is not exactly config-bound"):
        write_bridge_analysis(
            config,
            [tampered_evaluation_path, prediction_paths[1]],
            split="dev",
            destination=tmp_path / "tampered_evaluation_report.json",
            base_control_path=base_path,
        )

    tampered_runtime_path = tmp_path / "genuine_tampered_runtime.jsonl"
    tampered_runtime_rows = [dict(row) for row in read_jsonl(prediction_paths[0])]
    tampered_contract = tampered_runtime_rows[0]["model_runtime_contract"]
    tampered_contract["text_parameter_count"] -= 1
    unsigned_runtime_contract = dict(tampered_contract)
    unsigned_runtime_contract.pop("contract_sha256")
    tampered_contract["contract_sha256"] = hashlib.sha256(
        canonical_json(unsigned_runtime_contract).encode("utf-8")
    ).hexdigest()
    write_jsonl(tampered_runtime_path, tampered_runtime_rows)
    _write_runtime_sidecar(
        tampered_runtime_path, tampered_runtime_rows, config, full_runtime
    )
    with pytest.raises(ValueError, match="runtime proof mismatch"):
        write_bridge_analysis(
            config,
            [tampered_runtime_path, prediction_paths[1]],
            split="dev",
            destination=tmp_path / "tampered_runtime_report.json",
            base_control_path=base_path,
        )

    tampered_subset_path = tmp_path / "genuine_tampered_subset.jsonl"
    tampered_subset_rows = [dict(row) for row in read_jsonl(prediction_paths[0])]
    tampered_subset_rows[0]["generation_subset_selected"] = False
    write_jsonl(tampered_subset_path, tampered_subset_rows)
    _write_runtime_sidecar(
        tampered_subset_path, tampered_subset_rows, config, full_runtime
    )
    with pytest.raises(ValueError, match="Generation-subset selection flag mismatch"):
        write_bridge_analysis(
            config,
            [tampered_subset_path, prediction_paths[1]],
            split="dev",
            destination=tmp_path / "tampered_subset_report.json",
            base_control_path=base_path,
        )

    with prediction_paths[0].open("a", encoding="utf-8") as handle:
        handle.write("\n")
    verification = verify_bridge_gate_report(config, report_path, required="smoke")
    assert verification["pass"] is False
    assert any("changed" in failure for failure in verification["failures"])
