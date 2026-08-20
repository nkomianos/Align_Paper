from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from under_extinction import dev_diag_evaluation as evaluation
from under_extinction import dev_diag
from under_extinction.io import canonical_json


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _case(index: int) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": "Return exactly A or B."},
        {"role": "user", "content": f"AUDIT case {index}: choose an option."},
    ]
    return {
        "schema_version": "DID-v1",
        "diagnostic_id": "stage1_dev_diag_v1",
        "generator_version": "did_v1.1.0",
        "case_id": f"case-{index:04d}",
        "label_pair_id": f"pair-{index // 2:04d}",
        "semantic_unit_id": f"unit-{index // 2:04d}",
        "namespace": "AUDIT",
        "split": "dev",
        "panel": "static",
        "cue_regime": "semantic",
        "renderer_id": "audit_matrix_v1",
        "role_assignment": "genuine_first",
        "updated_channel": "none",
        "family": "none",
        "mode": "static_conflict",
        "direction": "genuine_route_0",
        "time": "t0",
        "encoding": "CAN0",
        "query_head": "LATENT",
        "explicit_objective": "none",
        "label_permutation": "identity" if index % 2 == 0 else "swap",
        "module": "objective_retention",
        "messages": messages,
        "messages_sha256": hashlib.sha256(
            canonical_json(messages).encode("utf-8")
        ).hexdigest(),
        "causal_state": {"index": index},
        "query": {"kind": "latent"},
        "update_event": None,
    }


def test_frozen_core_spec_hash_and_evaluator_contract_agree():
    repository = Path(__file__).resolve().parents[1]
    spec_path = repository / "configs" / "stage1_dev_diag_v1.yaml"
    loaded = evaluation._core_load_spec(spec_path)
    public = {
        key: value for key, value in loaded.items() if not str(key).startswith("_")
    }
    assert loaded["_spec_sha256"] == _json_sha256(public)
    assert loaded["_spec_file_sha256"] == _sha256(spec_path)
    validated = evaluation._validate_frozen_spec(public)
    assert validated["generation"]["expected_total_prompt_count"] == 19_200
    assert validated["inference_contract"]["generation_subset_size"] == 256


@pytest.fixture
def synthetic_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    case_count = 4
    generation_count = 2
    subset_ids = [f"case-{case_count - 1:04d}", "case-0001"]
    deployment_root = tmp_path / evaluation.BUNDLE_ROOT
    (deployment_root / "project" / "configs").mkdir(parents=True)
    (deployment_root / "inputs" / "public").mkdir(parents=True)
    (deployment_root / "inputs" / "historical").mkdir(parents=True)
    (deployment_root / "inputs" / "checkpoints").mkdir(parents=True)
    _write_json(deployment_root / evaluation.BUNDLE_MANIFEST, {"fixture": True})
    bootstrap_attestation_path = tmp_path / "bootstrap_runtime_attestation.json"
    _write_json(bootstrap_attestation_path, {"fixture": True})
    bootstrap_verification = {
        "schema_version": "1.0",
        "kind": "did_v1_verified_bootstrap_binding",
        "attestation_sha256": "1" * 64,
        "bundle_manifest_sha256": "2" * 64,
        "bundle_inventory_sha256": "3" * 64,
        "source_identity_sha256": "4" * 64,
        "executing_project_payload_inventory_sha256": "5" * 64,
        "git_head_commit": "6" * 40,
        "gpu_uuid_query": ["fixture-gh200"],
        "checks": {"fixture_exact_binding": True},
    }
    monkeypatch.setattr(
        evaluation,
        "verify_dev_diag_bootstrap_attestation",
        lambda attestation, bundle: json.loads(canonical_json(bootstrap_verification)),
    )
    fixture_token_pairs = [[f"case-{index:04d}", 745] for index in range(case_count)]
    fixture_token_counts = dict(fixture_token_pairs)
    fixture_generation_token_pairs = [
        [case_id, fixture_token_counts[case_id]] for case_id in subset_ids
    ]
    monkeypatch.setattr(evaluation, "FROZEN_EXPECTED_PROMPT_COUNT", case_count)
    monkeypatch.setattr(
        evaluation,
        "FROZEN_INFERENCE_CONTRACT",
        {
            "training": False,
            "optimizer": False,
            "reward_feedback": False,
            "parameter_updates": False,
            "batch_size": 2,
            "generation_subset_size": generation_count,
            "generation_batch_size": 1,
            "max_new_tokens": 1,
            "checkpoint_zero_base_probability_tolerance": 0.001,
            "token_length_audit": {
                "required_before_model_load": True,
                "truncation_allowed": False,
                "expected_prompt_count": case_count,
                "expected_max_prompt_tokens": 745,
                "expected_all_prompt_token_counts_sha256": _json_sha256(
                    fixture_token_pairs
                ),
                "expected_generation_subset_token_counts_sha256": _json_sha256(
                    fixture_generation_token_pairs
                ),
                "expected_chat_template_sha256": "e" * 64,
                "expected_ordered_case_candidate_token_ids_sha256": "f" * 64,
                "require_exact_max": True,
                "require_choice_boundary_single_token": True,
                "require_prompt_plus_generation_within_max_length": True,
            },
        },
    )

    config_path = deployment_root / "project" / "configs" / "bridge_pilot.yaml"
    public_config: dict[str, Any] = {
        "bridge": {"paired_initialization": True},
        "model": {
            "id": "fixture/model",
            "revision": "1" * 40,
            "max_length": 32,
        },
    }
    config_path.write_text(yaml.safe_dump(public_config, sort_keys=False), encoding="utf-8")
    bridge_config = {**public_config, "_config_path": str(config_path)}
    config_canonical_sha256 = evaluation.config_hash(bridge_config)

    dev_path = deployment_root / "inputs" / "historical" / "dev.jsonl"
    dev_path.write_text('{"world_id":"dev-only"}\n', encoding="utf-8")
    data_manifest_path = deployment_root / "inputs" / "historical" / "MANIFEST.json"
    data_manifest = {
        "config_sha256": config_canonical_sha256,
        "counts": {"dev": 1},
        "files": {
            "dev": {
                "path": dev_path.name,
                "bytes": dev_path.stat().st_size,
                "sha256": _sha256(dev_path),
            },
            # Metadata may bind TEST, but the corresponding file is never an
            # evaluator input and must never be opened.
            "test": {"path": "test.jsonl", "bytes": 1, "sha256": "9" * 64},
        },
    }
    _write_json(data_manifest_path, data_manifest)

    runtime = {"attestation_sha256": "8" * 64, "fixture": True}
    historical = {
        "archive_sha256": "a" * 64,
        "stage1_release_tag": "fixture-stage1-failed",
        "stage1_release_commit": "b" * 40,
        "historical_training_commit": "c" * 40,
        "stage1_report_sha256": "d" * 64,
        "bridge_config_file_sha256": _sha256(config_path),
        "bridge_config_canonical_sha256": config_canonical_sha256,
        "data_manifest_sha256": _sha256(data_manifest_path),
        "dev_file_sha256": _sha256(dev_path),
        "pair_seed": 11,
        "initial_environment_state_sha256": "7" * 64,
        "model_runtime_attestation_sha256": runtime["attestation_sha256"],
        "model_id": public_config["model"]["id"],
        "model_revision": public_config["model"]["revision"],
        "bridge_spec_sha256": "6" * 64,
    }
    checkpoint_contracts: dict[str, dict[str, Any]] = {}
    checkpoint_paths: dict[str, Path] = {}
    for policy, arm, update in (
        ("checkpoint_zero", "genuine", 0),
        ("genuine_final", "genuine", 300),
        ("proxy_final", "proxy", 300),
    ):
        root = deployment_root / "inputs" / "checkpoints" / policy
        root.mkdir()
        adapter_config_path = root / "adapter_config.json"
        adapter_model_path = root / "adapter_model.safetensors"
        _write_json(adapter_config_path, {"peft_type": "LORA", "policy": policy})
        adapter_model_path.write_bytes(f"weights-{policy}".encode("utf-8"))
        manifest = {
            "schema_version": "1.0",
            "kind": "bridge_policy_checkpoint",
            "arm": arm,
            "completed_updates": update,
            "pair_seed": historical["pair_seed"],
            "config_sha256": historical["bridge_config_canonical_sha256"],
            "bridge_spec_sha256": historical["bridge_spec_sha256"],
            "initial_environment_state_sha256": historical[
                "initial_environment_state_sha256"
            ],
            "model_runtime_attestation": runtime,
            "model_runtime_attestation_sha256": runtime["attestation_sha256"],
            "environment_provenance": {
                "config_sha256": historical["bridge_config_canonical_sha256"],
                "data_manifest_sha256": historical["data_manifest_sha256"],
                "file_sha256": {"dev": historical["dev_file_sha256"]},
            },
            "file_sha256": {
                "adapter_config.json": _sha256(adapter_config_path),
                "adapter_model.safetensors": _sha256(adapter_model_path),
                "bridge_state.pt": "5" * 64,
            },
        }
        manifest_path = root / "checkpoint_manifest.json"
        _write_json(manifest_path, manifest)
        checkpoint_contracts[policy] = {
            "arm": arm,
            "update": update,
            "checkpoint_manifest_sha256": _sha256(manifest_path),
            "adapter_config_sha256": _sha256(adapter_config_path),
            "adapter_model_sha256": _sha256(adapter_model_path),
        }
        checkpoint_paths[policy] = root

    monkeypatch.setattr(evaluation, "FROZEN_HISTORICAL_CONTRACT", historical)
    monkeypatch.setattr(evaluation, "FROZEN_CHECKPOINT_CONTRACTS", checkpoint_contracts)
    monkeypatch.setattr(evaluation, "validate_bridge_config", lambda config: None)
    monkeypatch.setattr(
        evaluation,
        "verify_model_runtime_attestation",
        lambda config, value: dict(value),
    )

    parents = {
        "archive_sha256": historical["archive_sha256"],
        "stage1_release_tag": historical["stage1_release_tag"],
        "stage1_release_commit": historical["stage1_release_commit"],
        "historical_training_commit": historical["historical_training_commit"],
        "stage1_report_sha256": historical["stage1_report_sha256"],
        "bridge_config_file_sha256": historical["bridge_config_file_sha256"],
        "bridge_config_canonical_sha256": historical[
            "bridge_config_canonical_sha256"
        ],
        "data_manifest_sha256": historical["data_manifest_sha256"],
        "dev_file_sha256": historical["dev_file_sha256"],
        "pair_seed": historical["pair_seed"],
        "initial_environment_state_sha256": historical[
            "initial_environment_state_sha256"
        ],
        "model_runtime_attestation_sha256": historical[
            "model_runtime_attestation_sha256"
        ],
        **checkpoint_contracts,
    }
    spec = {
        "schema_version": "1.0",
        "kind": "bridge_posthoc_dev_diagnostic_spec",
        "scientific_status": evaluation.POSTHOC_STATUS,
        "diagnostic_id": "stage1_dev_diag_v1",
        "access_contract": {
            "allowed_split": "dev",
            "other_split_access": "forbidden",
            "locked_test_accessed": False,
        },
        "parents": parents,
        "model": {
            "id": historical["model_id"],
            "revision": historical["model_revision"],
            "max_length": 768,
            "choice_labels": ["A", "B"],
            "enable_thinking": False,
            "use_kernels": False,
        },
        "policy_conditions": list(evaluation.POLICY_CONDITIONS),
        "inference_contract": dict(evaluation.FROZEN_INFERENCE_CONTRACT),
        "generation": {
            "generator_version": "did_v1.1.0",
            "seed": 123,
            "audit_renderer_ids": ["audit_matrix_v1", "audit_routefile_v1"],
            "calibration_renderer_ids": ["cal_sheet_v1", "cal_log_v1"],
            "static": {
                "expected_prompt_count": case_count,
                "world_count": 2,
            },
            "update": {
                "expected_prompt_count": 0,
                "semantic_unit_count": 0,
            },
            "expected_total_prompt_count": case_count,
        },
        "analysis": {
            "minimum_legal_choice_mass": 0.5,
            "minimum_exact_parse_rate": 0.99,
        },
        "decision_contract": {
            "cannot_reverse_stage1": True,
            "cannot_open_locked_test": True,
            "cannot_authorize_replication": True,
        },
    }
    spec_path = deployment_root / "project" / "configs" / "stage1_dev_diag_v1.yaml"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    spec_sha256 = _json_sha256(spec)
    spec_file_sha256 = _sha256(spec_path)

    cases = [_case(index) for index in range(case_count)]
    cases_path = deployment_root / "inputs" / "public" / "cases.jsonl"
    _write_jsonl(cases_path, cases)
    answer_key_path = tmp_path / "answer_key.jsonl"
    _write_jsonl(
        answer_key_path,
        [
            {"case_id": case["case_id"], "correct_action": "A"}
            for case in cases
        ],
    )
    commitment_path = (
        deployment_root / "inputs" / "public" / "ANSWER_KEY_COMMITMENT.json"
    )
    commitment = {
        "schema_version": "DID-v1",
        "kind": "did_v1_hidden_answer_key_commitment",
        "diagnostic_id": "stage1_dev_diag_v1",
        "case_set_sha256": _sha256(cases_path),
        "answer_key_sha256": _sha256(answer_key_path),
        "record_count": case_count,
        "answer_key_external_to_model_visible_bundle": True,
    }
    _write_json(commitment_path, commitment)
    # Deliberately differ from prediction/full-case order.  Production subset
    # order is sorted-stratum/hash-rank order and has the same property.
    case_manifest = {
        "schema_version": "DID-v1",
        "kind": "did_v1_model_visible_case_manifest",
        "diagnostic_id": "stage1_dev_diag_v1",
        "scientific_status": evaluation.POSTHOC_STATUS,
        "generator_version": "did_v1.1.0",
        "split": "dev",
        "diagnostic_spec_sha256": spec_sha256,
        "diagnostic_spec_file_sha256": spec_file_sha256,
        "parents": parents,
        "verified_source_parent": {
            "data_manifest_sha256": _sha256(data_manifest_path),
            "dev_file_sha256": _sha256(dev_path),
            "dev_file_bytes": dev_path.stat().st_size,
            "dev_record_count": 1,
        },
        "access_contract": dict(spec["access_contract"]),
        "files": {
            "cases": {
                "path": cases_path.name,
                "sha256": _sha256(cases_path),
                "bytes": cases_path.stat().st_size,
                "count": case_count,
            },
            "answer_key_commitment": {
                "path": commitment_path.name,
                "sha256": _sha256(commitment_path),
                "bytes": commitment_path.stat().st_size,
            },
        },
        "answer_key": {
            "sha256": _sha256(answer_key_path),
            "count": case_count,
            "external": True,
            "path_disclosed": False,
        },
        "counts": {
            "static_prompts": case_count,
            "static_semantic_units": 2,
            "update_prompts": 0,
            "update_semantic_units": 0,
            "total_prompts": case_count,
        },
        "generation_subset": {
            "method": "four_hash_ranked_cases_per_panel_module_cue_renderer_label_stratum_v1",
            "size": generation_count,
            "ordered_case_ids_sha256": _json_sha256(subset_ids),
            "case_ids": subset_ids,
        },
        "template_provenance": {
            "audit_renderer_ids": list(spec["generation"]["audit_renderer_ids"]),
            "calibration_renderer_ids": list(
                spec["generation"]["calibration_renderer_ids"]
            ),
            "renderer_template_sha256": {
                renderer_id: "f" * 64
                for renderer_id in (
                    spec["generation"]["audit_renderer_ids"]
                    + spec["generation"]["calibration_renderer_ids"]
                )
            },
            "calibration_and_audit_renderer_sets_disjoint": True,
            "calibration_not_model_scored": True,
        },
        "locked_test_opened_or_parsed": False,
        "existing_dev_prompts_reused": False,
    }
    case_manifest_path = deployment_root / "inputs" / "public" / "MANIFEST.json"
    _write_json(case_manifest_path, case_manifest)

    monkeypatch.setattr(
        evaluation,
        "_core_load_spec",
        lambda path: {
            **yaml.safe_load(Path(path).read_text(encoding="utf-8")),
            "_spec_sha256": spec_sha256,
            "_spec_file_sha256": spec_file_sha256,
        },
    )
    monkeypatch.setattr(
        evaluation,
        "_core_validate_cases",
        lambda records, loaded_spec: [dict(row) for row in records],
    )
    monkeypatch.setattr(
        evaluation,
        "_core_generation_subset_case_ids",
        lambda records, loaded_spec: list(subset_ids),
    )

    kwargs = {
        "spec_path": spec_path,
        "case_manifest_path": case_manifest_path,
        "cases_path": cases_path,
        "answer_key_commitment_path": commitment_path,
        "data_manifest_path": data_manifest_path,
        "dev_data_path": dev_path,
        "checkpoint_zero": checkpoint_paths["checkpoint_zero"],
        "genuine_final": checkpoint_paths["genuine_final"],
        "proxy_final": checkpoint_paths["proxy_final"],
        "deployment_root": deployment_root,
        "bootstrap_attestation_path": bootstrap_attestation_path,
    }
    return bridge_config, kwargs, tmp_path


def _completed_verifier_kwargs(kwargs: dict[str, Any]) -> dict[str, Path]:
    return {
        "spec_path": Path(kwargs["spec_path"]),
        "case_manifest_path": Path(kwargs["case_manifest_path"]),
        "cases_path": Path(kwargs["cases_path"]),
        "answer_key_commitment_path": Path(
            kwargs["answer_key_commitment_path"]
        ),
        "deployment_root": Path(kwargs["deployment_root"]),
        "bootstrap_attestation_path": Path(
            kwargs["bootstrap_attestation_path"]
        ),
    }


def _fixture_token_audit(
    cases: list[dict[str, Any]], generation_case_ids: list[str]
) -> evaluation.TokenLengthAudit:
    pairs = [[case["case_id"], 745] for case in cases]
    generation_count = int(
        evaluation.FROZEN_INFERENCE_CONTRACT["generation_subset_size"]
    )
    by_id = {case_id: count for case_id, count in pairs}
    assert len(generation_case_ids) == generation_count
    generation_pairs = [[case_id, by_id[case_id]] for case_id in generation_case_ids]
    report = {
        "schema_version": "1.0",
        "kind": "did_v1_no_truncation_tokenizer_audit",
        "diagnostic_id": "stage1_dev_diag_v1",
        "model_id": evaluation.FROZEN_HISTORICAL_CONTRACT["model_id"],
        "model_revision": evaluation.FROZEN_HISTORICAL_CONTRACT["model_revision"],
        "chat_template_sha256": evaluation.FROZEN_INFERENCE_CONTRACT[
            "token_length_audit"
        ]["expected_chat_template_sha256"],
        "max_length": 768,
        "max_new_tokens": 1,
        "truncation_allowed": False,
        "expected_max_prompt_tokens": 745,
        "observed_max_prompt_tokens": 745,
        "all_prompts": {
            "count": len(pairs),
            "maximum_prompt_tokens": 745,
            "ordered_case_token_counts_sha256": _json_sha256(pairs),
        },
        "generation_subset": {
            "count": generation_count,
            "ordered_case_token_counts_sha256": _json_sha256(generation_pairs),
        },
        "candidate_boundary": {
            "labels": ["A", "B"],
            "single_token_for_every_prompt": True,
            "distinct_for_every_prompt": True,
            "ordered_case_candidate_token_ids_sha256": "f" * 64,
        },
        "checks": {
            "full_prompt_grid_audited": True,
            "generation_subset_audited": True,
            "exact_maximum_matches": True,
            "all_prompt_counts_hash_matches": True,
            "generation_subset_counts_hash_matches": True,
            "chat_template_hash_matches": True,
            "ordered_candidate_token_ids_hash_matches": True,
            "every_prompt_within_max_length": True,
            "every_prompt_plus_generation_within_max_length": True,
            "truncation_disabled": True,
        },
    }
    return evaluation.TokenLengthAudit(
        report=report,
        report_sha256=_json_sha256(report),
        prompt_token_counts={case["case_id"]: 745 for case in cases},
    )


def test_verifier_uses_only_exact_allowlist_and_never_requires_bridge_state(
    synthetic_inputs, monkeypatch: pytest.MonkeyPatch
):
    bridge_config, kwargs, tmp_path = synthetic_inputs
    locked_test = tmp_path / "test.jsonl"
    locked_test.write_text("THIS MUST NOT BE OPENED", encoding="utf-8")
    original_open = Path.open

    def guarded_open(path: Path, *args, **open_kwargs):
        if path.resolve() == locked_test.resolve():
            raise AssertionError("locked TEST was opened")
        return original_open(path, *args, **open_kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    verified = evaluation.verify_dev_diagnostic_inputs(bridge_config, **kwargs)
    access = verified.bindings["scientific_input_access"]
    assert access["opened_equals_allowlist"] is True
    assert access["locked_test_accessed"] is False
    assert access["dev_content_parsed"] is False
    assert all(Path(path).name != "test.jsonl" for path in access["opened_paths"])
    assert all(checkpoint.manifest["file_sha256"].get("bridge_state.pt") for checkpoint in verified.checkpoints.values())
    assert not any((checkpoint.path / "bridge_state.pt").exists() for checkpoint in verified.checkpoints.values())


def test_verifier_rejects_adapter_tampering(synthetic_inputs):
    bridge_config, kwargs, _ = synthetic_inputs
    adapter = Path(kwargs["proxy_final"]) / "adapter_model.safetensors"
    adapter.write_bytes(adapter.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="proxy_final differs from frozen adapter_model_sha256"):
        evaluation.verify_dev_diagnostic_inputs(bridge_config, **kwargs)


def test_verifier_rejects_malformed_answer_key_commitment(synthetic_inputs):
    bridge_config, kwargs, _ = synthetic_inputs
    commitment_path = Path(kwargs["answer_key_commitment_path"])
    commitment = json.loads(commitment_path.read_text(encoding="utf-8"))
    commitment["uncommitted_extension"] = True
    _write_json(commitment_path, commitment)
    manifest_path = Path(kwargs["case_manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["answer_key_commitment"]["sha256"] = _sha256(
        commitment_path
    )
    manifest["files"]["answer_key_commitment"]["bytes"] = (
        commitment_path.stat().st_size
    )
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="answer-key commitment.*wrong fields"):
        evaluation.verify_dev_diagnostic_inputs(bridge_config, **kwargs)


@pytest.mark.parametrize("leak_kind", ("top_level_extra", "nested_answer"))
def test_verifier_rejects_public_manifest_extensions_and_nested_answer_material(
    synthetic_inputs, leak_kind: str
):
    bridge_config, kwargs, _ = synthetic_inputs
    manifest_path = Path(kwargs["case_manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if leak_kind == "top_level_extra":
        manifest["unregistered_notes"] = "not part of the frozen public schema"
        expected = "wrong fields"
    else:
        manifest["template_provenance"]["expected_by_policy"] = {
            "case-0000": {"genuine_final": "A"}
        }
        expected = "answer/target material"
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match=expected):
        evaluation.verify_dev_diagnostic_inputs(bridge_config, **kwargs)


def test_evaluator_rejects_historical_parent_drift(synthetic_inputs):
    _, kwargs, _ = synthetic_inputs
    spec = yaml.safe_load(Path(kwargs["spec_path"]).read_text(encoding="utf-8"))
    spec["parents"]["historical_training_commit"] = "e" * 40
    with pytest.raises(
        ValueError,
        match="historical parent differs for historical_training_commit",
    ):
        evaluation._validate_frozen_spec(spec)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "logp_A",
            2.0 * evaluation.LEGAL_CHOICE_LOG_MASS_TOLERANCE,
            "raw token log probability",
        ),
        (
            "log_legal_choice_mass",
            2.0 * evaluation.LEGAL_CHOICE_LOG_MASS_TOLERANCE,
            "legal-choice log mass",
        ),
        (
            "legal_choice_mass",
            1.0 + 2.0 * evaluation.LEGAL_CHOICE_LOG_MASS_TOLERANCE,
            "legal-choice mass outside",
        ),
        ("legal_choice_mass", -1e-9, "legal-choice mass outside"),
    ],
)
def test_choice_score_validator_rejects_impossible_probability_bounds(
    field: str, value: float, message: str
):
    score = {
        "logp_A": math.log(0.6),
        "logp_B": math.log(0.4),
        "probability_A": 0.6,
        "probability_B": 0.4,
        "log_legal_choice_mass": 0.0,
        "legal_choice_mass": 1.0,
    }
    score[field] = value
    with pytest.raises(ValueError, match=message):
        evaluation._validate_choice_score_record(score, label="fixture score")


class _FakeModel(torch.nn.Module):
    def __init__(self, *, frozen: bool) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1), requires_grad=not frozen)


class _CharacterTokenizer:
    """Deterministic tokenizer double whose token count equals character count."""

    chat_template = "fixture-character-template"
    pad_token_id = 0
    eos_token_id = 3

    def __init__(self) -> None:
        self.padding_side = "left"
        self.truncation_side = "left"
        self.truncation_arguments: list[bool | None] = []

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    ):
        assert tokenize is False
        assert add_generation_prompt is True
        return messages[-1]["content"]

    def __call__(
        self,
        texts,
        *,
        add_special_tokens=False,
        truncation=None,
        padding=False,
        return_tensors=None,
        **kwargs,
    ):
        assert add_special_tokens is False
        self.truncation_arguments.append(truncation)
        if truncation is True:
            raise AssertionError("test tokenizer was asked to truncate")
        is_batch = isinstance(texts, list)
        values = texts if is_batch else [texts]
        rows = [[ord(character) for character in text] for text in values]
        if return_tensors is None:
            return {"input_ids": rows if is_batch else rows[0]}
        assert return_tensors == "pt" and padding is True
        width = max(map(len, rows))
        input_ids = torch.full((len(rows), width), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for index, row in enumerate(rows):
            offset = width - len(row)
            input_ids[index, offset:] = torch.tensor(row, dtype=torch.long)
            attention_mask[index, offset:] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def batch_decode(self, rows, **kwargs):
        return [
            "".join(chr(int(token)) for token in row if int(token) != self.pad_token_id)
            for row in rows
        ]


class _CandidateDriftTokenizer(_CharacterTokenizer):
    """Keep every length fixed while changing only the A/B suffix token IDs."""

    def __call__(self, texts, *args, **kwargs):
        encoded = super().__call__(texts, *args, **kwargs)
        if (
            kwargs.get("return_tensors") is None
            and isinstance(texts, str)
            and texts.endswith(("A", "B"))
        ):
            encoded["input_ids"][-1] += 1_000
        return encoded


class _GeneratingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def generate(self, input_ids, **kwargs):
        continuation = torch.full(
            (input_ids.shape[0], 1), ord("A"), dtype=input_ids.dtype
        )
        return torch.cat([input_ids, continuation], dim=1)


def _length_case(case_id: str, length: int) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": "ignored by fixture template"},
        {"role": "user", "content": "x" * length},
    ]
    return {"case_id": case_id, "messages": messages}


def _length_spec(cases: list[dict[str, Any]], subset: list[str]) -> dict[str, Any]:
    pairs = [[case["case_id"], len(case["messages"][-1]["content"])] for case in cases]
    by_id = {case_id: length for case_id, length in pairs}
    subset_pairs = [[case_id, by_id[case_id]] for case_id in subset]
    candidate_rows = [[case["case_id"], ord("A"), ord("B")] for case in cases]
    return {
        "diagnostic_id": "stage1_dev_diag_v1",
        "model": {
            "id": "fixture/model",
            "revision": "1" * 40,
            "max_length": 768,
            "choice_labels": ["A", "B"],
        },
        "inference_contract": {
            "max_new_tokens": 1,
            "generation_subset_size": len(subset),
            "token_length_audit": {
                "required_before_model_load": True,
                "truncation_allowed": False,
                "expected_prompt_count": len(cases),
                "expected_max_prompt_tokens": 745,
                "expected_all_prompt_token_counts_sha256": _json_sha256(pairs),
                "expected_generation_subset_token_counts_sha256": _json_sha256(
                    subset_pairs
                ),
                "expected_chat_template_sha256": hashlib.sha256(
                    _CharacterTokenizer.chat_template.encode("utf-8")
                ).hexdigest(),
                "expected_ordered_case_candidate_token_ids_sha256": _json_sha256(
                    candidate_rows
                ),
                "require_exact_max": True,
                "require_choice_boundary_single_token": True,
                "require_prompt_plus_generation_within_max_length": True,
            },
        },
    }


def test_current_did_contract_commits_exact_745_token_max_and_length_hashes():
    repository = Path(__file__).resolve().parents[1]
    spec = evaluation._core_load_spec(
        repository / "configs" / "stage1_dev_diag_v1.yaml"
    )
    contract = spec["inference_contract"]["token_length_audit"]
    assert spec["model"]["max_length"] == 768
    assert contract["expected_prompt_count"] == 19_200
    assert contract["expected_max_prompt_tokens"] == 745
    assert contract["expected_all_prompt_token_counts_sha256"] == (
        "19987209a6211f3152ae6f8f8a4fbfc56242792354989a11f7934c49d11383ff"
    )
    assert contract["expected_generation_subset_token_counts_sha256"] == (
        "6e161c3503e74e914e07b3aa179c6beffa71dc9814815fbea44e03abe6710d88"
    )
    assert contract["expected_chat_template_sha256"] == (
        "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"
    )
    assert contract["expected_ordered_case_candidate_token_ids_sha256"] == (
        "9272c58e8a64e25384a1af6b0ba91d3663638b65aa1258293c80152e330586f4"
    )


def test_production_generation_token_counts_are_reindexed_to_committed_order():
    repository = Path(__file__).resolve().parents[1]
    spec = evaluation._core_load_spec(
        repository / "configs" / "stage1_dev_diag_v1.yaml"
    )
    cases = dev_diag._generate_dev_diag_cases(spec)
    committed_ids = dev_diag.generation_subset_case_ids(cases, spec)
    committed_set = set(committed_ids)
    prediction_order_ids = [
        case["case_id"] for case in cases if case["case_id"] in committed_set
    ]

    assert len(committed_ids) == 256
    assert committed_ids[0] == (
        "DID-AUDIT-STATIC-0041--t0--CAN0--MAP_G--identity"
    )
    assert prediction_order_ids[0] == (
        "DID-AUDIT-STATIC-0000--t0--CAN0--EXPLICIT_P--swap"
    )
    assert committed_ids != prediction_order_ids
    assert set(committed_ids) == set(prediction_order_ids)

    counts_by_case = {
        case_id: 454 + index for index, case_id in enumerate(prediction_order_ids)
    }
    rows = evaluation._ordered_generation_token_count_rows(
        counts_by_case,
        committed_ids,
        expected_size=256,
    )
    assert [row[0] for row in rows] == committed_ids
    assert rows == [[case_id, counts_by_case[case_id]] for case_id in committed_ids]


def test_preinference_token_audit_accepts_exact_max_and_hashes_every_length():
    cases = [_length_case("short", 12), _length_case("exact-max", 745)]
    subset = ["exact-max"]
    tokenizer = _CharacterTokenizer()
    spec = _length_spec(cases, subset)
    audit = evaluation.audit_dev_diag_token_lengths(
        tokenizer, cases, subset, spec
    )
    assert audit.report["observed_max_prompt_tokens"] == 745
    assert audit.report["all_prompts"]["maximum_total_tokens_after_generation"] == 746
    assert audit.prompt_token_counts == {"short": 12, "exact-max": 745}
    contract = spec["inference_contract"]["token_length_audit"]
    assert audit.report["checks"]["chat_template_hash_matches"] is True
    assert audit.report["checks"]["ordered_candidate_token_ids_hash_matches"] is True
    assert audit.report["chat_template_sha256"] == contract[
        "expected_chat_template_sha256"
    ]
    assert audit.report["candidate_boundary"][
        "ordered_case_candidate_token_ids_sha256"
    ] == contract["expected_ordered_case_candidate_token_ids_sha256"]
    assert all(value is False for value in tokenizer.truncation_arguments)


def test_preinference_token_audit_rejects_same_length_template_or_candidate_drift():
    cases = [_length_case("short", 12), _length_case("exact-max", 745)]
    subset = ["exact-max"]
    spec = _length_spec(cases, subset)

    changed_template = _CharacterTokenizer()
    changed_template.chat_template = "fixture-template-with-same-rendering"
    with pytest.raises(ValueError, match="chat template differs"):
        evaluation.audit_dev_diag_token_lengths(
            changed_template, cases, subset, spec
        )

    with pytest.raises(ValueError, match="candidate token IDs differ"):
        evaluation.audit_dev_diag_token_lengths(
            _CandidateDriftTokenizer(), cases, subset, spec
        )


def test_scorer_and_generator_never_truncate_and_reject_overlength(
    monkeypatch: pytest.MonkeyPatch,
):
    tokenizer = _CharacterTokenizer()
    exact = [_length_case("exact", 745)]
    observed: dict[str, Any] = {}

    def fake_choice_scores(model, supplied_tokenizer, cases, *, max_length):
        observed["max_length"] = max_length
        return torch.tensor([[-0.5108256, -0.9162907]]), torch.tensor(
            [[-0.5108256, -0.9162907]]
        )

    monkeypatch.setattr(
        evaluation, "differentiable_choice_log_probs", fake_choice_scores
    )
    scores = evaluation._score_model(
        object(),
        tokenizer,
        exact,
        batch_size=1,
        max_length=768,
        expected_prompt_token_counts={"exact": 745},
    )
    assert set(scores) == {"exact"}
    assert observed["max_length"] == 768

    generated = evaluation._generate_model(
        _GeneratingModel(),
        tokenizer,
        exact,
        batch_size=1,
        max_length=768,
        max_new_tokens=1,
        expected_prompt_token_counts={"exact": 745},
    )
    assert generated["exact"] == {
        "generated_output": "A",
        "parsed_action": "A",
        "parse_status": "exact",
    }
    assert all(value is False for value in tokenizer.truncation_arguments)

    prompt_fits_but_generation_does_not = [_length_case("too-tight", 768)]
    with pytest.raises(ValueError, match="plus 1 generated token"):
        evaluation._score_model(
            object(),
            tokenizer,
            prompt_fits_but_generation_does_not,
            batch_size=1,
            max_length=768,
        )
    with pytest.raises(ValueError, match="plus 1 generated token"):
        evaluation._generate_model(
            _GeneratingModel(),
            tokenizer,
            prompt_fits_but_generation_does_not,
            batch_size=1,
            max_length=768,
            max_new_tokens=1,
        )
    overlength = [_length_case("overlength", 769)]
    with pytest.raises(ValueError, match="would require truncation"):
        evaluation._score_model(
            object(), tokenizer, overlength, batch_size=1, max_length=768
        )


def test_four_policy_run_is_inference_only_atomic_and_non_overwriting(
    synthetic_inputs, monkeypatch: pytest.MonkeyPatch
):
    bridge_config, kwargs, tmp_path = synthetic_inputs
    load_counts = {"fresh_base": 0, "adapter": 0, "tokenizer": 0}

    def load_base(config, *, training):
        assert training is False
        load_counts["fresh_base"] += 1
        return _FakeModel(frozen=False)

    def load_adapter(config, path):
        load_counts["fresh_base"] += 1
        load_counts["adapter"] += 1
        return _FakeModel(frozen=True)

    monkeypatch.setattr(evaluation, "load_tokenizer", lambda config: load_counts.__setitem__("tokenizer", load_counts["tokenizer"] + 1) or object())
    monkeypatch.setattr(evaluation, "load_base_model", load_base)
    monkeypatch.setattr(evaluation, "load_adapter_model", load_adapter)
    monkeypatch.setattr(
        evaluation,
        "validate_loaded_base_runtime",
        lambda *args: {"contract_sha256": "base"},
    )
    monkeypatch.setattr(
        evaluation,
        "validate_loaded_lora_runtime",
        lambda *args: {"contract_sha256": "adapter"},
    )
    monkeypatch.setattr(evaluation, "environment_snapshot", lambda: {"cuda_available": False})
    monkeypatch.setattr(evaluation, "project_hash", lambda root: "3" * 64)
    monkeypatch.setattr(
        evaluation,
        "audit_dev_diag_token_lengths",
        lambda tokenizer, cases, subset, spec: _fixture_token_audit(
            list(cases), list(subset)
        ),
    )

    def score(model, tokenizer, cases, **ignored):
        assert model.training is False
        assert all(not parameter.requires_grad for parameter in model.parameters())
        return {
            case["case_id"]: {
                "logp_A": -0.5108256238,
                "logp_B": -0.9162907319,
                "probability_A": 0.6,
                "probability_B": 0.4,
                "log_legal_choice_mass": 0.0,
                "legal_choice_mass": 1.0,
            }
            for case in cases
        }

    def generate(model, tokenizer, cases, **ignored):
        return {
            case["case_id"]: {
                "generated_output": "A",
                "parsed_action": "A",
                "parse_status": "exact",
            }
            for case in cases
        }

    monkeypatch.setattr(evaluation, "_score_model", score)
    monkeypatch.setattr(evaluation, "_generate_model", generate)

    destination = tmp_path / "did_run"
    completed = evaluation.evaluate_dev_diagnostic(
        bridge_config,
        **kwargs,
        destination_dir=destination,
    )
    assert completed == destination.resolve()
    manifest = json.loads((destination / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["state"] == "COMPLETE"
    assert manifest["stage1_evidence"] is False
    assert manifest["complete_four_policy_grid"] is True
    assert [row["policy_condition"] for row in manifest["outputs"]] == list(
        evaluation.POLICY_CONDITIONS
    )
    for policy, output in zip(
        evaluation.POLICY_CONDITIONS, manifest["outputs"], strict=True
    ):
        assert output["predictions_path"] == f"{policy}/predictions.jsonl"
        assert output["summary_path"] == f"{policy}/summary.json"
        assert not Path(output["predictions_path"]).is_absolute()
        summary = json.loads(
            (destination / policy / "summary.json").read_text(encoding="utf-8")
        )
        assert summary["predictions_path"] == f"{policy}/predictions.jsonl"
    assert load_counts == {"fresh_base": 4, "adapter": 3, "tokenizer": 5}
    for policy in evaluation.POLICY_CONDITIONS:
        rows = [
            json.loads(line)
            for line in (destination / policy / "predictions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert len(rows) == evaluation.FROZEN_EXPECTED_PROMPT_COUNT
        assert all(row["stage1_evidence"] is False for row in rows)
        assert sum(row["generation_subset_selected"] for row in rows) == 2
        assert all("correct" not in row for row in rows)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        evaluation.evaluate_dev_diagnostic(
            bridge_config,
            **kwargs,
            destination_dir=destination,
        )

    original_run = evaluation.verify_completed_dev_diagnostic_run(
        destination,
        **_completed_verifier_kwargs(kwargs),
    )
    original_paths = list(original_run.prediction_paths)
    assert original_paths == [
        (destination / policy / "predictions.jsonl").resolve()
        for policy in evaluation.POLICY_CONDITIONS
    ]
    committed_generation_ids = manifest["input_bindings"]["generation_subset"][
        "case_ids"
    ]
    prediction_order_generation_ids = [
        row["case_id"]
        for row in [
            json.loads(line)
            for line in original_paths[0].read_text(encoding="utf-8").splitlines()
        ]
        if row["generation_subset_selected"]
    ]
    assert committed_generation_ids != prediction_order_generation_ids
    assert set(committed_generation_ids) == set(prediction_order_generation_ids)
    key_verification = evaluation.verify_dev_diagnostic_analysis_inputs(
        original_run,
        answer_key_path=tmp_path / "answer_key.jsonl",
    )
    assert key_verification["checks"]["answer_key_sha256_matches_commitment"] is True
    wrong_key = tmp_path / "wrong_answer_key.jsonl"
    _write_jsonl(wrong_key, [{"case_id": "wrong", "correct_action": "B"}])
    with pytest.raises(ValueError, match="does not match the committed hash"):
        evaluation.verify_dev_diagnostic_analysis_inputs(
            original_run,
            answer_key_path=wrong_key,
        )

    direct_report = {
        "schema_version": "DID-v1",
        "kind": "did_v1_posthoc_dev_diagnostic_analysis",
        "diagnostic_id": "stage1_dev_diag_v1",
        "all_gates_pass": True,
        "gates": {"fixture": True},
        "localization_outcome": "LICENSES_NEW_PREREGISTERED_E1B_DEV2_ONLY",
        "decision": (
            "UNVERIFIED_DIRECT_API_"
            "LICENSES_NEW_PREREGISTERED_E1B_DEV2_ONLY"
        ),
        "verification_status": "unverified_direct_api",
        "interpretation_contract": {
            "can_license_e1b": False,
            "verified_inference_run": False,
            "conditional_outcome_after_verified_finalize": (
                "LICENSES_NEW_PREREGISTERED_E1B_DEV2_ONLY"
            ),
        },
    }
    monkeypatch.setattr(
        dev_diag,
        "analyze_dev_diag_predictions",
        lambda *args, **analysis_kwargs: json.loads(canonical_json(direct_report)),
    )
    formal_destination = tmp_path / "formal_analysis.json"
    assert evaluation.finalize_verified_dev_diagnostic_analysis(
        run_dir=destination,
        deployment_root=kwargs["deployment_root"],
        bootstrap_attestation_path=kwargs["bootstrap_attestation_path"],
        spec_path=kwargs["spec_path"],
        case_manifest_path=kwargs["case_manifest_path"],
        cases_path=kwargs["cases_path"],
        answer_key_commitment_path=kwargs["answer_key_commitment_path"],
        answer_key_path=tmp_path / "answer_key.jsonl",
        destination=formal_destination,
    ) == formal_destination.resolve()
    formal = json.loads(formal_destination.read_text(encoding="utf-8"))
    assert formal["decision"] == "LICENSES_NEW_PREREGISTERED_E1B_DEV2_ONLY"
    assert formal["verification_status"] == "verified_completed_run"
    assert formal["interpretation_contract"]["can_license_e1b"] is True
    assert formal["interpretation_contract"]["verified_inference_run"] is True
    assert formal["inference_run"][
        "independent_completed_run_verification_passes"
    ] == 2

    changed_prediction = original_paths[0]
    unchanged_prediction_bytes = changed_prediction.read_bytes()

    def tampering_direct_analysis(*args, **analysis_kwargs):
        changed_prediction.write_bytes(unchanged_prediction_bytes + b"\n")
        return json.loads(canonical_json(direct_report))

    monkeypatch.setattr(
        dev_diag, "analyze_dev_diag_predictions", tampering_direct_analysis
    )
    rejected_destination = tmp_path / "must_not_exist.json"
    with pytest.raises(ValueError, match="Prediction hash mismatch|changed"):
        evaluation.finalize_verified_dev_diagnostic_analysis(
            run_dir=destination,
            deployment_root=kwargs["deployment_root"],
            bootstrap_attestation_path=kwargs["bootstrap_attestation_path"],
            spec_path=kwargs["spec_path"],
            case_manifest_path=kwargs["case_manifest_path"],
            cases_path=kwargs["cases_path"],
            answer_key_commitment_path=kwargs["answer_key_commitment_path"],
            answer_key_path=tmp_path / "answer_key.jsonl",
            destination=rejected_destination,
        )
    assert not rejected_destination.exists()
    changed_prediction.write_bytes(unchanged_prediction_bytes)

    alternate_bundle = tmp_path / "alternate_bundle"
    alternate_bundle.mkdir()
    alternate_cases = alternate_bundle / "cases.jsonl"
    alternate_rows = [
        json.loads(line)
        for line in Path(kwargs["cases_path"]).read_text(encoding="utf-8").splitlines()
    ]
    alternate_rows[0]["causal_state"]["cross_run_marker"] = True
    _write_jsonl(alternate_cases, alternate_rows)
    alternate_commitment = alternate_bundle / "ANSWER_KEY_COMMITMENT.json"
    commitment = json.loads(
        Path(kwargs["answer_key_commitment_path"]).read_text(encoding="utf-8")
    )
    commitment["case_set_sha256"] = _sha256(alternate_cases)
    _write_json(alternate_commitment, commitment)
    alternate_manifest = alternate_bundle / "MANIFEST.json"
    manifest = json.loads(
        Path(kwargs["case_manifest_path"]).read_text(encoding="utf-8")
    )
    manifest["files"]["cases"].update(
        {
            "path": "cases.jsonl",
            "sha256": _sha256(alternate_cases),
            "bytes": alternate_cases.stat().st_size,
        }
    )
    manifest["files"]["answer_key_commitment"].update(
        {
            "path": "ANSWER_KEY_COMMITMENT.json",
            "sha256": _sha256(alternate_commitment),
            "bytes": alternate_commitment.stat().st_size,
        }
    )
    _write_json(alternate_manifest, manifest)
    with pytest.raises(
        ValueError, match="not the attested deployment payload|differs from run binding"
    ):
        evaluation.verify_completed_dev_diagnostic_run(
            destination,
            spec_path=kwargs["spec_path"],
            case_manifest_path=alternate_manifest,
            cases_path=alternate_cases,
            answer_key_commitment_path=alternate_commitment,
            deployment_root=kwargs["deployment_root"],
            bootstrap_attestation_path=kwargs["bootstrap_attestation_path"],
        )

    relocated_parent = tmp_path / "relocated" / "nested"
    relocated_parent.mkdir(parents=True)
    relocated = relocated_parent / "portable_did_run"
    destination.rename(relocated)
    relocated_run = evaluation.verify_completed_dev_diagnostic_run(
        relocated,
        **_completed_verifier_kwargs(kwargs),
    )
    relocated_paths = list(relocated_run.prediction_paths)
    assert relocated_paths == [
        (relocated / policy / "predictions.jsonl").resolve()
        for policy in evaluation.POLICY_CONDITIONS
    ]
    assert all(path.is_file() for path in relocated_paths)

    tampered = relocated_paths[-1]
    tampered.write_text(
        tampered.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Prediction hash mismatch"):
        evaluation.verify_completed_dev_diagnostic_run(
            relocated,
            **_completed_verifier_kwargs(kwargs),
        )


def test_failed_later_policy_preserves_completed_atomic_output(
    synthetic_inputs, monkeypatch: pytest.MonkeyPatch
):
    bridge_config, kwargs, tmp_path = synthetic_inputs
    monkeypatch.setattr(evaluation, "environment_snapshot", lambda: {})
    monkeypatch.setattr(evaluation, "project_hash", lambda root: "2" * 64)
    monkeypatch.setattr(evaluation, "load_tokenizer", lambda config: object())
    monkeypatch.setattr(
        evaluation,
        "audit_dev_diag_token_lengths",
        lambda tokenizer, cases, subset, spec: _fixture_token_audit(
            list(cases), list(subset)
        ),
    )
    calls = 0

    def policy(policy_condition, verified):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("fixture second-policy failure")
        rows = [{"case_id": "only", "policy_condition": policy_condition}]
        summary = {"policy_condition": policy_condition, "stage1_evidence": False}
        return rows, summary

    monkeypatch.setattr(evaluation, "_evaluate_policy", policy)
    destination = tmp_path / "partial_run"
    with pytest.raises(RuntimeError, match="second-policy failure"):
        evaluation.evaluate_dev_diagnostic(
            bridge_config,
            **kwargs,
            destination_dir=destination,
        )
    manifest = json.loads((destination / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["state"] == "FAILED"
    assert manifest["error"]["type"] == "RuntimeError"
    first = destination / evaluation.POLICY_CONDITIONS[0]
    assert (first / "predictions.jsonl").is_file()
    assert (first / "summary.json").is_file()
    assert not (destination / evaluation.POLICY_CONDITIONS[1]).exists()
