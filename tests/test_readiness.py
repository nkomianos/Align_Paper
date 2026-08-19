from __future__ import annotations

import json
import os
import subprocess
import tarfile
from copy import deepcopy
from pathlib import Path

from under_extinction.deployment import create_bundle
from under_extinction.artifacts import _artifact_files
from under_extinction.bridge_env import build_bridge_data
import under_extinction.bridge_preflight as bridge_preflight
from under_extinction.bridge_preflight import (
    verify_bridge_preflight_attestation,
    write_bridge_preflight_attestation,
)
from under_extinction.config import config_hash, load_config
from under_extinction.cli import _bridge_dry_run_summary
from under_extinction.generator import generate_datasets
from under_extinction.manifest import create_manifest
from under_extinction.preflight import assert_immutable_revision
from under_extinction.readiness import dry_run_summary


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dry_run_is_ready_after_matching_build(smoke_config):
    generate_datasets(smoke_config)
    summary = dry_run_summary(smoke_config)
    assert summary["immutable_model_revision"]
    assert summary["data_ready_and_hash_matched"]
    assert summary["ready_for_paid_preflight"]
    assert summary["adapter_run_count"] == 6


def test_manifest_environment_is_allowlisted(smoke_config, tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    generate_datasets(smoke_config, data_dir)
    monkeypatch.setenv("HF_TOKEN", "must-not-appear")
    monkeypatch.setenv("UE_INSTANCE_TYPE", "test-instance")
    monkeypatch.setenv("UE_INSTANCE_ID", "test-instance-id")
    monkeypatch.setenv("UE_INSTANCE_LAUNCHED_AT", "2026-08-17T00:00:00Z")
    monkeypatch.setenv("UE_INSTANCE_START_EPOCH", "1786924800")
    monkeypatch.setenv("UE_HOURLY_USD", "2.29")
    manifest = create_manifest(
        smoke_config,
        run_dir=tmp_path / "run",
        controller="intended",
        seed=11,
        command_line=["under-extinction", "train"],
        data_files=[data_dir / "train.jsonl"],
    )
    serialized = json.dumps(manifest)
    assert "must-not-appear" not in serialized
    assert manifest["environment"]["safe_environment"]["UE_INSTANCE_TYPE"] == "test-instance"
    assert manifest["environment"]["safe_environment"]["UE_INSTANCE_ID"] == "test-instance-id"
    assert manifest["environment"]["safe_environment"]["UE_INSTANCE_START_EPOCH"] == "1786924800"
    assert manifest["environment"]["safe_environment"]["UE_HOURLY_USD"] == "2.29"
    assert {"huggingface-hub", "tokenizers", "causal-conv1d", "fla-core", "kernels"} <= set(
        manifest["environment"]["packages"]
    )


def test_manifest_discovers_standalone_repository_root(smoke_config, tmp_path):
    project = tmp_path / "Align_Paper"
    configs = project / "configs"
    configs.mkdir(parents=True)
    readme = project / "README.md"
    readme.write_text("standalone\n", encoding="utf-8")
    subprocess.run(["git", "init", "--quiet"], cwd=project, check=True)
    subprocess.run(["git", "config", "user.email", "tests@example.invalid"], cwd=project, check=True)
    subprocess.run(["git", "config", "user.name", "Test Runner"], cwd=project, check=True)
    subprocess.run(["git", "add", "README.md"], cwd=project, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "initial"], cwd=project, check=True)
    expected_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    config = deepcopy(smoke_config)
    config["_config_path"] = str(configs / "bridge_smoke.yaml")
    data = tmp_path / "data.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    manifest = create_manifest(
        config,
        run_dir=tmp_path / "clean-run",
        controller="intended",
        seed=11,
        command_line=["under-extinction", "train"],
        data_files=[data],
    )
    assert manifest["source"]["git_commit"] == expected_commit
    assert manifest["source"]["git_dirty"] is False

    readme.write_text("dirty standalone\n", encoding="utf-8")
    dirty_manifest = create_manifest(
        config,
        run_dir=tmp_path / "dirty-run",
        controller="intended",
        seed=11,
        command_line=["under-extinction", "train"],
        data_files=[data],
    )
    assert dirty_manifest["source"]["git_commit"] == expected_commit
    assert dirty_manifest["source"]["git_dirty"] is True


def test_bundle_excludes_weights_caches_and_legacy_tree(smoke_config, tmp_path):
    generate_datasets(smoke_config)
    bundle = create_bundle(smoke_config, tmp_path / "bundle.tar.gz")
    with tarfile.open(bundle, "r:gz") as archive:
        names = [member.name for member in archive.getmembers()]
    assert any(name.endswith("frozen_data/MANIFEST.json") for name in names)
    assert not any(".claude" in name for name in names)
    assert not any(name.endswith((".safetensors", ".bin", ".pt", ".pth")) for name in names)


def test_model_revision_is_immutable(smoke_config):
    assert_immutable_revision(smoke_config)
    smoke_config["model"]["revision"] = "main"
    try:
        assert_immutable_revision(smoke_config)
    except ValueError:
        pass
    else:
        raise AssertionError("Mutable model revision was accepted")


def test_paid_runtime_lock_uses_released_qwen35_compatible_pins() -> None:
    lock = (PROJECT_ROOT / "requirements" / "h100-cu12x.lock").read_text(
        encoding="utf-8"
    )
    project = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for requirement in (
        "accelerate==1.14.0",
        "datasets==5.0.1",
        "huggingface-hub==1.27.0",
        "pandas==2.3.3",
        "peft==0.20.0",
        "safetensors==0.8.0",
        "tokenizers==0.22.2",
        "transformers==5.15.0",
    ):
        assert requirement in lock.splitlines()
        assert f'"{requirement}"' in project
    assert "git+" not in lock
    assert "@main" not in lock


def test_bridge_configs_freeze_exact_gh200_contract_and_rate() -> None:
    expected_hardware = {
        "provider": "lambda",
        "instance_type": "gpu_1x_gh200",
        "architecture": "aarch64",
        "accelerator_count": 1,
        "accelerator_name": "NVIDIA GH200 480GB",
        "accelerator_memory_gib": 96,
        "minimum_accelerator_memory_gib": 90,
        "compute_capability_major": 9,
    }
    for name in ("bridge_smoke.yaml", "bridge_pilot.yaml"):
        config = load_config(PROJECT_ROOT / "configs" / name)
        assert config["hardware"] == expected_hardware
        assert config["budget"]["hourly_usd"] == 2.29


def test_bootstrap_fails_closed_on_exact_gh200_hardware_contract() -> None:
    script = (PROJECT_ROOT / "scripts" / "bootstrap_lambda.sh").read_text(
        encoding="utf-8"
    )
    assert '"$(uname -m)" != "aarch64"' in script
    assert "torch.cuda.device_count() != 1" in script
    assert 'name != "NVIDIA GH200 480GB"' in script
    assert "capability[0] != 9 or memory_gib < 90" in script


def test_bridge_readiness_exposes_exact_model_and_workload_contract(tmp_path) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "bridge_pilot.yaml")
    config["_config_path"] = str(tmp_path / "configs" / "bridge_pilot.yaml")
    config["_config_sha256"] = config_hash(config)
    summary = _bridge_dry_run_summary(config)
    assert summary["model"] == config["model"]
    assert summary["microbatch_size"] == 4
    assert summary["checkpoint_updates"] == [0, 30, 75, 150, 225, 300]
    assert summary["checkpoint_count_per_arm"] == 6
    assert summary["expected_lora_module_count"] == 248
    assert summary["expected_lora_trainable_parameter_count"] == 43_278_336


def test_paid_bridge_scripts_require_billing_start_and_frozen_hourly_rate() -> None:
    for name in (
        "preflight_bridge.sh",
        "run_bridge_stage1.sh",
        "run_bridge_replication.sh",
    ):
        script = (PROJECT_ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "UE_INSTANCE_LAUNCHED_AT" in script
        assert "UE_INSTANCE_ID" in script
        assert "UE_INSTANCE_START_EPOCH" in script
        assert "UE_HOURLY_USD" in script
        assert 'export UE_COMPUTE_DEADLINE_EPOCH="$COMPUTE_DEADLINE_EPOCH"' in script
        assert "config_scalar budget.hourly_usd" in script


def test_bootstrap_fails_closed_on_optional_delta_net_backends() -> None:
    script = (PROJECT_ROOT / "scripts" / "bootstrap_lambda.sh").read_text(
        encoding="utf-8"
    )
    assert '"causal-conv1d": "causal_conv1d"' in script
    assert '"fla-core": "fla"' in script
    assert '"kernels": "kernels"' in script
    assert "if present_optional_backends:" in script
    assert "torch_fallback_required contract forbids optional DeltaNet" in script


def test_bootstrap_scopes_dependency_validation_to_experiment_closure() -> None:
    script = (PROJECT_ROOT / "scripts" / "bootstrap_lambda.sh").read_text(
        encoding="utf-8"
    )
    assert "if ! python -m pip check; then" in script
    assert "Experiment dependency closure is inconsistent" in script
    assert 'Requirement("torch>=2.5,<3")' in script
    assert 'Requirement("under-extinction==0.1.0")' in script
    assert "root requires {requirement}" in script


def test_result_collection_keeps_all_bridge_science_checkpoints(tmp_path):
    artifacts = tmp_path / "artifacts"
    run = artifacts / "bridge" / "runs" / "genuine_seed11" / "checkpoints"
    old = run / "checkpoint-000010"
    latest = run / "checkpoint-000020"
    for checkpoint in (old, latest):
        checkpoint.mkdir(parents=True)
        (checkpoint / "checkpoint_manifest.json").write_text("{}", encoding="utf-8")
        (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
        (checkpoint / "bridge_state.pt").write_bytes(b"resume")
    selected = {path.resolve() for path in _artifact_files(artifacts)}
    assert (old / "adapter_model.safetensors").resolve() in selected
    assert (old / "bridge_state.pt").resolve() in selected
    assert (latest / "bridge_state.pt").resolve() in selected


def test_result_collection_preserves_resumable_final_adapter_state(tmp_path):
    artifacts = tmp_path / "artifacts"
    final_adapter = artifacts / "bridge" / "runs" / "genuine_seed11" / "final_adapter"
    final_adapter.mkdir(parents=True)
    (final_adapter / "checkpoint_manifest.json").write_text("{}", encoding="utf-8")
    (final_adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    state = final_adapter / "bridge_state.pt"
    state.write_bytes(b"resume")
    selected = {path.resolve() for path in _artifact_files(artifacts)}
    assert state.resolve() in selected


def test_bridge_preflight_pass_is_bound_to_project_configs_model_runtime_and_data(
    tmp_path, monkeypatch
):
    project = tmp_path / "project"
    configs = project / "configs"
    configs.mkdir(parents=True)
    smoke = load_config(PROJECT_ROOT / "configs" / "bridge_smoke.yaml")
    stage1 = deepcopy(smoke)
    smoke["_config_path"] = str(configs / "bridge_smoke.yaml")
    smoke["output_root"] = "artifacts/smoke"
    smoke["_config_sha256"] = config_hash(smoke)
    stage1["_config_path"] = str(configs / "bridge_pilot.yaml")
    stage1["experiment_name"] = "test_stage1"
    stage1["output_root"] = "artifacts/stage1"
    stage1["_config_sha256"] = config_hash(stage1)
    build_bridge_data(stage1)
    report = project / "artifacts" / "smoke" / "smoke_report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    fake_model_runtime = {"attestation_sha256": "a" * 64, "kind": "test_runtime"}
    report.write_text(
        json.dumps(
            {"kind": "test", "model_runtime_attestation": fake_model_runtime}
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        bridge_preflight,
        "verify_bridge_gate_report",
        lambda *_args, **_kwargs: {"pass": True, "failures": []},
    )
    monkeypatch.setattr(
        bridge_preflight,
        "verify_model_runtime_attestation",
        lambda _config, value: dict(value),
    )
    monkeypatch.setattr(
        bridge_preflight,
        "compact_model_runtime_contract",
        lambda _config, _value: {"kind": "test_compact_runtime"},
    )
    fake_projection = {
        "pass": True,
        "checks": {"within_budget": True},
        "projection": {
            "guarded_total_seconds": 120.0,
            "guarded_cost_usd": 0.11,
        },
    }
    monkeypatch.setattr(
        bridge_preflight,
        "build_stage1_cost_projection",
        lambda *_args, **_kwargs: deepcopy(fake_projection),
    )
    monkeypatch.setattr(
        bridge_preflight,
        "verify_stage1_cost_projection",
        lambda value, *_args, **_kwargs: dict(value),
    )

    attestation = write_bridge_preflight_attestation(
        smoke, stage1, smoke_report_path=report
    )
    verified = verify_bridge_preflight_attestation(
        smoke, stage1, attestation_path=attestation
    )
    assert verified["pass"] is True

    payload = json.loads(attestation.read_text(encoding="utf-8"))
    payload["stage1_config_sha256"] = "0" * 64
    attestation.write_text(json.dumps(payload), encoding="utf-8")
    try:
        verify_bridge_preflight_attestation(smoke, stage1, attestation_path=attestation)
    except ValueError as exc:
        assert "self-hash mismatch" in str(exc)
        assert "Stage 1 config hash mismatch" in str(exc)
    else:
        raise AssertionError("Tampered paid-preflight handoff was accepted")
