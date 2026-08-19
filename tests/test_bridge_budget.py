from __future__ import annotations

import json
from pathlib import Path

import pytest

import under_extinction.bridge_budget as bridge_budget
from under_extinction.config import load_config
from under_extinction.io import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _timing_components() -> dict[str, float]:
    return {
        "model_load": 10.0,
        "adapter_setup": 2.0,
        "update": 1.0,
        "checkpoint": 0.5,
        "reload_probe": 0.2,
        "finalize": 1.0,
        "unattributed_residual": 3.0,
    }


def _evaluation_components() -> dict[str, float]:
    return {
        "model_load": 5.0,
        "adapter_setup": 1.0,
        "reload_probe": 0.2,
        "forced_rate": 0.01,
        "generation_rate": 0.02,
        "write_rate": 0.001,
        "unattributed_residual": 2.0,
        "total": 9.0,
    }


def _projection_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    smoke = load_config(PROJECT_ROOT / "configs" / "bridge_smoke.yaml")
    stage1 = load_config(PROJECT_ROOT / "configs" / "bridge_pilot.yaml")
    seed = int(smoke["bridge"]["seeds"][0])
    split = str(smoke["bridge"]["splits"]["development"])
    run_root = tmp_path / "preflight"
    inputs: dict[str, str] = {}
    for arm in ("genuine", "proxy"):
        manifest = run_root / "runs" / f"{arm}_seed{seed}" / "bridge_manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("{}\n", encoding="utf-8")
        prediction = run_root / "predictions" / f"{arm}_seed{seed}_{split}.jsonl"
        prediction.parent.mkdir(parents=True, exist_ok=True)
        prediction.write_text("{}\n", encoding="utf-8")
        summary = prediction.with_suffix(".summary.json")
        summary.write_text("{}\n", encoding="utf-8")
        inputs[str(prediction.resolve())] = sha256_file(prediction)
        inputs[str(summary.resolve())] = sha256_file(summary)
    base = run_root / "predictions" / f"unchanged_base_seed{seed}_{split}.jsonl"
    base.write_text("{}\n", encoding="utf-8")
    base_summary = base.with_suffix(".summary.json")
    base_summary.write_text("{}\n", encoding="utf-8")
    inputs[str(base.resolve())] = sha256_file(base)
    inputs[str(base_summary.resolve())] = sha256_file(base_summary)
    report = run_root / "smoke_report.json"
    report.write_text(
        json.dumps(
            {
                "kind": "same_environment_bridge_report",
                "config_sha256": smoke["_config_sha256"],
                "split": split,
                "gates": {"smoke": {"pass": True}},
                "input_sha256": inputs,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    telemetry_path = run_root / "gpu_telemetry.csv"
    telemetry_path.write_text(
        "2026/08/17 12:00:00.000, 0, NVIDIA GH200 480GB, 50, 30000, 97871, 50, 300\n",
        encoding="utf-8",
    )
    profile_path = tmp_path / "profile.json"
    smoke_manifest = tmp_path / "smoke_manifest.json"
    stage1_manifest = tmp_path / "stage1_manifest.json"
    for path in (profile_path, smoke_manifest, stage1_manifest):
        path.write_text("{}\n", encoding="utf-8")
    runtime = {"attestation_sha256": "a" * 64}
    contract = {
        "deltanet_backend": "torch_fallback",
        "chat_template_sha256": "c" * 64,
    }
    monkeypatch.setattr(
        bridge_budget,
        "_expected_runtime",
        lambda *_args, **_kwargs: (runtime, contract),
    )
    monkeypatch.setattr(
        bridge_budget,
        "_training_timing",
        lambda *_args, **_kwargs: (_timing_components(), 30 * 1024**3),
    )

    def fake_evaluation(*_args, base_policy: bool, **_kwargs):
        values = [_evaluation_components()]
        if not base_policy:
            values *= 2
        return values, (0.0005 if not base_policy else 0.0), [32 * 1024**3] * len(values)

    monkeypatch.setattr(
        bridge_budget, "_validate_evaluation_summary", fake_evaluation
    )
    monkeypatch.setattr(
        bridge_budget,
        "_load_workload_profile",
        lambda *_args, **_kwargs: (
            {
                "training_update_scale": 1.1,
                "evaluation_per_record_scale": 1.05,
                "chat_template_sha256": "c" * 64,
                "stage1_development": {"case_count": 1792},
            },
            profile_path,
            smoke_manifest,
            stage1_manifest,
        ),
    )
    monkeypatch.setattr(
        bridge_budget,
        "_telemetry",
        lambda _path, *, hardware: {
            "schema_version": "1.0",
            "sample_count": 1,
            "gpu_index": "0",
            "gpu_name": hardware["accelerator_name"],
            "expected_gpu_name": hardware["accelerator_name"],
            "minimum_accelerator_memory_gib": hardware[
                "minimum_accelerator_memory_gib"
            ],
            "maximum_sampled_memory_used_mib": 30000.0,
            "device_memory_total_mib": 97871.0,
            "device_memory_total_bytes": int(97871 * 1024**2),
        },
    )
    return smoke, stage1, report, telemetry_path


def test_projection_scales_exact_workload_and_enforces_live_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smoke, stage1, report, _ = _projection_fixture(tmp_path, monkeypatch)
    environment = {
        "UE_INSTANCE_ID": "instance-test-001",
        "UE_INSTANCE_START_EPOCH": "1000",
        "UE_HARD_DEADLINE_EPOCH": "100000",
        "UE_HOURLY_USD": "2.29",
    }
    projection = bridge_budget.build_stage1_cost_projection(
        smoke,
        stage1,
        smoke_report_path=report,
        now_epoch=2000,
        environment=environment,
    )
    assert projection["pass"] is True
    assert projection["formal_workload"]["checkpoint_updates"] == [0, 30, 75, 150, 225, 300]
    assert projection["formal_workload"]["adapter_evaluation_count"] == 12
    assert projection["projection"]["training_seconds_per_arm"] == pytest.approx(
        10 + 2 + 300 * 1.0 * 1.1 + 6 * 0.5 + 5 * 0.2 + 1 + 3
    )
    assert projection["projection"]["guarded_total_seconds"] == pytest.approx(
        projection["projection"]["raw_total_seconds"] * 1.30
    )
    assert projection["checks"]["peak_vram_within_frozen_headroom"] is True
    assert projection["runtime"]["projected_training_peak_vram_bytes"] == pytest.approx(
        30 * 1024**3 * 1.1, abs=1
    )
    assert projection["runtime"]["projected_evaluation_peak_vram_bytes"] == pytest.approx(
        32 * 1024**3 * 1.05, abs=1
    )
    assert projection["runtime"]["maximum_peak_vram_bytes"] == projection[
        "runtime"
    ]["projected_evaluation_peak_vram_bytes"]
    assert projection["runtime"]["gpu_telemetry"]["gpu_name"] == (
        stage1["hardware"]["accelerator_name"]
    )

    too_short = dict(environment)
    too_short["UE_HARD_DEADLINE_EPOCH"] = "4000"
    rejected = bridge_budget.build_stage1_cost_projection(
        smoke,
        stage1,
        smoke_report_path=report,
        now_epoch=2000,
        environment=too_short,
    )
    assert rejected["pass"] is False
    assert rejected["checks"]["within_remaining_compute_window"] is False


def test_projection_verifier_detects_source_mutation_and_shrinking_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smoke, stage1, report, telemetry = _projection_fixture(tmp_path, monkeypatch)
    environment = {
        "UE_INSTANCE_ID": "instance-test-001",
        "UE_INSTANCE_START_EPOCH": "1000",
        "UE_HARD_DEADLINE_EPOCH": "100000",
        "UE_HOURLY_USD": "2.29",
    }
    stored = bridge_budget.build_stage1_cost_projection(
        smoke, stage1, smoke_report_path=report, now_epoch=2000, environment=environment
    )
    bridge_budget.verify_stage1_cost_projection(
        stored, smoke, stage1, smoke_report_path=report, now_epoch=2100, environment=environment
    )
    reset_start = dict(environment)
    reset_start["UE_INSTANCE_START_EPOCH"] = "1500"
    with pytest.raises(ValueError, match="instance_start_epoch"):
        bridge_budget.verify_stage1_cost_projection(
            stored,
            smoke,
            stage1,
            smoke_report_path=report,
            now_epoch=2100,
            environment=reset_start,
        )
    expanded_deadline = dict(environment)
    expanded_deadline["UE_HARD_DEADLINE_EPOCH"] = "110000"
    with pytest.raises(ValueError, match="deadline was expanded"):
        bridge_budget.verify_stage1_cost_projection(
            stored,
            smoke,
            stage1,
            smoke_report_path=report,
            now_epoch=2100,
            environment=expanded_deadline,
        )
    different_instance = dict(environment)
    different_instance["UE_INSTANCE_ID"] = "instance-test-002"
    with pytest.raises(ValueError, match="instance_id"):
        bridge_budget.verify_stage1_cost_projection(
            stored,
            smoke,
            stage1,
            smoke_report_path=report,
            now_epoch=2100,
            environment=different_instance,
        )
    telemetry.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source_file_sha256"):
        bridge_budget.verify_stage1_cost_projection(
            stored,
            smoke,
            stage1,
            smoke_report_path=report,
            now_epoch=2100,
            environment=environment,
        )


def test_projection_rejects_gh200_vram_headroom_overrun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smoke, stage1, report, _ = _projection_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        bridge_budget,
        "_training_timing",
        lambda *_args, **_kwargs: (_timing_components(), 82 * 1024**3),
    )
    projection = bridge_budget.build_stage1_cost_projection(
        smoke,
        stage1,
        smoke_report_path=report,
        now_epoch=2000,
        environment={
            "UE_INSTANCE_ID": "instance-test-001",
            "UE_INSTANCE_START_EPOCH": "1000",
            "UE_HARD_DEADLINE_EPOCH": "100000",
            "UE_HOURLY_USD": "2.29",
        },
    )
    assert projection["runtime"]["peak_vram_fraction_of_device"] > 0.90
    assert projection["checks"]["peak_vram_within_frozen_headroom"] is False
    assert projection["pass"] is False


def test_evaluation_timing_rejects_missing_or_nonfinite_measurements() -> None:
    timing = {
        "schema_version": "1.0",
        "model_and_tokenizer_load_wall_seconds": 1.0,
        "adapter_load_and_attestation_wall_seconds": 0.1,
        "adapter_reload_probe_wall_seconds": 0.1,
        "forced_scoring_wall_seconds": 1.0,
        "forced_record_count": 44,
        "generation_wall_seconds": 1.0,
        "generated_record_count": 44,
        "write_finalize_wall_seconds": 0.1,
        "total_wall_seconds": 3.5,
        "peak_vram_bytes": 1024,
    }
    values, peak = bridge_budget._evaluation_part_timing(
        {"timing": timing}, label="test", expected_records=44, expected_generated=44
    )
    assert values["forced_rate"] == pytest.approx(1 / 44)
    assert peak == 1024

    broken = {"timing": dict(timing)}
    broken["timing"]["generation_wall_seconds"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        bridge_budget._evaluation_part_timing(
            broken, label="test", expected_records=44, expected_generated=44
        )


def test_projection_rejects_throughput_contract_drift() -> None:
    smoke = load_config(PROJECT_ROOT / "configs" / "bridge_smoke.yaml")
    stage1 = load_config(PROJECT_ROOT / "configs" / "bridge_pilot.yaml")
    changed = json.loads(json.dumps(stage1))
    changed["evaluation"]["batch_size"] = 16
    with pytest.raises(ValueError, match="evaluation.batch_size"):
        bridge_budget._assert_throughput_contract(smoke, changed)

    changed = json.loads(json.dumps(stage1))
    changed["hardware"]["accelerator_name"] = "NVIDIA H100 PCIe"
    with pytest.raises(ValueError, match="same hardware contract"):
        bridge_budget._assert_throughput_contract(smoke, changed)


def test_telemetry_is_bound_to_exact_gpu_name_and_memory_floor(tmp_path: Path) -> None:
    hardware = {
        "provider": "lambda",
        "instance_type": "gpu_1x_gh200",
        "architecture": "aarch64",
        "accelerator_count": 1,
        "accelerator_name": "NVIDIA GH200 480GB",
        "accelerator_memory_gib": 96,
        "minimum_accelerator_memory_gib": 90,
        "compute_capability_major": 9,
    }
    path = tmp_path / "gpu_telemetry.csv"
    path.write_text(
        "2026/08/17 12:00:00.000, 0, NVIDIA GH200 480GB, 50, 30000, 97871, 50, 300\n",
        encoding="utf-8",
    )
    observed = bridge_budget._telemetry(path, hardware=hardware)
    assert observed["gpu_name"] == hardware["accelerator_name"]
    assert observed["device_memory_total_mib"] == 97871
    assert observed["minimum_accelerator_memory_gib"] == 90

    path.write_text(
        "2026/08/17 12:00:00.000, 0, NVIDIA H100 PCIe, 50, 30000, 97871, 50, 300\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match frozen hardware.accelerator_name"):
        bridge_budget._telemetry(path, hardware=hardware)

    path.write_text(
        "2026/08/17 12:00:00.000, 0, NVIDIA GH200 480GB, 50, 30000, 90000, 50, 300\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="below the frozen hardware memory floor"):
        bridge_budget._telemetry(path, hardware=hardware)


def test_telemetry_rejects_multiple_devices_and_impossible_usage(tmp_path: Path) -> None:
    hardware = {
        "accelerator_name": "NVIDIA GH200 480GB",
        "minimum_accelerator_memory_gib": 90,
    }
    path = tmp_path / "gpu_telemetry.csv"
    path.write_text(
        "2026/08/17 12:00:00.000, 0, NVIDIA GH200 480GB, 50, 30000, 97871, 50, 300\n"
        "2026/08/17 12:00:01.000, 1, NVIDIA GH200 480GB, 50, 30000, 97871, 50, 300\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly one stable device"):
        bridge_budget._telemetry(path, hardware=hardware)

    path.write_text(
        "2026/08/17 12:00:00.000, 0, NVIDIA GH200 480GB, 50, 98000, 97871, 50, 300\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="memory.used exceeds memory.total"):
        bridge_budget._telemetry(path, hardware=hardware)


def test_projection_rejects_workload_profile_chat_template_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smoke, stage1, report, _ = _projection_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        bridge_budget,
        "_expected_runtime",
        lambda *_args, **_kwargs: (
            {"attestation_sha256": "a" * 64},
            {
                "deltanet_backend": "torch_fallback",
                "chat_template_sha256": "d" * 64,
            },
        ),
    )
    environment = {
        "UE_INSTANCE_ID": "instance-test-001",
        "UE_INSTANCE_START_EPOCH": "1000",
        "UE_HARD_DEADLINE_EPOCH": "100000",
        "UE_HOURLY_USD": "2.29",
    }
    with pytest.raises(ValueError, match="different chat template"):
        bridge_budget.build_stage1_cost_projection(
            smoke,
            stage1,
            smoke_report_path=report,
            now_epoch=2000,
            environment=environment,
        )
