from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from under_extinction import dev_diag_deployment as deployment
from under_extinction.dev_diag_deployment import (
    BUNDLE_MANIFEST,
    BUNDLE_ROOT,
    DevDiagnosticBundleInputs,
    RESULTS_ROOT,
    collect_dev_diag_results,
    create_dev_diag_bundle,
    verify_dev_diag_bootstrap_attestation,
    verify_dev_diag_bundle,
    verify_dev_diag_results,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def _git(project: Path, *arguments: str) -> None:
    subprocess.run(["git", *arguments], cwd=project, check=True, capture_output=True)


def _commit_and_tag(project: Path, message: str, tag: str) -> None:
    _git(project, "add", "--all")
    _git(project, "commit", "-m", message)
    _git(project, "tag", tag)


def _checkpoint(root: Path, *, arm: str, completed_updates: int, probe: bool) -> Path:
    root.mkdir(parents=True)
    _write_json(root / "adapter_config.json", {"peft_type": "LORA"})
    (root / "adapter_model.safetensors").write_bytes(b"synthetic-adapter")
    if probe:
        _write_json(root / "reload_probe.json", {"passed": True})
    declared = {
        "adapter_config.json": _sha(root / "adapter_config.json"),
        "adapter_model.safetensors": _sha(root / "adapter_model.safetensors"),
        # These declarations prove that excluded historical state can remain in
        # the original manifest without entering the diagnostic bundle.
        "bridge_state.pt": "0" * 64,
        "optimizer.pt": "1" * 64,
    }
    if probe:
        declared["reload_probe.json"] = _sha(root / "reload_probe.json")
    _write_json(
        root / "checkpoint_manifest.json",
        {
            "kind": "bridge_policy_checkpoint",
            "arm": arm,
            "completed_updates": completed_updates,
            "file_sha256": declared,
        },
    )
    (root / "bridge_state.pt").write_bytes(b"must-not-be-bundled")
    (root / "optimizer.pt").write_bytes(b"must-not-be-bundled")
    return root


@pytest.fixture
def synthetic_bundle_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> DevDiagnosticBundleInputs:
    # The synthetic two-case fixture exercises the standard-library archive
    # contract. Formal creation separately invokes exact 19,200-case
    # regeneration against the parsed frozen spec.
    monkeypatch.setattr(
        deployment, "_validate_public_against_parsed_spec", lambda **_kwargs: None
    )
    project = tmp_path / "project"
    _write_text(project / "pyproject.toml", "[project]\nname='synthetic-did'\nversion='0'\n")
    _write_text(project / "README.md", "synthetic deployment fixture\n")
    _write_text(project / "requirements/h100-cu12x.lock", "PyYAML==6.0.2\n")
    _write_text(project / "scripts/run_dev_diag_remote.sh", "#!/usr/bin/env bash\nexit 0\n")
    _write_text(project / "scripts/bootstrap_dev_diag.sh", "#!/usr/bin/env bash\nexit 0\n")
    _write_text(project / "src/under_extinction/__init__.py", "\n")
    _write_text(project / "src/under_extinction/cli.py", "def main(): return 0\n")
    _write_text(project / "src/under_extinction/dev_diag.py", "DIAGNOSTIC_ID='DID-v1'\n")
    _write_text(
        project / "src/under_extinction/dev_diag_bootstrap.py",
        "def audit_dependency_closures(): return None\n",
    )
    _write_text(
        project / "src/under_extinction/dev_diag_evaluation.py",
        "def evaluate_dev_diagnostic(): return None\n",
    )
    _write_text(
        project / "src/under_extinction/dev_diag_deployment.py",
        "def verify_dev_diag_bundle(path): return path\n",
    )
    # Non-runtime/cache/secret-like paths are not selected by the exact source
    # allowlist and therefore never enter the archive.
    _write_text(project / ".env", "HF_TOKEN=not-a-real-token\n")
    _write_text(project / "src/under_extinction/__pycache__/ignored.pyc", "ignored\n")
    _write_text(project / "tests/test_unbundled.py", "ANSWER_KEY = 'not shipped'\n")

    spec = project / "configs/stage1_dev_diag_v1.yaml"
    config = project / "configs/bridge_pilot.yaml"
    _write_text(spec, "diagnostic_id: stage1_dev_diag_v1\n")
    _write_text(config, "experiment_name: historical_bridge\n")

    public = tmp_path / "model_visible"
    cases = public / "cases.jsonl"
    state = {
        "state_schema": "did_two_channel_state_v1",
        "cue_regime": "semantic",
        "channel_order": ["genuine", "proxy"],
        "routes": {
            "route_0": {"physical_route": "route_0", "route_name": "route-zero"},
            "route_1": {"physical_route": "route_1", "route_name": "route-one"},
        },
        "channels": {
            objective: {
                "objective": objective,
                "visible_name": f"{objective}-register",
                "semantic_name": f"{objective}-condition",
                "outcome_by_route": {
                    "route_0": f"{objective}-high",
                    "route_1": f"{objective}-low",
                },
                "values": {
                    f"{objective}-high": 1.0,
                    f"{objective}-low": 0.0,
                    f"{objective}-alternate": 0.0,
                    f"{objective}-unreachable": 0.25,
                },
                "alternate_low_outcome": f"{objective}-alternate",
                "unreachable_outcome": f"{objective}-unreachable",
                "reserve_link": f"{objective}-unreachable",
            }
            for objective in ("genuine", "proxy")
        },
    }
    rows = []
    for index, permutation in enumerate(("identity", "swap"), start=1):
        messages = [
            {"role": "system", "content": "Return exactly A or B."},
            {"role": "user", "content": f"Synthetic AUDIT case {index}."},
        ]
        rows.append(
            {
                "schema_version": "1.0",
                "diagnostic_id": "stage1_dev_diag_v1",
                "generator_version": "did_v1.1.0",
                "case_id": f"case-{index}",
                "label_pair_id": "synthetic-pair",
                "semantic_unit_id": "synthetic-unit",
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
                "label_permutation": permutation,
                "module": "static_latent",
                "messages": messages,
                "messages_sha256": hashlib.sha256(
                    json.dumps(messages, sort_keys=True, separators=(",", ":")).encode(
                        "utf-8"
                    )
                ).hexdigest(),
                "causal_state": state,
                "query": {"kind": "latent_objective"},
                "update_event": None,
            }
        )
    _write_text(cases, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    commitment = public / "ANSWER_KEY_COMMITMENT.json"
    _write_json(
        commitment,
        {
            "schema_version": "DID-v1",
            "kind": "did_v1_hidden_answer_key_commitment",
            "diagnostic_id": "stage1_dev_diag_v1",
            "answer_key_external_to_model_visible_bundle": True,
            "case_set_sha256": _sha(cases),
            "record_count": len(rows),
            "answer_key_sha256": "2" * 64,
        },
    )
    case_manifest = public / "MANIFEST.json"
    _write_json(
        case_manifest,
        {
            "schema_version": "DID-v1",
            "kind": "did_v1_model_visible_case_manifest",
            "diagnostic_id": "stage1_dev_diag_v1",
            "scientific_status": "post_hoc_exploratory_failure_localization",
            "generator_version": "did_v1.1.0",
            "split": "dev",
            "diagnostic_spec_sha256": "1" * 64,
            "diagnostic_spec_file_sha256": "0" * 64,
            "parents": {},
            "verified_source_parent": {
                "data_manifest_sha256": "3" * 64,
                "dev_file_sha256": "4" * 64,
                "dev_file_bytes": 1,
                "dev_record_count": 1,
            },
            "locked_test_opened_or_parsed": False,
            "existing_dev_prompts_reused": False,
            "access_contract": {
                "allowed_split": "dev",
                "other_split_access": "forbidden",
                "existing_dev_prompts_reused": False,
                "locked_test_accessed": False,
            },
            "counts": {
                "static_prompts": 2,
                "static_semantic_units": 1,
                "update_prompts": 0,
                "update_semantic_units": 0,
                "total_prompts": len(rows),
            },
            "answer_key": {
                "external": True,
                "path_disclosed": False,
                "sha256": "2" * 64,
                "count": len(rows),
            },
            "generation_subset": {
                "method": (
                    "four_hash_ranked_cases_per_panel_module_cue_renderer_"
                    "label_stratum_v1"
                ),
                "size": len(rows),
                "ordered_case_ids_sha256": hashlib.sha256(
                    json.dumps(
                        [row["case_id"] for row in rows],
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
                "case_ids": [row["case_id"] for row in rows],
            },
            "template_provenance": {
                "audit_renderer_ids": ["audit_matrix_v1", "audit_routefile_v1"],
                "calibration_renderer_ids": ["cal_sheet_v1", "cal_log_v1"],
                "renderer_template_sha256": {
                    "audit_matrix_v1": (
                        "e61f85291150241d2df88e242182afa71a82362e248deff58cd799f020eaa5a1"
                    ),
                    "audit_routefile_v1": (
                        "a74bd0b520b3afea79f14e072d2040d20f9b327496d2c26e998649e79db41dca"
                    ),
                    "cal_log_v1": (
                        "679fb800dfc1e61daf2998e913c7379f452b99ab40388515d738f920bb7b15a3"
                    ),
                    "cal_sheet_v1": (
                        "2bab8cb27812cbac1484195f24e05b6709afb6e231f5691d3407f82ff15a77e1"
                    ),
                },
                "calibration_and_audit_renderer_sets_disjoint": True,
                "calibration_not_model_scored": True,
            },
            "files": {
                "cases": {
                    "path": cases.name,
                    "bytes": cases.stat().st_size,
                    "sha256": _sha(cases),
                    "count": len(rows),
                },
                "answer_key_commitment": {
                    "path": commitment.name,
                    "bytes": commitment.stat().st_size,
                    "sha256": _sha(commitment),
                },
            },
        },
    )
    _write_json(public / "ANSWER_KEY.json", {"case-1": "A"})

    historical = tmp_path / "historical"
    dev = historical / "dev.jsonl"
    _write_text(dev, '{"split":"dev","world_id":"one"}\n')
    data_manifest = historical / "MANIFEST.json"
    _write_json(
        data_manifest,
        {
            "kind": "frozen_two_channel_choice_environment",
            "files": {
                "dev": {
                    "path": dev.name,
                    "bytes": dev.stat().st_size,
                    "sha256": _sha(dev),
                },
                # Referencing hashes is allowed; the non-DEV bytes do not exist
                # in this fixture and must never be opened or bundled.
                "train": {"path": "train.jsonl", "bytes": 1, "sha256": "3" * 64},
                "test": {"path": "test.jsonl", "bytes": 1, "sha256": "4" * 64},
            },
        },
    )

    checkpoints = tmp_path / "checkpoints"
    checkpoint_zero = _checkpoint(
        checkpoints / "checkpoint_zero", arm="genuine", completed_updates=0, probe=False
    )
    genuine_final = _checkpoint(
        checkpoints / "genuine_final", arm="genuine", completed_updates=300, probe=True
    )
    proxy_final = _checkpoint(
        checkpoints / "proxy_final", arm="proxy", completed_updates=300, probe=True
    )
    manifest_value = json.loads(case_manifest.read_text(encoding="utf-8"))
    manifest_value["diagnostic_spec_file_sha256"] = _sha(spec)
    manifest_value["parents"] = {
        "archive_sha256": "8" * 64,
        "stage1_release_tag": "stage1-dev-20260819-failed",
        "stage1_release_commit": "9" * 40,
        "historical_training_commit": "a" * 40,
        "stage1_report_sha256": "b" * 64,
        "bridge_config_file_sha256": _sha(config),
        "bridge_config_canonical_sha256": "c" * 64,
        "data_manifest_sha256": _sha(data_manifest),
        "dev_file_sha256": _sha(dev),
        "pair_seed": 11,
        "initial_environment_state_sha256": "d" * 64,
        "model_runtime_attestation_sha256": "e" * 64,
    }
    for condition, checkpoint in {
        "checkpoint_zero": checkpoint_zero,
        "genuine_final": genuine_final,
        "proxy_final": proxy_final,
    }.items():
        identity = {
            "checkpoint_zero": ("genuine", 0),
            "genuine_final": ("genuine", 300),
            "proxy_final": ("proxy", 300),
        }[condition]
        manifest_value["parents"][condition] = {
            "arm": identity[0],
            "update": identity[1],
            "checkpoint_manifest_sha256": _sha(checkpoint / "checkpoint_manifest.json"),
            "adapter_config_sha256": _sha(checkpoint / "adapter_config.json"),
            "adapter_model_sha256": _sha(checkpoint / "adapter_model.safetensors"),
        }
    _write_json(case_manifest, manifest_value)
    _git(project, "init", "-b", "main")
    _git(project, "config", "core.autocrlf", "false")
    _git(project, "config", "user.email", "did-fixture@example.invalid")
    _git(project, "config", "user.name", "DID Fixture")
    _commit_and_tag(project, "freeze synthetic DID source", "did-fixture-v1")
    return DevDiagnosticBundleInputs(
        project_root=project,
        diagnostic_spec=spec,
        case_manifest=case_manifest,
        cases=cases,
        answer_key_commitment=commitment,
        bridge_config=config,
        historical_data_manifest=data_manifest,
        dev_data=dev,
        checkpoint_zero=checkpoint_zero,
        genuine_final=genuine_final,
        proxy_final=proxy_final,
    )


def _write_synthetic_bootstrap_attestation(
    path: Path, extracted_root: Path, manifest: dict[str, object]
) -> None:
    inventory = {
        row["path"]: row
        for row in manifest["inventory"]  # type: ignore[index]
    }
    project_hashes = {
        relative: row["sha256"]
        for relative, row in inventory.items()
        if relative.startswith("project/")
    }
    python_hashes = {
        relative: digest
        for relative, digest in project_hashes.items()
        if relative.startswith("project/src/") and relative.endswith(".py")
    }
    canonical = lambda value: json.dumps(  # noqa: E731
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    venv = "/opt/did-runtime/.venv"
    provider = "/opt/lambda-stack/lib/python3.12/site-packages"
    dependency_row = lambda name, version, location, inside, policy: {  # noqa: E731
        "name": name,
        "canonical_name": name.lower(),
        "version": version,
        "location": location,
        "inside_venv": inside,
        "origin_policy": policy,
    }
    source_identity = manifest["source_identity"]
    attestation = {
        "schema_version": "1.0",
        "kind": "did_v1_remote_bootstrap_attestation",
        "passed": True,
        "created_at_utc": "2026-08-19T00:00:00+00:00",
        "bundle": {
            "manifest_sha256": _sha(extracted_root / BUNDLE_MANIFEST),
            "inventory_sha256": manifest["inventory_sha256"],
            "source_identity_sha256": hashlib.sha256(
                canonical(source_identity)
            ).hexdigest(),
            "git": source_identity["git"],  # type: ignore[index]
            "project_inventory_sha256": source_identity[  # type: ignore[index]
                "project_inventory_sha256"
            ],
            "executing_project_payload_sha256": project_hashes,
            "executing_project_payload_inventory_sha256": hashlib.sha256(
                canonical(project_hashes)
            ).hexdigest(),
        },
        "hardware": {
            "architecture": "aarch64",
            "hostname": "synthetic-gh200",
            "cuda_device_count": 1,
            "device_name": "NVIDIA GH200 480GB",
            "compute_capability": [9, 0],
            "total_memory_bytes": 100 * 1024**3,
            "total_memory_gib": 100.0,
            "nvidia_smi_query": [
                "0, GPU-synthetic, NVIDIA GH200 480GB, 570.00, 97871, 9.0"
            ],
        },
        "runtime": {
            "python": "3.12.9",
            "python_executable": f"{venv}/bin/python",
            "venv_root": venv,
            "torch": "2.7.1+cu128",
            "torch_path": f"{provider}/torch/__init__.py",
            "cuda": "12.8",
            "cudnn": 90701,
            "numpy": "1.26.4",
            "transformers": "5.15.0",
            "peft": "0.20.0",
            "qwen_text_loader": "Qwen3_5ForCausalLM",
            "module_origins": {"numpy": f"{venv}/lib/python3.12/site-packages/numpy"},
        },
        "dependency_closure": {
            "lock_sha256": inventory[
                "project/requirements/h100-cu12x.lock"
            ]["sha256"],
            "schema_version": "1.0",
            "policy": "isolated_experiment_plus_explicit_provider_torch_support_v1",
            "experiment_roots": ["PyYAML==6.0.2"],
            "provider_torch_root": "torch>=2.5,<3",
            "provider_torch_support_allowlist": [
                "flash-attn",
                "nvidia-cublas-cu12",
                "nvidia-cuda-cupti-cu12",
                "nvidia-cuda-nvcc-cu12",
                "nvidia-cuda-nvrtc-cu12",
                "nvidia-cuda-runtime-cu12",
                "nvidia-cudnn-cu12",
                "nvidia-cufft-cu12",
                "nvidia-cufile-cu12",
                "nvidia-curand-cu12",
                "nvidia-cusolver-cu12",
                "nvidia-cusparse-cu12",
                "nvidia-cusparselt-cu12",
                "nvidia-nccl-cu12",
                "nvidia-nvjitlink-cu12",
                "nvidia-nvtx-cu12",
                "optree",
                "pytorch-triton",
                "pytorch-triton-rocm",
                "torch",
                "torchaudio",
                "torchvision",
                "triton",
                "triton-kernels",
            ],
            "experiment_closure": {
                "numpy": dependency_row(
                    "numpy", "1.26.4", f"{venv}/lib/python3.12/site-packages", True,
                    "diagnostic_venv",
                ),
                "peft": dependency_row(
                    "peft", "0.20.0", f"{venv}/lib/python3.12/site-packages", True,
                    "diagnostic_venv",
                ),
                "pyyaml": dependency_row(
                    "pyyaml", "6.0.2", f"{venv}/lib/python3.12/site-packages", True,
                    "diagnostic_venv",
                ),
                "torch": dependency_row(
                    "torch", "2.7.1", provider, False, "provider_torch_boundary"
                ),
                "transformers": dependency_row(
                    "transformers", "5.15.0", f"{venv}/lib/python3.12/site-packages", True,
                    "diagnostic_venv",
                ),
            },
            "provider_torch_closure": {
                "torch": dependency_row(
                    "torch", "2.7.1", provider, False,
                    "attested_provider_torch_support",
                )
            },
            "installed_provider_allowlist_snapshot": {
                "torch": {
                    "name": "torch",
                    "canonical_name": "torch",
                    "version": "2.7.1",
                    "location": provider,
                    "inside_venv": False,
                }
            },
            "checks": {
                "all_locked_roots_version_matched": True,
                "all_non_torch_experiment_dependencies_inside_venv": True,
                "provider_torch_is_explicit_experiment_boundary": True,
                "provider_torch_closure_fully_traversed": True,
                "every_external_torch_support_distribution_allowlisted": True,
            },
        },
        "kernel_contract": {
            "delta_net_policy": "torch_fallback_required",
            "optional_delta_net_packages_present": {
                "causal-conv1d": False,
                "fla-core": False,
                "kernels": False,
            },
            "torch_numpy_cpu_abi_roundtrip": True,
            "cuda_tensor_probe": True,
            "cuda_bfloat16_matmul_probe": True,
            "cuda_sdpa_probe": True,
        },
        "source": {
            "package_path": (
                "/opt/under_extinction_dev_diag/project/src/under_extinction/__init__.py"
            ),
            "diagnostic_import_probe": True,
            "python_file_count": len(python_hashes),
            "python_file_sha256": python_hashes,
            "python_inventory_sha256": hashlib.sha256(canonical(python_hashes)).hexdigest(),
        },
    }
    _write_json(path, attestation)


def test_create_bundle_has_exact_dev_only_inventory(
    synthetic_bundle_inputs: DevDiagnosticBundleInputs, tmp_path: Path
) -> None:
    destination = tmp_path / "deployment" / "did.tar.gz"
    bundle = create_dev_diag_bundle(synthetic_bundle_inputs, destination)
    assert bundle == destination.absolute()
    assert Path(f"{bundle}.sha256").is_file()
    manifest = verify_dev_diag_bundle(bundle)
    assert manifest["contract"] == {
        "allowed_split": "dev",
        "locked_test_included": False,
        "hidden_answer_key_included": False,
        "bridge_state_included": False,
        "optimizer_state_included": False,
        "caches_included": False,
        "secrets_included": False,
        "checkpoint_conditions": ["checkpoint_zero", "genuine_final", "proxy_final"],
        "checkpoint_files": [
            "checkpoint_manifest.json",
            "adapter_config.json",
            "adapter_model.safetensors",
        ],
        "optional_checkpoint_files": ["reload_probe.json"],
        "resume_supported": False,
    }
    with tarfile.open(bundle, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
    assert len(names) == len(set(names))
    assert all(member.isreg() for member in members)
    assert f"{BUNDLE_ROOT}/{BUNDLE_MANIFEST}" in names
    assert sum(name.endswith("adapter_model.safetensors") for name in names) == 3
    assert sum(name.endswith("reload_probe.json") for name in names) == 2
    lowered = "\n".join(names).lower()
    assert "test.jsonl" not in lowered
    assert "train.jsonl" not in lowered
    assert "answer_key.json" not in lowered
    assert "bridge_state.pt" not in lowered
    assert "optimizer.pt" not in lowered
    assert "__pycache__" not in lowered
    assert "/.env" not in lowered

    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with tarfile.open(bundle, "r:gz") as archive:
        archive.extractall(extracted, filter="data")
    extracted_manifest = verify_dev_diag_bundle(extracted)
    assert extracted_manifest["inventory_sha256"] == manifest["inventory_sha256"]

    second = create_dev_diag_bundle(synthetic_bundle_inputs, tmp_path / "did-copy.tar.gz")
    assert _sha(second) == _sha(bundle)


def test_bootstrap_attestation_strictly_binds_runtime_and_bundle(
    synthetic_bundle_inputs: DevDiagnosticBundleInputs, tmp_path: Path
) -> None:
    bundle = create_dev_diag_bundle(synthetic_bundle_inputs, tmp_path / "binding.tar.gz")
    extracted_parent = tmp_path / "binding-extracted"
    extracted_parent.mkdir()
    with tarfile.open(bundle, "r:gz") as archive:
        archive.extractall(extracted_parent, filter="data")
    extracted_root = extracted_parent / BUNDLE_ROOT
    manifest = verify_dev_diag_bundle(extracted_root)
    attestation = tmp_path / "bootstrap_runtime_attestation.json"
    _write_synthetic_bootstrap_attestation(attestation, extracted_root, manifest)

    binding = verify_dev_diag_bootstrap_attestation(attestation, extracted_root)
    assert binding["kind"] == "did_v1_verified_bootstrap_binding"
    assert binding["checks"] == {
        "bundle_reverified": True,
        "source_hashes_match_bundle": True,
        "git_source_identity_bound": True,
        "exact_gh200_runtime": True,
        "dependency_closures_valid": True,
        "torch_numpy_abi_valid": True,
        "kernel_probes_valid": True,
    }
    assert binding == verify_dev_diag_bootstrap_attestation(attestation, extracted_root)

    tampered = json.loads(attestation.read_text(encoding="utf-8"))
    tampered["hardware"]["device_name"] = "NVIDIA H100 80GB HBM3"
    _write_json(attestation, tampered)
    with pytest.raises(ValueError, match="exact 1x GH200"):
        verify_dev_diag_bootstrap_attestation(attestation, extracted_root)


def test_postflight_attestation_reverification_rejects_payload_mutation(
    synthetic_bundle_inputs: DevDiagnosticBundleInputs, tmp_path: Path
) -> None:
    bundle = create_dev_diag_bundle(synthetic_bundle_inputs, tmp_path / "postflight.tar.gz")
    extracted_parent = tmp_path / "postflight-extracted"
    extracted_parent.mkdir()
    with tarfile.open(bundle, "r:gz") as archive:
        archive.extractall(extracted_parent, filter="data")
    extracted_root = extracted_parent / BUNDLE_ROOT
    manifest = verify_dev_diag_bundle(extracted_root)
    attestation = tmp_path / "postflight-attestation.json"
    _write_synthetic_bootstrap_attestation(attestation, extracted_root, manifest)
    verify_dev_diag_bootstrap_attestation(attestation, extracted_root)

    source = extracted_root / "project/src/under_extinction/dev_diag.py"
    source.write_text("MUTATED_DURING_INFERENCE = True\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_dev_diag_bootstrap_attestation(attestation, extracted_root)


@pytest.mark.parametrize(
    ("location", "extra_key"),
    [("top", "answer_key_records"), ("source_identity", "answer_payload")],
)
def test_bundle_verifier_rejects_outer_manifest_metadata_smuggling(
    location: str,
    extra_key: str,
    synthetic_bundle_inputs: DevDiagnosticBundleInputs,
    tmp_path: Path,
) -> None:
    bundle = create_dev_diag_bundle(
        synthetic_bundle_inputs, tmp_path / f"outer-{location}.tar.gz"
    )
    extracted_parent = tmp_path / f"outer-{location}-extracted"
    extracted_parent.mkdir()
    with tarfile.open(bundle, "r:gz") as archive:
        archive.extractall(extracted_parent, filter="data")
    extracted_root = extracted_parent / BUNDLE_ROOT
    manifest_path = extracted_root / BUNDLE_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target = manifest if location == "top" else manifest["source_identity"]
    target[extra_key] = {"case-1": "A"}
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="schema differs"):
        verify_dev_diag_bundle(extracted_root)


def test_bundle_fails_before_publication_when_cases_drift(
    synthetic_bundle_inputs: DevDiagnosticBundleInputs, tmp_path: Path
) -> None:
    synthetic_bundle_inputs.cases.write_text(
        synthetic_bundle_inputs.cases.read_text(encoding="utf-8")
        + '{"case_id":"case-3","split":"dev","namespace":"AUDIT"}\n',
        encoding="utf-8",
    )
    destination = tmp_path / "bad.tar.gz"
    with pytest.raises(ValueError, match="cases.bytes"):
        create_dev_diag_bundle(synthetic_bundle_inputs, destination)
    assert not destination.exists()
    assert not Path(f"{destination}.sha256").exists()


def test_bundle_rejects_public_manifest_answer_leakage_or_extra_schema(
    synthetic_bundle_inputs: DevDiagnosticBundleInputs, tmp_path: Path
) -> None:
    manifest_path = synthetic_bundle_inputs.case_manifest
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    leaked = dict(original)
    leaked["answer_key_records"] = [{"case_id": "case-1", "answer": "A"}]
    _write_json(manifest_path, leaked)
    with pytest.raises(ValueError, match="case manifest schema differs"):
        create_dev_diag_bundle(synthetic_bundle_inputs, tmp_path / "top-level-leak.tar.gz")

    nested = json.loads(json.dumps(original))
    nested["parents"]["answer_payload"] = {"case-1": "A"}
    _write_json(manifest_path, nested)
    with pytest.raises(ValueError, match="manifest.parents schema differs"):
        create_dev_diag_bundle(synthetic_bundle_inputs, tmp_path / "nested-leak.tar.gz")

    renderer_extension = json.loads(json.dumps(original))
    renderer_extension["template_provenance"]["renderer_template_sha256"][
        "answer_renderer"
    ] = "f" * 64
    _write_json(manifest_path, renderer_extension)
    with pytest.raises(ValueError, match="differs from frozen renderers"):
        create_dev_diag_bundle(
            synthetic_bundle_inputs, tmp_path / "renderer-extension.tar.gz"
        )


def test_parsed_spec_value_binding_rejects_same_schema_parent_mutation(
    synthetic_bundle_inputs: DevDiagnosticBundleInputs,
) -> None:
    manifest = json.loads(
        synthetic_bundle_inputs.case_manifest.read_text(encoding="utf-8")
    )
    expected = {
        "parents": json.loads(json.dumps(manifest["parents"])),
        "template_provenance": json.loads(
            json.dumps(manifest["template_provenance"])
        ),
        "generation_subset": json.loads(json.dumps(manifest["generation_subset"])),
    }
    manifest["parents"]["bridge_config_canonical_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="parsed frozen spec for: parents"):
        deployment._require_manifest_matches_parsed_spec(manifest, expected)


def test_formal_content_commitment_rejects_answer_text_rehash() -> None:
    deployment._validate_frozen_formal_content_commitments(
        count=19_200,
        cases_sha256=(
            "a7750246e2701e024fc13d25f975bebf141eb8bfbadf9431c1fd575da2b66173"
        ),
        answer_key_sha256=(
            "c158cafcfe19016161319ec3e152fd89a2a51b714af9ca88e7ff19c7ccc58353"
        ),
        subset_ids_sha256=(
            "edb69a50b7c1870600971c078f92ca4df5f77558b5966c61c62351556d38cefb"
        ),
    )
    with pytest.raises(ValueError, match="differ from the freeze"):
        deployment._validate_frozen_formal_content_commitments(
            count=19_200,
            # Any otherwise-schema-valid message edit changes this digest.
            cases_sha256="0" * 64,
            answer_key_sha256=(
                "c158cafcfe19016161319ec3e152fd89a2a51b714af9ca88e7ff19c7ccc58353"
            ),
            subset_ids_sha256=(
                "edb69a50b7c1870600971c078f92ca4df5f77558b5966c61c62351556d38cefb"
            ),
        )


@pytest.mark.parametrize("smuggled_field", ["answer", "solution_hint"])
def test_bundle_rejects_nested_case_answer_smuggling_even_when_rehashed(
    smuggled_field: str,
    synthetic_bundle_inputs: DevDiagnosticBundleInputs,
    tmp_path: Path,
) -> None:
    cases_path = synthetic_bundle_inputs.cases
    rows = [
        json.loads(line)
        for line in cases_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows[0]["query"][smuggled_field] = "A"
    _write_text(
        cases_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )
    commitment_path = synthetic_bundle_inputs.answer_key_commitment
    commitment = json.loads(commitment_path.read_text(encoding="utf-8"))
    commitment["case_set_sha256"] = _sha(cases_path)
    _write_json(commitment_path, commitment)
    manifest_path = synthetic_bundle_inputs.case_manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["cases"].update(
        {
            "bytes": cases_path.stat().st_size,
            "sha256": _sha(cases_path),
        }
    )
    manifest["files"]["answer_key_commitment"].update(
        {
            "bytes": commitment_path.stat().st_size,
            "sha256": _sha(commitment_path),
        }
    )
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="query schema differs"):
        create_dev_diag_bundle(
            synthetic_bundle_inputs,
            tmp_path / f"nested-case-{smuggled_field}.tar.gz",
        )


def test_bundle_fails_if_checkpoint_manifest_does_not_bind_adapter(
    synthetic_bundle_inputs: DevDiagnosticBundleInputs, tmp_path: Path
) -> None:
    (synthetic_bundle_inputs.proxy_final / "adapter_model.safetensors").write_bytes(b"drift")
    with pytest.raises(ValueError, match="does not bind adapter_model.safetensors"):
        create_dev_diag_bundle(synthetic_bundle_inputs, tmp_path / "bad-checkpoint.tar.gz")


def test_bundle_fails_if_spec_or_checkpoint_identity_differs(
    synthetic_bundle_inputs: DevDiagnosticBundleInputs, tmp_path: Path
) -> None:
    synthetic_bundle_inputs.diagnostic_spec.write_text(
        "diagnostic_id: changed-after-case-freeze\n", encoding="utf-8"
    )
    _commit_and_tag(
        synthetic_bundle_inputs.project_root,
        "mutate spec for rejection test",
        "did-fixture-bad-spec",
    )
    with pytest.raises(ValueError, match="exact spec file"):
        create_dev_diag_bundle(synthetic_bundle_inputs, tmp_path / "bad-spec.tar.gz")

    # Restore the fixture's spec and its cross-binding, then make G300 claim the
    # wrong update. The deployment layer must catch this before archiving.
    synthetic_bundle_inputs.diagnostic_spec.write_text(
        "diagnostic_id: stage1_dev_diag_v1\n", encoding="utf-8"
    )
    _commit_and_tag(
        synthetic_bundle_inputs.project_root,
        "restore spec for checkpoint test",
        "did-fixture-restored-spec",
    )
    checkpoint_manifest = synthetic_bundle_inputs.genuine_final / "checkpoint_manifest.json"
    value = json.loads(checkpoint_manifest.read_text(encoding="utf-8"))
    value["completed_updates"] = 299
    _write_json(checkpoint_manifest, value)
    with pytest.raises(ValueError, match="required arm/update"):
        create_dev_diag_bundle(synthetic_bundle_inputs, tmp_path / "bad-update.tar.gz")


def test_bundle_requires_clean_tagged_git_source_but_ignores_retrieval_artifacts(
    synthetic_bundle_inputs: DevDiagnosticBundleInputs, tmp_path: Path
) -> None:
    source = synthetic_bundle_inputs.project_root / "src/under_extinction/cli.py"
    source.write_text("def main(): return 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean tagged source worktree"):
        create_dev_diag_bundle(synthetic_bundle_inputs, tmp_path / "dirty.tar.gz")

    _git(synthetic_bundle_inputs.project_root, "restore", "src/under_extinction/cli.py")
    ignored = synthetic_bundle_inputs.project_root / "retrieved/prior/result.log"
    _write_text(ignored, "prior retrieved evidence\n")
    archive = create_dev_diag_bundle(
        synthetic_bundle_inputs, tmp_path / "clean-with-retrieval.tar.gz"
    )
    manifest = verify_dev_diag_bundle(archive)
    assert manifest["source_identity"]["git"]["worktree_clean_for_bundle"] is True
    assert manifest["source_identity"]["git"]["ignored_untracked_paths"] == [
        "retrieved/prior/result.log"
    ]
    assert not any(
        row["path"].endswith("prior/result.log") for row in manifest["inventory"]
    )


def test_bundle_rejects_ignored_untracked_python_payload(
    synthetic_bundle_inputs: DevDiagnosticBundleInputs, tmp_path: Path
) -> None:
    project = synthetic_bundle_inputs.project_root
    _write_text(project / ".gitignore", "src/under_extinction/ignored_generated.py\n")
    _commit_and_tag(project, "freeze ignore rule", "did-fixture-ignore-rule")
    ignored_python = project / "src/under_extinction/ignored_generated.py"
    _write_text(ignored_python, "SHOULD_NEVER_SHIP = True\n")
    assert not subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    with pytest.raises(ValueError, match="not Git-tracked at HEAD"):
        create_dev_diag_bundle(
            synthetic_bundle_inputs, tmp_path / "ignored-python.tar.gz"
        )


@pytest.mark.parametrize("unsafe_name", ["../escape", "/absolute"])
def test_archive_verifier_rejects_traversal_or_absolute_member(
    unsafe_name: str, tmp_path: Path
) -> None:
    archive_path = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo(unsafe_name)
        info.size = 1
        archive.addfile(info, fileobj=__import__("io").BytesIO(b"x"))
    with pytest.raises(ValueError, match="safe"):
        verify_dev_diag_bundle(archive_path)


def test_archive_verifier_rejects_links(tmp_path: Path) -> None:
    archive_path = tmp_path / "link.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo(f"{BUNDLE_ROOT}/linked")
        info.type = tarfile.SYMTYPE
        info.linkname = "../outside"
        archive.addfile(info)
    with pytest.raises(ValueError, match="non-regular"):
        verify_dev_diag_bundle(archive_path)


def test_collect_and_verify_failed_partial_results_without_weights(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    _write_text(evidence / "logs/remote.log", "evaluator stopped after two policies\n")
    _write_json(evidence / "inference/run_manifest.json", {"state": "FAILED"})
    _write_text(
        evidence / "inference/checkpoint_zero/predictions.jsonl",
        '{"case_id":"case-1","probability_A":0.5}\n',
    )
    destination = tmp_path / "retrieval" / "partial.tar.gz"
    archive = collect_dev_diag_results(evidence, destination)
    manifest = verify_dev_diag_results(archive)
    assert manifest["run_status"]["reported_state"] == "FAILED"
    assert manifest["run_status"]["complete_claimed_by_run"] is False
    assert manifest["run_status"]["partial_and_failed_runs_are_retrievable"] is True
    with tarfile.open(archive, "r:gz") as handle:
        names = [member.name for member in handle.getmembers()]
    assert all(name.startswith(f"{RESULTS_ROOT}/") for name in names)
    assert not any(name.endswith((".safetensors", ".pt", ".bin")) for name in names)


def test_result_collector_refuses_weights_and_secret_like_logs(tmp_path: Path) -> None:
    with_weight = tmp_path / "weight-evidence"
    _write_text(with_weight / "logs/remote.log", "partial\n")
    weight = with_weight / "inference/adapter_model.safetensors"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"weight")
    destination = tmp_path / "must-not-exist.tar.gz"
    with pytest.raises(ValueError, match="non-evidence file"):
        collect_dev_diag_results(with_weight, destination)
    assert not destination.exists()

    with_secret = tmp_path / "secret-evidence"
    _write_text(with_secret / "logs/remote.log", "token hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890\n")
    with pytest.raises(ValueError, match="secret-like"):
        collect_dev_diag_results(with_secret, tmp_path / "secret.tar.gz")


def test_remote_runner_is_dev_only_immutable_and_does_not_claim_resume() -> None:
    script = (PROJECT_ROOT / "scripts/run_dev_diag_remote.sh").read_text(encoding="utf-8")
    bootstrap = (PROJECT_ROOT / "scripts/bootstrap_dev_diag.sh").read_text(encoding="utf-8")
    closure_policy = (
        PROJECT_ROOT / "src/under_extinction/dev_diag_bootstrap.py"
    ).read_text(encoding="utf-8")
    assert "bridge-dev-diag-evaluate" in script
    assert 'inputs/historical/dev.jsonl"' in script
    assert "inputs/public/ANSWER_KEY_COMMITMENT.json" in script
    assert "inputs/checkpoints/checkpoint_zero" in script
    assert "inputs/checkpoints/genuine_final" in script
    assert "inputs/checkpoints/proxy_final" in script
    assert "verify_dev_diag_bundle" in script
    assert "collect_dev_diag_results" in script
    assert "bootstrap_dev_diag.sh" in script
    assert "requirements/h100-cu12x.lock" in bootstrap
    assert "PYTHONDONTWRITEBYTECODE=1" in script
    assert "--resume" not in script
    assert "test.jsonl" not in script.lower()
    assert "train.jsonl" not in script.lower()
    assert "bridge_state.pt" not in script.lower()
    assert '"$(uname -m)" != "aarch64"' in bootstrap
    assert 'name != "NVIDIA GH200 480GB"' in bootstrap
    assert "torch.cuda.device_count() != 1" in bootstrap
    assert "memory_gib < 90" in bootstrap
    assert "Experiment/provider dependency closure is inconsistent" in closure_policy
    assert "leaked from outside the venv" in bootstrap
    assert "torch.from_numpy" in bootstrap
    assert "scaled_dot_product_attention" in bootstrap
    assert "torch_fallback_required" in bootstrap
    assert "did_v1_remote_bootstrap_attestation" in bootstrap
    assert "diagnostic_import_probe" in bootstrap
    assert "verify_dev_diag_bootstrap_attestation" in script
    assert "bootstrap_binding_preflight.json" in script
    assert "bootstrap_binding_postflight.json" in script
    assert 'chmod -R a-w "$BUNDLE_ROOT"' in script
