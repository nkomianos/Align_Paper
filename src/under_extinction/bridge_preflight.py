"""Hash-bound handoff from the paid bridge smoke test to formal Stage 1."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .bridge_analysis import verify_bridge_gate_report
from .bridge_budget import (
    build_stage1_cost_projection,
    verify_stage1_cost_projection,
)
from .config import config_hash, output_root
from .io import canonical_json, sha256_file, write_json
from .manifest import environment_snapshot, project_hash
from .modeling import (
    QWEN35_TEXT_LOADER,
    compact_model_runtime_contract,
    verify_model_runtime_attestation,
)


PREFLIGHT_KIND = "bridge_paid_preflight_pass"
RUNTIME_FIELDS = (
    "python",
    "platform",
    "machine",
    "packages",
    "torch",
    "cuda_available",
    "torch_cuda",
    "cudnn",
    "gpu",
    "bf16_supported",
)
RUNTIME_ENVIRONMENT_FIELDS = (
    "UE_INSTANCE_ID",
    "UE_INSTANCE_TYPE",
    "UE_LAMBDA_IMAGE_ID",
)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _assert_loaded_config_hash(config: Mapping[str, Any], *, label: str) -> None:
    claimed = config.get("_config_sha256")
    actual = config_hash(dict(config))
    if claimed != actual:
        raise ValueError(f"{label} config was modified after loading")


def bridge_runtime_identity(snapshot: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Select stable runtime fields; deadlines and cache paths are intentionally excluded."""
    source = dict(snapshot) if snapshot is not None else environment_snapshot()
    identity = {key: source.get(key) for key in RUNTIME_FIELDS}
    safe_environment = source.get("safe_environment")
    if not isinstance(safe_environment, Mapping):
        safe_environment = {}
    identity["safe_environment"] = {
        key: safe_environment[key]
        for key in RUNTIME_ENVIRONMENT_FIELDS
        if key in safe_environment
    }
    return json.loads(canonical_json(identity))


def _frozen_data_attestation(config: Mapping[str, Any]) -> dict[str, Any]:
    data_dir = output_root(dict(config)) / "data"
    manifest_path = data_dir / "MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Formal frozen-data manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("config_sha256") != config["_config_sha256"]:
        raise ValueError("Formal frozen data does not match the Stage 1 config")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("Formal frozen-data manifest has no file inventory")
    verified: dict[str, str] = {}
    for item in files.values():
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
            raise ValueError("Malformed formal frozen-data file inventory")
        path = data_dir / item["path"]
        expected = item.get("sha256")
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Formal frozen-data hash mismatch: {path}")
        verified[str(item["path"])] = str(expected)
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "file_sha256": dict(sorted(verified.items())),
    }


def write_bridge_preflight_attestation(
    smoke_config: Mapping[str, Any],
    stage1_config: Mapping[str, Any],
    *,
    smoke_report_path: str | Path,
    destination: str | Path | None = None,
) -> Path:
    """Create a PASS artifact only after re-verifying smoke evidence and formal data."""
    _assert_loaded_config_hash(smoke_config, label="Smoke")
    _assert_loaded_config_hash(stage1_config, label="Stage 1")
    smoke_root = Path(str(smoke_config["_config_path"])).resolve().parent.parent
    stage1_root = Path(str(stage1_config["_config_path"])).resolve().parent.parent
    if smoke_root != stage1_root:
        raise ValueError("Smoke and Stage 1 configs must belong to the same project tree")
    if smoke_config["model"] != stage1_config["model"]:
        raise ValueError("Smoke and Stage 1 must use the exact same pinned model specification")
    report = Path(smoke_report_path).resolve()
    if not report.is_file() or not report.is_relative_to(smoke_root):
        raise ValueError("Smoke report must be an existing artifact inside this project")
    gate = verify_bridge_gate_report(smoke_config, report, required="smoke")
    if gate.get("pass") is not True:
        raise ValueError(f"Cannot attest a failed bridge smoke gate: {gate.get('failures')}")
    smoke_report = json.loads(report.read_text(encoding="utf-8"))
    raw_model_runtime = smoke_report.get("model_runtime_attestation")
    qwen35_required = (
        dict(stage1_config.get("model") or {}).get("loader_class")
        == QWEN35_TEXT_LOADER
    )
    model_runtime_attestation: dict[str, Any] | None = None
    model_runtime_contract: dict[str, Any] | None = None
    if raw_model_runtime is not None:
        if not isinstance(raw_model_runtime, Mapping):
            raise ValueError("Smoke report has a malformed model runtime attestation")
        model_runtime_attestation = verify_model_runtime_attestation(
            smoke_config, raw_model_runtime
        )
        verify_model_runtime_attestation(stage1_config, model_runtime_attestation)
        model_runtime_contract = compact_model_runtime_contract(
            stage1_config, model_runtime_attestation
        )
    elif qwen35_required:
        raise ValueError(
            "Qwen3.5 paid smoke report lacks the required model runtime attestation"
        )

    stage1_cost_projection = build_stage1_cost_projection(
        smoke_config,
        stage1_config,
        smoke_report_path=report,
    )
    if stage1_cost_projection.get("pass") is not True:
        failed = [
            name
            for name, passed in stage1_cost_projection.get("checks", {}).items()
            if not passed
        ]
        raise ValueError(
            "Exact-model smoke does not authorize Stage 1: " + ", ".join(failed)
        )

    runtime = bridge_runtime_identity()
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": PREFLIGHT_KIND,
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "smoke_config_path": str(Path(str(smoke_config["_config_path"])).resolve()),
        "smoke_config_sha256": str(smoke_config["_config_sha256"]),
        "stage1_config_path": str(Path(str(stage1_config["_config_path"])).resolve()),
        "stage1_config_sha256": str(stage1_config["_config_sha256"]),
        "model": json.loads(canonical_json(stage1_config["model"])),
        "model_sha256": _sha256_json(stage1_config["model"]),
        "model_runtime_attestation": model_runtime_attestation,
        "model_runtime_attestation_sha256": (
            None
            if model_runtime_attestation is None
            else model_runtime_attestation["attestation_sha256"]
        ),
        "model_runtime_contract": model_runtime_contract,
        "project_tree_sha256": project_hash(stage1_root),
        "runtime_identity": runtime,
        "runtime_identity_sha256": _sha256_json(runtime),
        "smoke_report_path": str(report),
        "smoke_report_sha256": sha256_file(report),
        "stage1_frozen_data": _frozen_data_attestation(stage1_config),
        "stage1_cost_projection": stage1_cost_projection,
    }
    payload["attestation_sha256"] = _sha256_json(payload)
    target = (
        Path(destination).resolve()
        if destination is not None
        else output_root(dict(stage1_config)) / "PREFLIGHT_PASS.json"
    )
    write_json(target, payload)
    return target


def verify_bridge_preflight_attestation(
    smoke_config: Mapping[str, Any],
    stage1_config: Mapping[str, Any],
    *,
    attestation_path: str | Path,
) -> dict[str, Any]:
    """Fail closed unless the prior smoke pass matches this exact Stage 1 invocation."""
    _assert_loaded_config_hash(smoke_config, label="Smoke")
    _assert_loaded_config_hash(stage1_config, label="Stage 1")
    path = Path(attestation_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing required bridge preflight PASS artifact: {path}")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    claimed_digest = artifact.get("attestation_sha256")
    unsigned = dict(artifact)
    unsigned.pop("attestation_sha256", None)
    failures: list[str] = []
    if artifact.get("kind") != PREFLIGHT_KIND or artifact.get("status") != "PASS":
        failures.append("wrong attestation kind/status")
    if claimed_digest != _sha256_json(unsigned):
        failures.append("attestation self-hash mismatch")
    if artifact.get("smoke_config_sha256") != smoke_config["_config_sha256"]:
        failures.append("smoke config hash mismatch")
    if artifact.get("stage1_config_sha256") != stage1_config["_config_sha256"]:
        failures.append("Stage 1 config hash mismatch")
    if artifact.get("smoke_config_path") != str(
        Path(str(smoke_config["_config_path"])).resolve()
    ):
        failures.append("smoke config path mismatch")
    if artifact.get("stage1_config_path") != str(
        Path(str(stage1_config["_config_path"])).resolve()
    ):
        failures.append("Stage 1 config path mismatch")
    if artifact.get("model") != stage1_config["model"] or artifact.get(
        "model_sha256"
    ) != _sha256_json(stage1_config["model"]):
        failures.append("pinned model mismatch")
    raw_model_runtime = artifact.get("model_runtime_attestation")
    qwen35_required = (
        dict(stage1_config.get("model") or {}).get("loader_class")
        == QWEN35_TEXT_LOADER
    )
    if raw_model_runtime is None:
        if qwen35_required:
            failures.append("Qwen3.5 model runtime attestation is missing")
    elif not isinstance(raw_model_runtime, Mapping):
        failures.append("model runtime attestation is malformed")
    else:
        try:
            verified_runtime = verify_model_runtime_attestation(
                stage1_config, raw_model_runtime
            )
            expected_contract = compact_model_runtime_contract(
                stage1_config, verified_runtime
            )
            if (
                artifact.get("model_runtime_attestation_sha256")
                != verified_runtime["attestation_sha256"]
                or artifact.get("model_runtime_contract") != expected_contract
            ):
                failures.append("model runtime attestation/contract mismatch")
        except (TypeError, ValueError) as exc:
            failures.append(f"model runtime attestation failed: {exc}")

    project_root = Path(str(stage1_config["_config_path"])).resolve().parent.parent
    if Path(str(smoke_config["_config_path"])).resolve().parent.parent != project_root:
        failures.append("configs belong to different project trees")
    if artifact.get("project_tree_sha256") != project_hash(project_root):
        failures.append("project tree changed after paid preflight")
    runtime = bridge_runtime_identity()
    if artifact.get("runtime_identity") != runtime or artifact.get(
        "runtime_identity_sha256"
    ) != _sha256_json(runtime):
        failures.append("runtime changed after paid preflight")

    report = Path(str(artifact.get("smoke_report_path", ""))).resolve()
    if (
        not report.is_file()
        or not report.is_relative_to(project_root)
        or artifact.get("smoke_report_sha256") != sha256_file(report)
    ):
        failures.append("smoke report is missing, moved outside the project, or changed")
    else:
        gate = verify_bridge_gate_report(smoke_config, report, required="smoke")
        if gate.get("pass") is not True:
            failures.append("smoke gate no longer verifies")
        else:
            current_report = json.loads(report.read_text(encoding="utf-8"))
            if current_report.get("model_runtime_attestation") != raw_model_runtime:
                failures.append("smoke report model runtime attestation changed")
    if artifact.get("stage1_frozen_data") != _frozen_data_attestation(stage1_config):
        failures.append("formal frozen data changed after paid preflight")
    stored_projection = artifact.get("stage1_cost_projection")
    if not isinstance(stored_projection, Mapping):
        failures.append("Stage-1 cost projection is missing")
    elif report.is_file():
        try:
            verify_stage1_cost_projection(
                stored_projection,
                smoke_config,
                stage1_config,
                smoke_report_path=report,
            )
        except (FileNotFoundError, TypeError, ValueError) as exc:
            failures.append(f"Stage-1 cost projection failed: {exc}")
    if failures:
        raise ValueError("Bridge preflight attestation failed: " + "; ".join(failures))
    return {
        "pass": True,
        "attestation_path": str(path),
        "attestation_sha256": sha256_file(path),
        "project_tree_sha256": artifact["project_tree_sha256"],
        "runtime_identity_sha256": artifact["runtime_identity_sha256"],
        "model_runtime_attestation_sha256": artifact.get(
            "model_runtime_attestation_sha256"
        ),
        "guarded_stage1_seconds": artifact["stage1_cost_projection"][
            "projection"
        ]["guarded_total_seconds"],
        "guarded_stage1_cost_usd": artifact["stage1_cost_projection"][
            "projection"
        ]["guarded_cost_usd"],
    }
