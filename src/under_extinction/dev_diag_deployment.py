"""Fail-closed deployment and retrieval archives for the DID-v1 DEV diagnostic.

This module is intentionally separate from the historical bridge deployment
layer.  A DID bundle contains the three frozen adapters required for inference,
but it never contains optimizer/environment state, the hidden answer key, or any
non-DEV data file.  The bundle verifier uses only the Python standard library so
it can run before the pinned GPU environment is bootstrapped.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Mapping, Sequence


BUNDLE_SCHEMA_VERSION = "1.0"
BUNDLE_KIND = "did_v1_dev_only_deployment_bundle"
BUNDLE_ROOT = "under_extinction_dev_diag"
BUNDLE_MANIFEST = "DEV_DIAG_BUNDLE_MANIFEST.json"

RESULTS_KIND = "did_v1_dev_only_result_bundle"
RESULTS_ROOT = "under_extinction_dev_diag_results"
RESULTS_MANIFEST = "DEV_DIAG_RESULTS_MANIFEST.json"

CHECKPOINT_CONDITIONS = ("checkpoint_zero", "genuine_final", "proxy_final")
CHECKPOINT_IDENTITIES = {
    "checkpoint_zero": {"arm": "genuine", "completed_updates": 0},
    "genuine_final": {"arm": "genuine", "completed_updates": 300},
    "proxy_final": {"arm": "proxy", "completed_updates": 300},
}
REQUIRED_CHECKPOINT_FILES = (
    "checkpoint_manifest.json",
    "adapter_config.json",
    "adapter_model.safetensors",
)
OPTIONAL_CHECKPOINT_FILES = ("reload_probe.json",)

MAX_ARCHIVE_MEMBERS = 4096
MAX_ARCHIVE_BYTES = 8 * 1024**3
_HASH_CHUNK_SIZE = 1024 * 1024
_ALLOWED_RESULT_SUFFIXES = {".json", ".jsonl", ".log", ".txt", ".sha256"}
_FORBIDDEN_PATH_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".hf_cache",
    "hf_cache",
    ".torch_cache",
    "torch_cache",
    ".triton_cache",
    "triton_cache",
    "wandb",
}
_FORBIDDEN_BUNDLE_BASENAMES = {
    ".env",
    "answer_key.json",
    "bridge_state.pt",
    "optimizer.pt",
    "optimizer.bin",
    "scheduler.pt",
    "trainer_state.json",
    "test.jsonl",
    "train.jsonl",
}
_FORBIDDEN_RESULT_SUFFIXES = {
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".key",
    ".pem",
}
_FORBIDDEN_ANSWER_FIELDS = {
    "answer",
    "answer_key",
    "correct",
    "correct_action",
    "expected_action",
    "expected_actions",
    "oracle_action",
    "oracle_actions",
    "target",
    "target_action",
}
_PUBLIC_CASE_KEYS = {
    "schema_version",
    "diagnostic_id",
    "generator_version",
    "case_id",
    "label_pair_id",
    "semantic_unit_id",
    "namespace",
    "split",
    "panel",
    "cue_regime",
    "renderer_id",
    "role_assignment",
    "updated_channel",
    "family",
    "mode",
    "direction",
    "time",
    "encoding",
    "query_head",
    "explicit_objective",
    "label_permutation",
    "module",
    "messages",
    "messages_sha256",
    "causal_state",
    "query",
    "update_event",
}
_PARENT_KEYS = {
    "archive_sha256",
    "stage1_release_tag",
    "stage1_release_commit",
    "historical_training_commit",
    "stage1_report_sha256",
    "bridge_config_file_sha256",
    "bridge_config_canonical_sha256",
    "data_manifest_sha256",
    "dev_file_sha256",
    "pair_seed",
    "initial_environment_state_sha256",
    "model_runtime_attestation_sha256",
    "checkpoint_zero",
    "genuine_final",
    "proxy_final",
}
_CHECKPOINT_PARENT_KEYS = {
    "arm",
    "update",
    "checkpoint_manifest_sha256",
    "adapter_config_sha256",
    "adapter_model_sha256",
}
_GENERATION_SUBSET_METHOD = (
    "four_hash_ranked_cases_per_panel_module_cue_renderer_label_stratum_v1"
)
_FROZEN_FORMAL_CASES_SHA256 = (
    "a7750246e2701e024fc13d25f975bebf141eb8bfbadf9431c1fd575da2b66173"
)
_FROZEN_FORMAL_ANSWER_KEY_SHA256 = (
    "c158cafcfe19016161319ec3e152fd89a2a51b714af9ca88e7ff19c7ccc58353"
)
_FROZEN_FORMAL_SUBSET_IDS_SHA256 = (
    "edb69a50b7c1870600971c078f92ca4df5f77558b5966c61c62351556d38cefb"
)
_FROZEN_TEMPLATE_PROVENANCE = {
    "audit_renderer_ids": ["audit_matrix_v1", "audit_routefile_v1"],
    "calibration_renderer_ids": ["cal_sheet_v1", "cal_log_v1"],
    "renderer_template_sha256": {
        "audit_matrix_v1": "e61f85291150241d2df88e242182afa71a82362e248deff58cd799f020eaa5a1",
        "audit_routefile_v1": "a74bd0b520b3afea79f14e072d2040d20f9b327496d2c26e998649e79db41dca",
        "cal_log_v1": "679fb800dfc1e61daf2998e913c7379f452b99ab40388515d738f920bb7b15a3",
        "cal_sheet_v1": "2bab8cb27812cbac1484195f24e05b6709afb6e231f5691d3407f82ff15a77e1",
    },
    "calibration_and_audit_renderer_sets_disjoint": True,
    "calibration_not_model_scored": True,
}
_ALLOWED_UNTRACKED_GIT_PREFIXES = ("artifacts/", "deployment/", "retrieved/")
PROVIDER_TORCH_SUPPORT_ALLOWLIST = frozenset(
    {
        "torch",
        "torchvision",
        "torchaudio",
        "triton",
        "pytorch-triton",
        "pytorch-triton-rocm",
        "triton-kernels",
        "optree",
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
    }
)
BOOTSTRAP_ATTESTATION_SCHEMA_VERSION = "1.0"
BOOTSTRAP_ATTESTATION_KIND = "did_v1_remote_bootstrap_attestation"
VERIFIED_BOOTSTRAP_BINDING_KIND = "did_v1_verified_bootstrap_binding"
DEPENDENCY_CLOSURE_POLICY = (
    "isolated_experiment_plus_explicit_provider_torch_support_v1"
)
_SECRET_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        rb"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----",
        rb"\bhf_[A-Za-z0-9]{30,}\b",
        rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b",
        rb"\bAKIA[0-9A-Z]{16}\b",
    )
)


@dataclass(frozen=True)
class DevDiagnosticBundleInputs:
    """Exact local inputs admitted to a DEV-only diagnostic bundle."""

    project_root: Path
    diagnostic_spec: Path
    case_manifest: Path
    cases: Path
    answer_key_commitment: Path
    bridge_config: Path
    historical_data_manifest: Path
    dev_data: Path
    checkpoint_zero: Path
    genuine_final: Path
    proxy_final: Path


@dataclass(frozen=True)
class _ArchiveInput:
    source: Path
    path: str
    role: str
    executable: bool = False


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _pretty_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _sha256_stream(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    while True:
        chunk = handle.read(_HASH_CHUNK_SIZE)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _sha256_path(path: Path) -> str:
    with path.open("rb") as handle:
        return _sha256_stream(handle)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_relative_path(value: str, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"{label} is not a safe POSIX path: {value!r}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or ":" in path.parts[0]
    ):
        raise ValueError(f"{label} is not a safe relative path: {value!r}")
    return path


def _assert_no_casefold_collisions(paths: Sequence[str], *, label: str) -> None:
    if len(paths) != len(set(paths)):
        raise ValueError(f"{label} contains duplicate paths")
    folded = [path.casefold() for path in paths]
    if len(folded) != len(set(folded)):
        raise ValueError(f"{label} contains case-insensitive path collisions")


def _checked_regular_file(path: str | Path, *, label: str) -> Path:
    source = Path(path).absolute()
    if source.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {source}")
    if any(parent.is_symlink() for parent in source.parents):
        raise ValueError(f"{label} must not traverse a symlinked directory: {source}")
    try:
        mode = source.lstat().st_mode
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing {label}: {source}") from exc
    if not stat.S_ISREG(mode):
        raise ValueError(f"{label} is not a regular file: {source}")
    return source


def _checked_directory(path: str | Path, *, label: str) -> Path:
    directory = Path(path).absolute()
    if (
        directory.is_symlink()
        or any(parent.is_symlink() for parent in directory.parents)
        or not directory.is_dir()
    ):
        raise ValueError(f"{label} must be a real directory: {directory}")
    return directory


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot parse {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object: {path}")
    return value


def _assert_no_secret_bytes(path: Path, *, label: str) -> None:
    overlap = b""
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_HASH_CHUNK_SIZE)
            if not chunk:
                break
            probe = overlap + chunk
            if any(pattern.search(probe) for pattern in _SECRET_PATTERNS):
                raise ValueError(f"{label} contains secret-like material: {path}")
            overlap = probe[-256:]


def _reject_embedded_answer_material(value: Any, *, label: str) -> None:
    forbidden = {
        "answer_key_records",
        "correct_action",
        "expected_action",
        "expected_actions",
        "expected_by_policy",
        "gold_answer",
        "oracle_action",
        "oracle_actions",
    }
    if isinstance(value, Mapping):
        overlap = forbidden & {str(key).casefold() for key in value}
        if overlap:
            raise ValueError(f"{label} embeds forbidden answer material: {sorted(overlap)}")
        for key, nested in value.items():
            _reject_embedded_answer_material(nested, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_embedded_answer_material(nested, label=f"{label}[{index}]")


def _require_exact_mapping_keys(value: Any, expected: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} schema differs")
    return value


def _validate_public_metadata_schema(
    manifest: Mapping[str, Any], commitment: Mapping[str, Any]
) -> None:
    _require_exact_mapping_keys(
        manifest,
        {
            "schema_version",
            "kind",
            "diagnostic_id",
            "scientific_status",
            "generator_version",
            "split",
            "diagnostic_spec_sha256",
            "diagnostic_spec_file_sha256",
            "parents",
            "verified_source_parent",
            "access_contract",
            "counts",
            "files",
            "answer_key",
            "generation_subset",
            "template_provenance",
            "locked_test_opened_or_parsed",
            "existing_dev_prompts_reused",
        },
        label="diagnostic case manifest",
    )
    _require_exact_mapping_keys(
        commitment,
        {
            "schema_version",
            "kind",
            "diagnostic_id",
            "case_set_sha256",
            "answer_key_sha256",
            "record_count",
            "answer_key_external_to_model_visible_bundle",
        },
        label="answer-key commitment",
    )
    parents = _require_exact_mapping_keys(
        manifest.get("parents"), _PARENT_KEYS, label="manifest.parents"
    )
    for condition in CHECKPOINT_CONDITIONS:
        _require_exact_mapping_keys(
            parents.get(condition),
            _CHECKPOINT_PARENT_KEYS,
            label=f"manifest.parents.{condition}",
        )
    files = _require_exact_mapping_keys(
        manifest.get("files"), {"cases", "answer_key_commitment"}, label="manifest.files"
    )
    _require_exact_mapping_keys(
        files.get("cases"), {"path", "sha256", "bytes", "count"}, label="manifest.files.cases"
    )
    _require_exact_mapping_keys(
        files.get("answer_key_commitment"),
        {"path", "sha256", "bytes"},
        label="manifest.files.answer_key_commitment",
    )
    _require_exact_mapping_keys(
        manifest.get("answer_key"),
        {"sha256", "count", "external", "path_disclosed"},
        label="manifest.answer_key",
    )
    _require_exact_mapping_keys(
        manifest.get("access_contract"),
        {
            "allowed_split",
            "other_split_access",
            "existing_dev_prompts_reused",
            "locked_test_accessed",
        },
        label="manifest.access_contract",
    )
    _require_exact_mapping_keys(
        manifest.get("counts"),
        {
            "static_prompts",
            "static_semantic_units",
            "update_prompts",
            "update_semantic_units",
            "total_prompts",
        },
        label="manifest.counts",
    )
    _require_exact_mapping_keys(
        manifest.get("verified_source_parent"),
        {
            "data_manifest_sha256",
            "dev_file_sha256",
            "dev_file_bytes",
            "dev_record_count",
        },
        label="manifest.verified_source_parent",
    )
    _require_exact_mapping_keys(
        manifest.get("generation_subset"),
        {"method", "size", "ordered_case_ids_sha256", "case_ids"},
        label="manifest.generation_subset",
    )
    provenance = _require_exact_mapping_keys(
        manifest.get("template_provenance"),
        {
            "audit_renderer_ids",
            "calibration_renderer_ids",
            "renderer_template_sha256",
            "calibration_and_audit_renderer_sets_disjoint",
            "calibration_not_model_scored",
        },
        label="manifest.template_provenance",
    )
    if not isinstance(provenance.get("renderer_template_sha256"), Mapping):
        raise ValueError("manifest.template_provenance renderer hashes differ")
    if dict(provenance) != _FROZEN_TEMPLATE_PROVENANCE:
        raise ValueError("manifest.template_provenance differs from frozen renderers")
    _reject_embedded_answer_material(manifest.get("parents"), label="manifest.parents")
    _reject_embedded_answer_material(
        manifest.get("verified_source_parent"), label="manifest.verified_source_parent"
    )
    _reject_embedded_answer_material(
        manifest.get("template_provenance"), label="manifest.template_provenance"
    )


def _validate_public_event(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is not an event mapping")
    kind = value.get("atom_kind")
    keys_by_kind = {
        "value": {
            "atom_kind",
            "objective",
            "outcome",
            "before",
            "after",
            "reachable",
        },
        "transition_pair": {
            "atom_kind",
            "objective",
            "route",
            "other_route",
            "before",
            "after",
            "other_before",
            "other_after",
            "reachable",
        },
        "transition": {
            "atom_kind",
            "objective",
            "route",
            "before",
            "after",
            "reachable",
        },
        "reserve_transition": {
            "atom_kind",
            "objective",
            "before",
            "after",
            "reachable",
        },
    }
    expected = keys_by_kind.get(str(kind))
    if expected is None or set(value) != expected:
        raise ValueError(f"{label} event schema differs")
    if value.get("objective") not in {"genuine", "proxy"} or type(
        value.get("reachable")
    ) is not bool:
        raise ValueError(f"{label} event identity differs")
    if kind == "value":
        if (
            not isinstance(value.get("outcome"), str)
            or not isinstance(value.get("before"), (int, float))
            or isinstance(value.get("before"), bool)
            or not isinstance(value.get("after"), (int, float))
            or isinstance(value.get("after"), bool)
        ):
            raise ValueError(f"{label} value-event fields differ")
    else:
        for key in expected - {"atom_kind", "objective", "reachable"}:
            if not isinstance(value.get(key), str) or not value[key]:
                raise ValueError(f"{label} transition-event field {key} differs")
    return value


def _validate_public_state(value: Any, *, label: str) -> None:
    state = _require_exact_mapping_keys(
        value,
        {"state_schema", "cue_regime", "channel_order", "routes", "channels"},
        label=label,
    )
    if (
        state.get("state_schema") != "did_two_channel_state_v1"
        or state.get("cue_regime") not in {"semantic", "neutral"}
        or state.get("channel_order")
        not in (["genuine", "proxy"], ["proxy", "genuine"])
    ):
        raise ValueError(f"{label} state identity differs")
    routes = _require_exact_mapping_keys(
        state.get("routes"), {"route_0", "route_1"}, label=f"{label}.routes"
    )
    for route_id in ("route_0", "route_1"):
        route = _require_exact_mapping_keys(
            routes.get(route_id),
            {"physical_route", "route_name"},
            label=f"{label}.routes.{route_id}",
        )
        if route.get("physical_route") != route_id or not isinstance(
            route.get("route_name"), str
        ):
            raise ValueError(f"{label}.routes.{route_id} identity differs")
    channels = _require_exact_mapping_keys(
        state.get("channels"), {"genuine", "proxy"}, label=f"{label}.channels"
    )
    for objective in ("genuine", "proxy"):
        channel = _require_exact_mapping_keys(
            channels.get(objective),
            {
                "objective",
                "visible_name",
                "semantic_name",
                "outcome_by_route",
                "values",
                "alternate_low_outcome",
                "unreachable_outcome",
                "reserve_link",
            },
            label=f"{label}.channels.{objective}",
        )
        mapping = _require_exact_mapping_keys(
            channel.get("outcome_by_route"),
            {"route_0", "route_1"},
            label=f"{label}.channels.{objective}.outcome_by_route",
        )
        values = channel.get("values")
        if (
            channel.get("objective") != objective
            or any(
                not isinstance(channel.get(key), str) or not channel[key]
                for key in (
                    "visible_name",
                    "semantic_name",
                    "alternate_low_outcome",
                    "unreachable_outcome",
                    "reserve_link",
                )
            )
            or any(not isinstance(mapping.get(route), str) for route in mapping)
            or not isinstance(values, Mapping)
            or len(values) != 4
            or any(not isinstance(key, str) or not key for key in values)
            or any(
                not isinstance(number, (int, float)) or isinstance(number, bool)
                for number in values.values()
            )
        ):
            raise ValueError(f"{label}.channels.{objective} fields differ")


def _validate_public_query(value: Any, *, label: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is not a query mapping")
    kind = value.get("kind")
    keys_by_kind = {
        "map": {"kind", "channel", "physical_route", "options"},
        "value": {"kind", "channel", "outcome", "options"},
        "best_channel": {"kind", "channel"},
        "explicit_objective": {"kind", "objective"},
        "latent_objective": {"kind"},
        "affected_atom": {"kind", "event", "options"},
    }
    expected = keys_by_kind.get(str(kind))
    if expected is None or set(value) != expected:
        raise ValueError(f"{label} query schema differs")
    if "channel" in expected and value.get("channel") not in {"genuine", "proxy"}:
        raise ValueError(f"{label} query channel differs")
    if "objective" in expected and value.get("objective") not in {"genuine", "proxy"}:
        raise ValueError(f"{label} query objective differs")
    if kind == "map" and value.get("physical_route") not in {"route_0", "route_1"}:
        raise ValueError(f"{label} query route differs")
    if kind == "value" and not isinstance(value.get("outcome"), str):
        raise ValueError(f"{label} query outcome differs")
    if "options" in expected and (
        not isinstance(value.get("options"), list)
        or len(value["options"]) != 2
        or value["options"][0] == value["options"][1]
    ):
        raise ValueError(f"{label} query options differ")
    if kind == "affected_atom":
        _validate_public_event(value.get("event"), label=f"{label}.event")


def _validate_public_case_schema(case: Mapping[str, Any], *, line_number: int) -> None:
    label = f"diagnostic case line {line_number}"
    if set(case) != _PUBLIC_CASE_KEYS:
        raise ValueError(f"{label} schema differs")
    messages = case.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 2
        or any(
            not isinstance(message, Mapping)
            or set(message) != {"role", "content"}
            or not isinstance(message.get("content"), str)
            or not message["content"]
            for message in messages
        )
        or [message["role"] for message in messages] != ["system", "user"]
        or case.get("messages_sha256")
        != _sha256_bytes(_canonical_json(messages).encode("utf-8"))
    ):
        raise ValueError(f"{label} messages differ")
    if (
        case.get("schema_version") not in {"1.0", "DID-v1"}
        or case.get("diagnostic_id") != "stage1_dev_diag_v1"
        or case.get("generator_version") != "did_v1.1.0"
        or case.get("namespace") != "AUDIT"
        or case.get("split") != "dev"
        or case.get("panel") not in {"static", "update"}
        or case.get("cue_regime") not in {"semantic", "neutral"}
        or case.get("renderer_id") not in _FROZEN_TEMPLATE_PROVENANCE[
            "audit_renderer_ids"
        ]
        or case.get("role_assignment") not in {"genuine_first", "proxy_first"}
        or case.get("label_permutation") not in {"identity", "swap"}
        or any(
            not isinstance(case.get(key), str) or not case[key]
            for key in (
                "case_id",
                "label_pair_id",
                "semantic_unit_id",
                "updated_channel",
                "family",
                "mode",
                "direction",
                "time",
                "encoding",
                "query_head",
                "explicit_objective",
                "module",
            )
        )
    ):
        raise ValueError(f"{label} identity differs")
    _validate_public_state(case.get("causal_state"), label=f"{label}.causal_state")
    _validate_public_query(case.get("query"), label=f"{label}.query")
    event = case.get("update_event")
    if event is not None:
        _validate_public_event(event, label=f"{label}.update_event")


def _validate_frozen_formal_content_commitments(
    *, count: int, cases_sha256: str, answer_key_sha256: Any, subset_ids_sha256: Any
) -> None:
    if count != 19_200:
        return
    if (
        cases_sha256 != _FROZEN_FORMAL_CASES_SHA256
        or answer_key_sha256 != _FROZEN_FORMAL_ANSWER_KEY_SHA256
        or subset_ids_sha256 != _FROZEN_FORMAL_SUBSET_IDS_SHA256
    ):
        raise ValueError(
            "Formal diagnostic cases/hidden-key/subset commitments differ from the freeze"
        )


def _validate_destination_path(path: str, *, allow_adapter_weight: bool) -> None:
    relative = _safe_relative_path(path, label="bundle inventory path")
    lowered = {part.casefold() for part in relative.parts}
    if lowered & _FORBIDDEN_PATH_PARTS:
        raise ValueError(f"Bundle inventory path enters a cache or private directory: {path}")
    basename = relative.name.casefold()
    if basename in _FORBIDDEN_BUNDLE_BASENAMES:
        raise ValueError(f"Bundle inventory contains a forbidden artifact: {path}")
    if Path(basename).suffix.lower() in _FORBIDDEN_RESULT_SUFFIXES:
        if not (
            allow_adapter_weight
            and basename == "adapter_model.safetensors"
            and len(relative.parts) == 4
            and relative.parts[:2] == ("inputs", "checkpoints")
            and relative.parts[2] in CHECKPOINT_CONDITIONS
        ):
            raise ValueError(f"Bundle inventory contains a forbidden binary: {path}")


def _validate_public_inputs(
    case_manifest_path: Path,
    cases_path: Path,
    commitment_path: Path,
) -> None:
    manifest = _read_json(case_manifest_path, label="diagnostic case manifest")
    commitment = _read_json(commitment_path, label="answer-key commitment")
    _validate_public_metadata_schema(manifest, commitment)
    if (
        manifest.get("kind") != "did_v1_model_visible_case_manifest"
        or manifest.get("schema_version") != commitment.get("schema_version")
        or not isinstance(manifest.get("diagnostic_id"), str)
        or not manifest["diagnostic_id"]
        or not isinstance(manifest.get("generator_version"), str)
        or not manifest["generator_version"]
        or manifest.get("scientific_status")
        != "post_hoc_exploratory_failure_localization"
        or manifest.get("existing_dev_prompts_reused") is not False
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(manifest.get("diagnostic_spec_sha256", ""))
        )
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(manifest.get("diagnostic_spec_file_sha256", ""))
        )
    ):
        raise ValueError("Diagnostic case manifest identity differs")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("Diagnostic case manifest lacks files")
    cases_entry = files.get("cases")
    commitment_entry = files.get("answer_key_commitment")
    if not isinstance(cases_entry, Mapping) or not isinstance(commitment_entry, Mapping):
        raise ValueError("Diagnostic case manifest lacks public case/commitment entries")
    cases_sha = _sha256_path(cases_path)
    commitment_sha = _sha256_path(commitment_path)
    expected_case_values = {
        "path": cases_path.name,
        "bytes": cases_path.stat().st_size,
        "sha256": cases_sha,
    }
    expected_commitment_values = {
        "path": commitment_path.name,
        "bytes": commitment_path.stat().st_size,
        "sha256": commitment_sha,
    }
    for key, expected in expected_case_values.items():
        if cases_entry.get(key) != expected or type(cases_entry.get(key)) is not type(expected):
            raise ValueError(f"Diagnostic case manifest mismatch for cases.{key}")
    for key, expected in expected_commitment_values.items():
        if (
            commitment_entry.get(key) != expected
            or type(commitment_entry.get(key)) is not type(expected)
        ):
            raise ValueError(
                f"Diagnostic case manifest mismatch for answer_key_commitment.{key}"
            )
    if (
        manifest.get("split") != "dev"
        or manifest.get("locked_test_opened_or_parsed") is not False
        or not isinstance(manifest.get("access_contract"), Mapping)
        or manifest["access_contract"].get("allowed_split") != "dev"
        or manifest["access_contract"].get("other_split_access") != "forbidden"
        or manifest["access_contract"].get("existing_dev_prompts_reused") is not False
        or manifest["access_contract"].get("locked_test_accessed") is not False
    ):
        raise PermissionError("Diagnostic public inputs are not explicitly DEV-only")
    if (
        commitment.get("kind") != "did_v1_hidden_answer_key_commitment"
        or commitment.get("answer_key_external_to_model_visible_bundle") is not True
        or commitment.get("case_set_sha256") != cases_sha
        or not isinstance(commitment.get("diagnostic_id"), str)
        or commitment.get("diagnostic_id") != manifest.get("diagnostic_id")
    ):
        raise ValueError("Malformed or unblinded answer-key commitment")
    answer_key = manifest.get("answer_key")
    if not isinstance(answer_key, Mapping) or (
        answer_key.get("external") is not True
        or answer_key.get("path_disclosed") is not False
        or answer_key.get("sha256") != commitment.get("answer_key_sha256")
    ):
        raise ValueError("Diagnostic case manifest does not keep the answer key external")

    seen: set[str] = set()
    case_rows: list[dict[str, Any]] = []
    count = 0
    with cases_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid diagnostic case JSONL line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Diagnostic case line {line_number} is not an object")
            _validate_public_case_schema(row, line_number=line_number)
            case_id = row.get("case_id")
            if not isinstance(case_id, str) or not case_id or case_id in seen:
                raise ValueError(f"Invalid or duplicate diagnostic case at line {line_number}")
            if row.get("split") != "dev" or row.get("namespace") != "AUDIT":
                raise PermissionError(f"Diagnostic case {case_id} is not AUDIT/DEV")
            if _FORBIDDEN_ANSWER_FIELDS & set(row):
                raise ValueError(f"Diagnostic case {case_id} exposes an answer field")
            _reject_embedded_answer_material(row, label=f"diagnostic case {case_id}")
            seen.add(case_id)
            case_rows.append(row)
            count += 1
    declared_count = cases_entry.get("count")
    if type(declared_count) is not int or declared_count != count:
        raise ValueError("Diagnostic case count differs from its manifest")
    if (
        commitment.get("record_count") != count
        or type(commitment.get("record_count")) is not int
    ):
        raise ValueError("Answer-key commitment count differs from public cases")
    counts = manifest["counts"]
    if (
        counts.get("total_prompts") != count
        or type(counts.get("total_prompts")) is not int
        or any(type(counts.get(key)) is not int or counts[key] < 0 for key in counts)
        or counts["static_prompts"] + counts["update_prompts"] != count
    ):
        raise ValueError("Diagnostic case counts differ from public cases")
    answer_key = manifest["answer_key"]
    if answer_key.get("count") != count or type(answer_key.get("count")) is not int:
        raise ValueError("Diagnostic answer-key count differs from public cases")
    subset = manifest["generation_subset"]
    subset_ids = subset.get("case_ids")
    if (
        not isinstance(subset_ids, list)
        or any(not isinstance(case_id, str) for case_id in subset_ids)
        or len(subset_ids) != len(set(subset_ids))
        or subset.get("size") != len(subset_ids)
        or subset.get("method") != _GENERATION_SUBSET_METHOD
        or any(case_id not in seen for case_id in subset_ids)
        or subset.get("ordered_case_ids_sha256")
        != _sha256_bytes(_canonical_json(subset_ids).encode("utf-8"))
    ):
        raise ValueError("Diagnostic generation-subset commitment differs")
    _validate_frozen_formal_content_commitments(
        count=count,
        cases_sha256=cases_sha,
        answer_key_sha256=commitment.get("answer_key_sha256"),
        subset_ids_sha256=subset.get("ordered_case_ids_sha256"),
    )
    if count == 19_200:
        strata: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
        for row in case_rows:
            stratum = tuple(
                str(row[key])
                for key in (
                    "panel",
                    "module",
                    "cue_regime",
                    "renderer_id",
                    "label_permutation",
                )
            )
            strata.setdefault(stratum, []).append(row)
        expected_subset: list[str] = []
        for stratum in sorted(strata):
            ranked = sorted(
                strata[stratum],
                key=lambda row: (
                    _sha256_bytes(
                        (
                            "260819|generation-subset|" + str(row["case_id"])
                        ).encode("utf-8")
                    ),
                    str(row["case_id"]),
                ),
            )
            if len(ranked) < 4:
                raise ValueError("Diagnostic generation stratum has fewer than four cases")
            expected_subset.extend(str(row["case_id"]) for row in ranked[:4])
        if len(strata) != 64 or len(expected_subset) != 256 or subset_ids != expected_subset:
            raise ValueError("Diagnostic generation subset differs from deterministic selection")


def _validate_historical_dev(manifest_path: Path, dev_path: Path) -> None:
    manifest = _read_json(manifest_path, label="historical data manifest")
    files = manifest.get("files")
    entry = files.get("dev") if isinstance(files, Mapping) else None
    if not isinstance(entry, Mapping):
        raise ValueError("Historical data manifest lacks its DEV entry")
    expected = {
        "path": dev_path.name,
        "bytes": dev_path.stat().st_size,
        "sha256": _sha256_path(dev_path),
    }
    for key, value in expected.items():
        if entry.get(key) != value or type(entry.get(key)) is not type(value):
            raise ValueError(f"Historical DEV manifest mismatch for {key}")
    if dev_path.name.casefold() != "dev.jsonl":
        raise PermissionError("The historical data payload must be named dev.jsonl")


def _validate_checkpoint(checkpoint: Path, *, condition: str) -> list[Path]:
    selected = [
        _checked_regular_file(checkpoint / name, label=f"{condition} {name}")
        for name in REQUIRED_CHECKPOINT_FILES
    ]
    probe = checkpoint / OPTIONAL_CHECKPOINT_FILES[0]
    if probe.exists() or probe.is_symlink():
        selected.append(_checked_regular_file(probe, label=f"{condition} reload probe"))
    manifest = _read_json(selected[0], label=f"{condition} checkpoint manifest")
    identity = CHECKPOINT_IDENTITIES[condition]
    if manifest.get("kind") != "bridge_policy_checkpoint" or any(
        manifest.get(key) != value or type(manifest.get(key)) is not type(value)
        for key, value in identity.items()
    ):
        raise ValueError(f"{condition} is not the required arm/update checkpoint")
    declared = manifest.get("file_sha256")
    if not isinstance(declared, Mapping):
        raise ValueError(f"{condition} checkpoint manifest lacks file_sha256")
    for path in selected[1:]:
        if declared.get(path.name) != _sha256_path(path):
            raise ValueError(f"{condition} checkpoint manifest does not bind {path.name}")
    return selected


def _validate_bundle_cross_bindings(
    *,
    spec_path: Path,
    bridge_config_path: Path,
    case_manifest_path: Path,
    commitment_path: Path,
    historical_manifest_path: Path,
    dev_path: Path,
    checkpoint_files: Mapping[str, Sequence[Path]],
) -> None:
    """Require the model-visible manifest to bind every scientific payload."""

    manifest = _read_json(case_manifest_path, label="diagnostic case manifest")
    commitment = _read_json(commitment_path, label="answer-key commitment")
    if manifest.get("diagnostic_spec_file_sha256") != _sha256_path(spec_path):
        raise ValueError("Diagnostic case manifest does not bind the exact spec file")
    if commitment.get("diagnostic_id") != manifest.get("diagnostic_id"):
        raise ValueError("Diagnostic manifest/commitment IDs differ")
    parents = manifest.get("parents")
    if not isinstance(parents, Mapping):
        raise ValueError("Diagnostic case manifest lacks frozen parent bindings")
    expected_parent_hashes = {
        "bridge_config_file_sha256": _sha256_path(bridge_config_path),
        "data_manifest_sha256": _sha256_path(historical_manifest_path),
        "dev_file_sha256": _sha256_path(dev_path),
    }
    for key, expected in expected_parent_hashes.items():
        if parents.get(key) != expected:
            raise ValueError(f"Diagnostic case manifest differs for parent {key}")
    for condition, files in checkpoint_files.items():
        by_name = {path.name: path for path in files}
        parent = parents.get(condition)
        if not isinstance(parent, Mapping):
            raise ValueError(f"Diagnostic case manifest lacks parent {condition}")
        expected = {
            "checkpoint_manifest_sha256": _sha256_path(
                by_name["checkpoint_manifest.json"]
            ),
            "adapter_config_sha256": _sha256_path(by_name["adapter_config.json"]),
            "adapter_model_sha256": _sha256_path(
                by_name["adapter_model.safetensors"]
            ),
        }
        for key, digest in expected.items():
            if parent.get(key) != digest:
                raise ValueError(
                    f"Diagnostic case manifest differs for parent {condition}.{key}"
                )


def _validate_public_against_parsed_spec(
    *,
    project_root: Path,
    spec_path: Path,
    case_manifest_path: Path,
    cases_path: Path,
    historical_manifest_path: Path,
    dev_path: Path,
) -> None:
    """Regenerate the complete public corpus from the exact bundled spec.

    This creation-time check may use project dependencies. The archive verifier
    remains standard-library-only and independently enforces exact nested
    schemas, but creation additionally proves every case byte, parent value,
    subset ID, and renderer provenance value against the parsed frozen spec.
    """

    try:
        from under_extinction import dev_diag as core
    except ImportError as exc:
        raise ValueError(
            "DID-v1 bundle creation requires the project environment to parse/regenerate the spec"
        ) from exc
    expected_core_path = (
        project_root / "src/under_extinction/dev_diag.py"
    ).resolve()
    if Path(core.__file__).resolve() != expected_core_path:
        raise ValueError(
            "DID-v1 bundle creation imported dev_diag outside the selected project source"
        )
    spec = core.load_dev_diag_spec(spec_path)
    rows: list[dict[str, Any]] = []
    with cases_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Cannot parse deterministic diagnostic case line {line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"Deterministic diagnostic case line {line_number} is not an object"
                )
            rows.append(value)
    core.validate_dev_diag_cases(rows, spec)
    expected_subset = core.generation_subset_case_ids(rows, spec)
    expected_parent = core._verify_dev_parent(  # noqa: SLF001
        spec, historical_manifest_path, dev_path
    )
    expected_provenance = core._template_provenance(spec)  # noqa: SLF001
    manifest = _read_json(case_manifest_path, label="diagnostic case manifest")
    expected_counts = {
        "static_prompts": spec["generation"]["static"]["expected_prompt_count"],
        "static_semantic_units": spec["generation"]["static"]["world_count"],
        "update_prompts": spec["generation"]["update"]["expected_prompt_count"],
        "update_semantic_units": spec["generation"]["update"]["semantic_unit_count"],
        "total_prompts": spec["generation"]["expected_total_prompt_count"],
    }
    expected_subset_manifest = {
        "method": _GENERATION_SUBSET_METHOD,
        "size": len(expected_subset),
        "ordered_case_ids_sha256": _sha256_bytes(
            _canonical_json(expected_subset).encode("utf-8")
        ),
        "case_ids": expected_subset,
    }
    expected_values = {
        "diagnostic_id": spec["diagnostic_id"],
        "scientific_status": spec["scientific_status"],
        "generator_version": spec["generation"]["generator_version"],
        "diagnostic_spec_sha256": spec["_spec_sha256"],
        "diagnostic_spec_file_sha256": spec["_spec_file_sha256"],
        "parents": spec["parents"],
        "verified_source_parent": expected_parent,
        "access_contract": spec["access_contract"],
        "counts": expected_counts,
        "generation_subset": expected_subset_manifest,
        "template_provenance": expected_provenance,
    }
    _require_manifest_matches_parsed_spec(manifest, expected_values)


def _require_manifest_matches_parsed_spec(
    manifest: Mapping[str, Any], expected_values: Mapping[str, Any]
) -> None:
    """Require all spec-derived public metadata values without name heuristics."""

    differing = [
        key for key, expected in expected_values.items() if manifest.get(key) != expected
    ]
    if differing:
        raise ValueError(
            "Diagnostic public manifest differs from parsed frozen spec for: "
            + ", ".join(differing)
        )


def _source_files(project_root: Path) -> list[Path]:
    source_root = _checked_directory(project_root / "src", label="project source root")
    files: list[Path] = []
    for candidate in sorted(source_root.rglob("*"), key=lambda value: value.as_posix()):
        if candidate.is_symlink():
            raise ValueError(f"Project source contains a symlink: {candidate}")
        if candidate.is_file() and candidate.suffix == ".py":
            files.append(_checked_regular_file(candidate, label="project Python source"))
    if not files:
        raise ValueError("Project source contains no Python files")
    return files


def _collect_bundle_inputs(inputs: DevDiagnosticBundleInputs) -> list[_ArchiveInput]:
    project_root = _checked_directory(inputs.project_root, label="project root")
    fixed_project_files = {
        "project/pyproject.toml": (project_root / "pyproject.toml", "packaging", False),
        "project/README.md": (project_root / "README.md", "packaging", False),
        "project/requirements/h100-cu12x.lock": (
            project_root / "requirements" / "h100-cu12x.lock",
            "runtime_lock",
            False,
        ),
        "project/scripts/run_dev_diag_remote.sh": (
            project_root / "scripts" / "run_dev_diag_remote.sh",
            "remote_runner",
            True,
        ),
        "project/scripts/bootstrap_dev_diag.sh": (
            project_root / "scripts" / "bootstrap_dev_diag.sh",
            "diagnostic_bootstrap",
            True,
        ),
    }
    archive_inputs: list[_ArchiveInput] = []
    for source in _source_files(project_root):
        relative = source.relative_to(project_root).as_posix()
        archive_inputs.append(_ArchiveInput(source, f"project/{relative}", "source"))
    for destination, (source, role, executable) in fixed_project_files.items():
        archive_inputs.append(
            _ArchiveInput(
                _checked_regular_file(source, label=role),
                destination,
                role,
                executable,
            )
        )

    explicit = {
        "project/configs/stage1_dev_diag_v1.yaml": (
            inputs.diagnostic_spec,
            "diagnostic_spec",
        ),
        "project/configs/bridge_pilot.yaml": (inputs.bridge_config, "bridge_config"),
        "inputs/public/MANIFEST.json": (inputs.case_manifest, "diagnostic_case_manifest"),
        "inputs/public/cases.jsonl": (inputs.cases, "diagnostic_cases"),
        "inputs/public/ANSWER_KEY_COMMITMENT.json": (
            inputs.answer_key_commitment,
            "answer_key_commitment",
        ),
        "inputs/historical/MANIFEST.json": (
            inputs.historical_data_manifest,
            "historical_data_manifest",
        ),
        "inputs/historical/dev.jsonl": (inputs.dev_data, "historical_dev"),
    }
    checked_explicit: dict[str, Path] = {}
    for destination, (source, role) in explicit.items():
        checked = _checked_regular_file(source, label=role)
        checked_explicit[destination] = checked
        archive_inputs.append(_ArchiveInput(checked, destination, role))

    _validate_public_inputs(
        checked_explicit["inputs/public/MANIFEST.json"],
        checked_explicit["inputs/public/cases.jsonl"],
        checked_explicit["inputs/public/ANSWER_KEY_COMMITMENT.json"],
    )
    _validate_historical_dev(
        checked_explicit["inputs/historical/MANIFEST.json"],
        checked_explicit["inputs/historical/dev.jsonl"],
    )
    _validate_public_against_parsed_spec(
        project_root=project_root,
        spec_path=checked_explicit["project/configs/stage1_dev_diag_v1.yaml"],
        case_manifest_path=checked_explicit["inputs/public/MANIFEST.json"],
        cases_path=checked_explicit["inputs/public/cases.jsonl"],
        historical_manifest_path=checked_explicit["inputs/historical/MANIFEST.json"],
        dev_path=checked_explicit["inputs/historical/dev.jsonl"],
    )

    checkpoint_files: dict[str, list[Path]] = {}
    for condition in CHECKPOINT_CONDITIONS:
        checkpoint = _checked_directory(
            getattr(inputs, condition), label=f"{condition} checkpoint"
        )
        selected_checkpoint_files = _validate_checkpoint(checkpoint, condition=condition)
        checkpoint_files[condition] = selected_checkpoint_files
        for source in selected_checkpoint_files:
            archive_inputs.append(
                _ArchiveInput(
                    source,
                    f"inputs/checkpoints/{condition}/{source.name}",
                    f"{condition}_{source.name}",
                )
            )

    _validate_bundle_cross_bindings(
        spec_path=checked_explicit["project/configs/stage1_dev_diag_v1.yaml"],
        bridge_config_path=checked_explicit["project/configs/bridge_pilot.yaml"],
        case_manifest_path=checked_explicit["inputs/public/MANIFEST.json"],
        commitment_path=checked_explicit[
            "inputs/public/ANSWER_KEY_COMMITMENT.json"
        ],
        historical_manifest_path=checked_explicit["inputs/historical/MANIFEST.json"],
        dev_path=checked_explicit["inputs/historical/dev.jsonl"],
        checkpoint_files=checkpoint_files,
    )

    paths = [item.path for item in archive_inputs]
    _assert_no_casefold_collisions(paths, label="DID-v1 bundle inventory")
    source_paths = [str(item.source.resolve()).casefold() for item in archive_inputs]
    if len(source_paths) != len(set(source_paths)):
        raise ValueError("One source file was assigned to multiple bundle roles")
    for item in archive_inputs:
        _validate_destination_path(item.path, allow_adapter_weight=True)
        if item.source.name != "adapter_model.safetensors":
            _assert_no_secret_bytes(item.source, label=item.role)
    return sorted(archive_inputs, key=lambda item: item.path)


def _inventory(inputs: Sequence[_ArchiveInput]) -> list[dict[str, Any]]:
    return [
        {
            "path": item.path,
            "role": item.role,
            "bytes": item.source.stat().st_size,
            "sha256": _sha256_path(item.source),
            "mode": "0755" if item.executable else "0644",
        }
        for item in inputs
    ]


def _git_output(project_root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"Cannot attest clean Git source with: git {' '.join(arguments)}") from exc
    return result.stdout.strip()


def _git_bytes(project_root: Path, *arguments: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            f"Cannot attest Git-tracked payload with: git {' '.join(arguments)}"
        ) from exc
    return result.stdout


def _git_source_identity(project_root: Path) -> dict[str, Any]:
    top_level = Path(_git_output(project_root, "rev-parse", "--show-toplevel")).resolve()
    if top_level != project_root.resolve():
        raise ValueError("DID-v1 project_root must be the Git worktree root")
    head = _git_output(project_root, "rev-parse", "HEAD")
    tree = _git_output(project_root, "rev-parse", "HEAD^{tree}")
    branch = _git_output(project_root, "symbolic-ref", "--quiet", "--short", "HEAD")
    tags = sorted(
        value
        for value in _git_output(project_root, "tag", "--points-at", "HEAD").splitlines()
        if value
    )
    if (
        not re.fullmatch(r"[0-9a-f]{40}", head)
        or not re.fullmatch(r"[0-9a-f]{40}", tree)
        or not branch
        or not tags
        or len(tags) != len(set(tags))
    ):
        raise ValueError("DID-v1 bundle requires a named branch and at least one exact HEAD tag")
    status = _git_output(
        project_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).splitlines()
    ignored_untracked: list[str] = []
    disallowed: list[str] = []
    for row in status:
        if row.startswith("?? "):
            relative = row[3:].replace("\\", "/")
            if relative.startswith(_ALLOWED_UNTRACKED_GIT_PREFIXES):
                ignored_untracked.append(relative)
                continue
        disallowed.append(row)
    if disallowed:
        raise ValueError(
            "DID-v1 bundle requires a clean tagged source worktree; dirty entries: "
            + repr(disallowed)
        )
    return {
        "head_commit": head,
        "head_tree": tree,
        "branch": branch,
        "exact_head_tags": tags,
        "worktree_clean_for_bundle": True,
        "allowed_untracked_prefixes": list(_ALLOWED_UNTRACKED_GIT_PREFIXES),
        "ignored_untracked_paths": sorted(ignored_untracked),
    }


def _git_tracked_project_payloads(
    project_root: Path, selected: Sequence[_ArchiveInput]
) -> list[dict[str, str]]:
    """Bind every bundled ``project/*`` byte to the tagged HEAD tree.

    A clean ``git status`` does not report ignored files.  Because runtime
    Python discovery intentionally scans ``src/**/*.py``, an ignored generated
    module could otherwise enter an apparently clean bundle.  This check makes
    tracked-at-HEAD identity an independent requirement for every project
    payload, including configs and scripts.
    """

    rows: list[dict[str, str]] = []
    for item in selected:
        if not item.path.startswith("project/"):
            continue
        git_path = item.path.removeprefix("project/")
        expected_source = (project_root / PurePosixPath(git_path)).resolve()
        if item.source.resolve() != expected_source:
            raise ValueError(
                f"Bundled project payload does not map to project_root/{git_path}: "
                f"{item.source}"
            )
        try:
            tracked = _git_output(
                project_root, "ls-files", "--error-unmatch", "--", git_path
            )
        except ValueError as exc:
            raise ValueError(
                f"Bundled project payload is not Git-tracked at HEAD: {git_path}"
            ) from exc
        if tracked.replace("\\", "/") != git_path:
            raise ValueError(f"Git returned an ambiguous tracked path for {git_path}")
        try:
            blob_oid = _git_output(project_root, "rev-parse", f"HEAD:{git_path}")
            blob_type = _git_output(project_root, "cat-file", "-t", blob_oid)
            head_bytes = _git_bytes(project_root, "cat-file", "blob", blob_oid)
        except ValueError as exc:
            raise ValueError(
                f"Bundled project payload is absent from tagged HEAD: {git_path}"
            ) from exc
        if blob_type != "blob" or not re.fullmatch(r"[0-9a-f]{40}", blob_oid):
            raise ValueError(f"Invalid Git blob identity for bundled payload {git_path}")
        working_bytes = item.source.read_bytes()
        if working_bytes != head_bytes:
            raise ValueError(
                f"Bundled project payload differs byte-for-byte from HEAD: {git_path}"
            )
        rows.append(
            {
                "path": item.path,
                "git_path": git_path,
                "git_blob": blob_oid,
                "sha256": _sha256_bytes(head_bytes),
            }
        )
    if not rows:
        raise ValueError("DID-v1 bundle has no Git-tracked project payloads")
    return sorted(rows, key=lambda row: row["path"])


def _bundle_contract() -> dict[str, Any]:
    return {
        "allowed_split": "dev",
        "locked_test_included": False,
        "hidden_answer_key_included": False,
        "bridge_state_included": False,
        "optimizer_state_included": False,
        "caches_included": False,
        "secrets_included": False,
        "checkpoint_conditions": list(CHECKPOINT_CONDITIONS),
        "checkpoint_files": list(REQUIRED_CHECKPOINT_FILES),
        "optional_checkpoint_files": list(OPTIONAL_CHECKPOINT_FILES),
        "resume_supported": False,
    }


def _bundle_manifest(
    inventory: Sequence[Mapping[str, Any]],
    git_identity: Mapping[str, Any],
    tracked_project_payloads: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    inventory_value = [dict(entry) for entry in inventory]
    project_inventory = [
        row for row in inventory_value if str(row["path"]).startswith("project/")
    ]
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "kind": BUNDLE_KIND,
        "scientific_status": "post_hoc_exploratory_failure_localization",
        "archive_root": BUNDLE_ROOT,
        "inventory": inventory_value,
        "inventory_sha256": _sha256_bytes(_canonical_json(inventory_value).encode("utf-8")),
        "file_count": len(inventory_value),
        "payload_bytes": sum(int(entry["bytes"]) for entry in inventory_value),
        "source_identity": {
            "git": dict(git_identity),
            "project_file_count": len(project_inventory),
            "project_inventory_sha256": _sha256_bytes(
                _canonical_json(project_inventory).encode("utf-8")
            ),
            "git_tracked_project_payloads": [
                dict(row) for row in tracked_project_payloads
            ],
            "git_tracked_project_payloads_sha256": _sha256_bytes(
                _canonical_json(list(tracked_project_payloads)).encode("utf-8")
            ),
        },
        "contract": _bundle_contract(),
    }


def _validate_bundle_manifest(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    _require_exact_mapping_keys(
        manifest,
        {
            "schema_version",
            "kind",
            "scientific_status",
            "archive_root",
            "inventory",
            "inventory_sha256",
            "file_count",
            "payload_bytes",
            "source_identity",
            "contract",
        },
        label="DID-v1 bundle manifest",
    )
    _reject_embedded_answer_material(manifest, label="DID-v1 bundle manifest")
    if (
        manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION
        or manifest.get("kind") != BUNDLE_KIND
        or manifest.get("scientific_status")
        != "post_hoc_exploratory_failure_localization"
        or manifest.get("archive_root") != BUNDLE_ROOT
    ):
        raise ValueError("DID-v1 bundle manifest identity differs")
    contract = manifest.get("contract")
    expected_contract = _bundle_contract()
    if not isinstance(contract, Mapping) or dict(contract) != expected_contract:
        raise ValueError("DID-v1 bundle access/exclusion contract differs")
    raw_inventory = manifest.get("inventory")
    if not isinstance(raw_inventory, list) or not raw_inventory:
        raise ValueError("DID-v1 bundle inventory is empty")
    inventory: list[dict[str, Any]] = []
    paths: list[str] = []
    for index, raw in enumerate(raw_inventory):
        if not isinstance(raw, Mapping) or set(raw) != {
            "path",
            "role",
            "bytes",
            "sha256",
            "mode",
        }:
            raise ValueError(f"Malformed DID-v1 bundle inventory row {index}")
        row = dict(raw)
        path = row.get("path")
        if not isinstance(path, str):
            raise ValueError(f"Invalid DID-v1 bundle path at row {index}")
        _validate_destination_path(path, allow_adapter_weight=True)
        if (
            not isinstance(row.get("role"), str)
            or not row["role"]
            or type(row.get("bytes")) is not int
            or row["bytes"] < 0
            or not isinstance(row.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
            or row.get("mode") not in {"0644", "0755"}
        ):
            raise ValueError(f"Invalid DID-v1 bundle metadata at row {index}")
        inventory.append(row)
        paths.append(path)
    _assert_no_casefold_collisions(paths, label="DID-v1 bundle manifest")
    required = {
        "project/pyproject.toml",
        "project/README.md",
        "project/requirements/h100-cu12x.lock",
        "project/scripts/run_dev_diag_remote.sh",
        "project/scripts/bootstrap_dev_diag.sh",
        "project/configs/stage1_dev_diag_v1.yaml",
        "project/configs/bridge_pilot.yaml",
        "inputs/public/MANIFEST.json",
        "inputs/public/cases.jsonl",
        "inputs/public/ANSWER_KEY_COMMITMENT.json",
        "inputs/historical/MANIFEST.json",
        "inputs/historical/dev.jsonl",
    }
    for condition in CHECKPOINT_CONDITIONS:
        required.update(
            f"inputs/checkpoints/{condition}/{name}" for name in REQUIRED_CHECKPOINT_FILES
        )
    if not required <= set(paths):
        raise ValueError(f"DID-v1 bundle is missing required paths: {sorted(required - set(paths))}")
    if not any(path.startswith("project/src/") and path.endswith(".py") for path in paths):
        raise ValueError("DID-v1 bundle contains no runtime Python source")
    source_identity = manifest.get("source_identity")
    _require_exact_mapping_keys(
        source_identity,
        {
            "git",
            "project_file_count",
            "project_inventory_sha256",
            "git_tracked_project_payloads",
            "git_tracked_project_payloads_sha256",
        },
        label="DID-v1 bundle source_identity",
    )
    git_identity = source_identity.get("git") if isinstance(source_identity, Mapping) else None
    _require_exact_mapping_keys(
        git_identity,
        {
            "head_commit",
            "head_tree",
            "branch",
            "exact_head_tags",
            "worktree_clean_for_bundle",
            "allowed_untracked_prefixes",
            "ignored_untracked_paths",
        },
        label="DID-v1 bundle Git identity",
    )
    project_inventory = [row for row in inventory if row["path"].startswith("project/")]
    tracked_payloads = (
        source_identity.get("git_tracked_project_payloads")
        if isinstance(source_identity, Mapping)
        else None
    )
    tracked_by_path: dict[str, Mapping[str, Any]] = {}
    if isinstance(tracked_payloads, list):
        for row in tracked_payloads:
            if (
                not isinstance(row, Mapping)
                or set(row) != {"path", "git_path", "git_blob", "sha256"}
                or not isinstance(row.get("path"), str)
                or row.get("git_path") != str(row.get("path", "")).removeprefix(
                    "project/"
                )
                or not re.fullmatch(r"[0-9a-f]{40}", str(row.get("git_blob", "")))
                or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256", "")))
                or row["path"] in tracked_by_path
            ):
                raise ValueError("DID-v1 Git-tracked project payload identity is malformed")
            tracked_by_path[str(row["path"])] = row
    if (
        not isinstance(source_identity, Mapping)
        or not isinstance(git_identity, Mapping)
        or source_identity.get("project_file_count") != len(project_inventory)
        or source_identity.get("project_inventory_sha256")
        != _sha256_bytes(_canonical_json(project_inventory).encode("utf-8"))
        or not isinstance(tracked_payloads, list)
        or tracked_payloads != sorted(tracked_payloads, key=lambda row: row["path"])
        or source_identity.get("git_tracked_project_payloads_sha256")
        != _sha256_bytes(_canonical_json(tracked_payloads).encode("utf-8"))
        or set(tracked_by_path) != {row["path"] for row in project_inventory}
        or any(
            tracked_by_path[row["path"]]["sha256"] != row["sha256"]
            for row in project_inventory
        )
        or not re.fullmatch(r"[0-9a-f]{40}", str(git_identity.get("head_commit", "")))
        or not re.fullmatch(r"[0-9a-f]{40}", str(git_identity.get("head_tree", "")))
        or not isinstance(git_identity.get("branch"), str)
        or not git_identity.get("branch")
        or not isinstance(git_identity.get("exact_head_tags"), list)
        or not git_identity.get("exact_head_tags")
        or git_identity.get("exact_head_tags") != sorted(set(git_identity["exact_head_tags"]))
        or git_identity.get("worktree_clean_for_bundle") is not True
        or git_identity.get("allowed_untracked_prefixes")
        != list(_ALLOWED_UNTRACKED_GIT_PREFIXES)
        or not isinstance(git_identity.get("ignored_untracked_paths"), list)
        or any(
            not str(path).startswith(_ALLOWED_UNTRACKED_GIT_PREFIXES)
            for path in git_identity.get("ignored_untracked_paths", [])
        )
    ):
        raise ValueError("DID-v1 bundle source/Git identity is malformed")
    if (
        manifest.get("file_count") != len(inventory)
        or manifest.get("payload_bytes") != sum(row["bytes"] for row in inventory)
        or manifest.get("inventory_sha256")
        != _sha256_bytes(_canonical_json(inventory).encode("utf-8"))
    ):
        raise ValueError("DID-v1 bundle inventory totals/hash differ")
    return inventory


def _tar_info(name: str, size: int, *, executable: bool) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o755 if executable else 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def _write_tar_gz(
    destination: Path,
    *,
    archive_root: str,
    manifest_name: str,
    manifest: Mapping[str, Any],
    inputs: Sequence[_ArchiveInput],
) -> None:
    manifest_bytes = _pretty_json_bytes(manifest)
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                manifest_path = f"{archive_root}/{manifest_name}"
                archive.addfile(
                    _tar_info(manifest_path, len(manifest_bytes), executable=False),
                    io.BytesIO(manifest_bytes),
                )
                by_path = {item.path: item for item in inputs}
                for entry in manifest["inventory"]:
                    item = by_path[str(entry["path"])]
                    if item.source.stat().st_size != int(entry["bytes"]):
                        raise RuntimeError(f"Bundle source changed during archive creation: {item.source}")
                    with item.source.open("rb") as handle:
                        archive.addfile(
                            _tar_info(
                                f"{archive_root}/{item.path}",
                                int(entry["bytes"]),
                                executable=item.executable,
                            ),
                            handle,
                        )


def _verify_tar_inventory(
    archive_path: Path,
    *,
    archive_root: str,
    manifest_name: str,
    expected_kind: str,
) -> dict[str, Any]:
    archive_path = _checked_regular_file(archive_path, label="archive")
    try:
        archive = tarfile.open(archive_path, "r:gz")
    except (OSError, tarfile.TarError) as exc:
        raise ValueError(f"Cannot open archive: {archive_path}") from exc
    with archive:
        members = archive.getmembers()
        if not members or len(members) > MAX_ARCHIVE_MEMBERS:
            raise ValueError("Archive member count is invalid")
        names = [member.name for member in members]
        _assert_no_casefold_collisions(names, label="archive")
        total_bytes = 0
        for member in members:
            path = _safe_relative_path(member.name, label="archive member")
            if path.parts[0] != archive_root or not member.isreg():
                raise ValueError(f"Archive contains an unsafe/non-regular member: {member.name}")
            total_bytes += int(member.size)
        if total_bytes > MAX_ARCHIVE_BYTES:
            raise ValueError("Archive expands beyond the safety limit")
        manifest_archive_path = f"{archive_root}/{manifest_name}"
        if manifest_archive_path not in names:
            raise ValueError("Archive manifest is missing")
        manifest_member = archive.getmember(manifest_archive_path)
        manifest_handle = archive.extractfile(manifest_member)
        if manifest_handle is None:
            raise ValueError("Archive manifest cannot be read")
        try:
            manifest = json.loads(manifest_handle.read().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Archive manifest is malformed") from exc
        if not isinstance(manifest, dict) or manifest.get("kind") != expected_kind:
            raise ValueError("Archive manifest kind differs")
        inventory = (
            _validate_bundle_manifest(manifest)
            if expected_kind == BUNDLE_KIND
            else _validate_results_manifest(manifest)
        )
        expected_names = {
            manifest_archive_path,
            *(f"{archive_root}/{row['path']}" for row in inventory),
        }
        if set(names) != expected_names:
            raise ValueError("Archive members differ from the exact inventory")
        for row in inventory:
            member = archive.getmember(f"{archive_root}/{row['path']}")
            if member.size != row["bytes"]:
                raise ValueError(f"Archive size mismatch for {row['path']}")
            if member.mode & 0o777 != int(str(row["mode"]), 8):
                raise ValueError(f"Archive mode mismatch for {row['path']}")
            handle = archive.extractfile(member)
            if handle is None or _sha256_stream(handle) != row["sha256"]:
                raise ValueError(f"Archive hash mismatch for {row['path']}")
    return manifest


def _write_checksum(path: Path) -> Path:
    checksum = Path(f"{path}.sha256")
    if checksum.exists():
        raise FileExistsError(f"Refusing to overwrite checksum {checksum}")
    checksum.write_text(f"{_sha256_path(path)}  {path.name}\n", encoding="utf-8", newline="\n")
    return checksum


def create_dev_diag_bundle(
    inputs: DevDiagnosticBundleInputs,
    destination: str | Path,
) -> Path:
    """Create and re-verify one exact DEV-only deployment archive.

    The caller must freeze/commit the spec and code separately before using this
    function for a scientific run.  This function does not claim that freeze.
    """

    destination_path = Path(destination).absolute()
    if destination_path.exists() or Path(f"{destination_path}.sha256").exists():
        raise FileExistsError(f"Refusing to overwrite bundle/checksum at {destination_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    project_root = _checked_directory(inputs.project_root, label="project root")
    selected = _collect_bundle_inputs(inputs)
    git_identity = _git_source_identity(project_root)
    tracked_project_payloads = _git_tracked_project_payloads(project_root, selected)
    inventory = _inventory(selected)
    manifest = _bundle_manifest(inventory, git_identity, tracked_project_payloads)
    _validate_bundle_manifest(manifest)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.", suffix=".tmp", dir=destination_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        _write_tar_gz(
            temporary,
            archive_root=BUNDLE_ROOT,
            manifest_name=BUNDLE_MANIFEST,
            manifest=manifest,
            inputs=selected,
        )
        _verify_tar_inventory(
            temporary,
            archive_root=BUNDLE_ROOT,
            manifest_name=BUNDLE_MANIFEST,
            expected_kind=BUNDLE_KIND,
        )
        os.replace(temporary, destination_path)
        _write_checksum(destination_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        destination_path.unlink(missing_ok=True)
        Path(f"{destination_path}.sha256").unlink(missing_ok=True)
        raise
    return destination_path


def _resolve_extracted_root(path: Path, expected_root: str, manifest_name: str) -> Path:
    direct = path / manifest_name
    nested = path / expected_root / manifest_name
    if direct.is_file() and path.name == expected_root:
        return path
    if nested.is_file():
        return path / expected_root
    raise FileNotFoundError(f"Cannot locate {expected_root}/{manifest_name} below {path}")


def _verify_extracted_inventory(
    root: Path,
    manifest: Mapping[str, Any],
    inventory: Sequence[Mapping[str, Any]],
) -> None:
    observed: list[str] = []
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise ValueError(f"Extracted archive contains a symlink: {candidate}")
        if candidate.is_file():
            observed.append(candidate.relative_to(root).as_posix())
        elif not candidate.is_dir():
            raise ValueError(f"Extracted archive contains a non-regular entry: {candidate}")
    expected = {str(row["path"]) for row in inventory} | {
        BUNDLE_MANIFEST if manifest.get("kind") == BUNDLE_KIND else RESULTS_MANIFEST
    }
    _assert_no_casefold_collisions(observed, label="extracted archive")
    if set(observed) != expected:
        raise ValueError("Extracted archive files differ from the exact inventory")
    for row in inventory:
        path = _checked_regular_file(root / str(row["path"]), label="inventoried file")
        if path.stat().st_size != row["bytes"] or _sha256_path(path) != row["sha256"]:
            raise ValueError(f"Extracted archive hash mismatch for {row['path']}")


def verify_dev_diag_bundle(path: str | Path) -> dict[str, Any]:
    """Verify a DID deployment tarball or an already extracted bundle root."""

    source = Path(path).absolute()
    if source.is_file():
        return _verify_tar_inventory(
            source,
            archive_root=BUNDLE_ROOT,
            manifest_name=BUNDLE_MANIFEST,
            expected_kind=BUNDLE_KIND,
        )
    root = _resolve_extracted_root(
        _checked_directory(source, label="extracted bundle"), BUNDLE_ROOT, BUNDLE_MANIFEST
    )
    manifest = _read_json(root / BUNDLE_MANIFEST, label="DID-v1 bundle manifest")
    inventory = _validate_bundle_manifest(manifest)
    _verify_extracted_inventory(root, manifest, inventory)
    _validate_public_inputs(
        root / "inputs/public/MANIFEST.json",
        root / "inputs/public/cases.jsonl",
        root / "inputs/public/ANSWER_KEY_COMMITMENT.json",
    )
    _validate_historical_dev(
        root / "inputs/historical/MANIFEST.json",
        root / "inputs/historical/dev.jsonl",
    )
    checkpoint_files = {
        condition: _validate_checkpoint(
            root / "inputs/checkpoints" / condition, condition=condition
        )
        for condition in CHECKPOINT_CONDITIONS
    }
    _validate_bundle_cross_bindings(
        spec_path=root / "project/configs/stage1_dev_diag_v1.yaml",
        bridge_config_path=root / "project/configs/bridge_pilot.yaml",
        case_manifest_path=root / "inputs/public/MANIFEST.json",
        commitment_path=root / "inputs/public/ANSWER_KEY_COMMITMENT.json",
        historical_manifest_path=root / "inputs/historical/MANIFEST.json",
        dev_path=root / "inputs/historical/dev.jsonl",
        checkpoint_files=checkpoint_files,
    )
    return manifest


def _verified_bundle_manifest_and_sha256(
    bundle_root_or_manifest: str | Path | Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    if isinstance(bundle_root_or_manifest, Mapping):
        manifest = dict(bundle_root_or_manifest)
        _validate_bundle_manifest(manifest)
        return manifest, _sha256_bytes(_pretty_json_bytes(manifest))

    source = Path(bundle_root_or_manifest).absolute()
    manifest = verify_dev_diag_bundle(source)
    if source.is_file():
        with tarfile.open(source, "r:gz") as archive:
            handle = archive.extractfile(f"{BUNDLE_ROOT}/{BUNDLE_MANIFEST}")
            if handle is None:
                raise ValueError("Verified bundle manifest cannot be reread")
            manifest_bytes = handle.read()
    else:
        root = _resolve_extracted_root(source, BUNDLE_ROOT, BUNDLE_MANIFEST)
        manifest_bytes = (root / BUNDLE_MANIFEST).read_bytes()
    return manifest, _sha256_bytes(manifest_bytes)


def _strict_object(value: Any, *, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} schema differs")
    return value


def _is_posix_descendant(path_value: Any, root_value: Any) -> bool:
    if not isinstance(path_value, str) or not isinstance(root_value, str):
        return False
    path = PurePosixPath(path_value)
    root = PurePosixPath(root_value)
    if not path.is_absolute() or not root.is_absolute():
        return False
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_dependency_row(
    name: str, row_value: Any, *, venv_root: str, kind: str
) -> None:
    row = _strict_object(
        row_value,
        keys={
            "name",
            "canonical_name",
            "version",
            "location",
            "inside_venv",
            "origin_policy",
        },
        label=f"bootstrap {kind} dependency {name}",
    )
    if (
        row.get("canonical_name") != name
        or not isinstance(row.get("name"), str)
        or not row["name"]
        or not isinstance(row.get("version"), str)
        or not row["version"]
        or type(row.get("inside_venv")) is not bool
        or row["inside_venv"]
        != _is_posix_descendant(row.get("location"), venv_root)
    ):
        raise ValueError(f"bootstrap {kind} dependency {name} identity differs")
    if kind == "experiment":
        expected_policy = (
            "provider_torch_boundary" if name == "torch" else "diagnostic_venv"
        )
        expected_inside = name != "torch"
    else:
        expected_inside = bool(row["inside_venv"])
        expected_policy = (
            "diagnostic_venv_satisfies_torch"
            if expected_inside
            else "attested_provider_torch_support"
        )
        if not expected_inside and name not in PROVIDER_TORCH_SUPPORT_ALLOWLIST:
            raise ValueError(
                f"bootstrap provider dependency is not allowlisted: {name}"
            )
    if row["inside_venv"] is not expected_inside or row.get("origin_policy") != expected_policy:
        raise ValueError(f"bootstrap {kind} dependency {name} origin differs")


def verify_dev_diag_bootstrap_attestation(
    attestation_path: str | Path,
    bundle_root_or_manifest: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    """Verify paid runtime evidence and bind it to immutable bundle bytes.

    The deterministic return value is the exact object the evaluator must embed
    in its run manifest. Calling this function again after inference against the
    extracted bundle is also the end-of-run source/payload mutation check.
    """

    manifest, manifest_sha256 = _verified_bundle_manifest_and_sha256(
        bundle_root_or_manifest
    )
    attestation_file = _checked_regular_file(
        attestation_path, label="DID-v1 bootstrap attestation"
    )
    attestation = _read_json(attestation_file, label="DID-v1 bootstrap attestation")
    _strict_object(
        attestation,
        keys={
            "schema_version",
            "kind",
            "passed",
            "created_at_utc",
            "bundle",
            "hardware",
            "runtime",
            "dependency_closure",
            "kernel_contract",
            "source",
        },
        label="bootstrap attestation",
    )
    if (
        attestation.get("schema_version") != BOOTSTRAP_ATTESTATION_SCHEMA_VERSION
        or attestation.get("kind") != BOOTSTRAP_ATTESTATION_KIND
        or attestation.get("passed") is not True
        or not isinstance(attestation.get("created_at_utc"), str)
    ):
        raise ValueError("DID-v1 bootstrap attestation identity differs")
    try:
        created = datetime.fromisoformat(
            str(attestation["created_at_utc"]).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValueError("Bootstrap attestation timestamp is malformed") from exc
    if created.tzinfo is None:
        raise ValueError("Bootstrap attestation timestamp lacks a timezone")

    inventory = {str(row["path"]): row for row in manifest["inventory"]}
    project_hashes = {
        path: str(row["sha256"])
        for path, row in inventory.items()
        if path.startswith("project/")
    }
    source_identity = manifest["source_identity"]
    bundle = _strict_object(
        attestation.get("bundle"),
        keys={
            "manifest_sha256",
            "inventory_sha256",
            "source_identity_sha256",
            "git",
            "project_inventory_sha256",
            "executing_project_payload_sha256",
            "executing_project_payload_inventory_sha256",
        },
        label="bootstrap bundle binding",
    )
    expected_project_hash_inventory = _sha256_bytes(
        _canonical_json(project_hashes).encode("utf-8")
    )
    if (
        bundle.get("manifest_sha256") != manifest_sha256
        or bundle.get("inventory_sha256") != manifest["inventory_sha256"]
        or bundle.get("source_identity_sha256")
        != _sha256_bytes(_canonical_json(source_identity).encode("utf-8"))
        or bundle.get("git") != source_identity["git"]
        or bundle.get("project_inventory_sha256")
        != source_identity["project_inventory_sha256"]
        or bundle.get("executing_project_payload_sha256") != project_hashes
        or bundle.get("executing_project_payload_inventory_sha256")
        != expected_project_hash_inventory
    ):
        raise ValueError("Bootstrap attestation is not bound to the verified bundle/source")
    critical = {
        "project/src/under_extinction/dev_diag.py",
        "project/src/under_extinction/dev_diag_bootstrap.py",
        "project/src/under_extinction/dev_diag_deployment.py",
        "project/src/under_extinction/dev_diag_evaluation.py",
        "project/src/under_extinction/cli.py",
        "project/scripts/bootstrap_dev_diag.sh",
        "project/scripts/run_dev_diag_remote.sh",
    }
    if not critical <= set(project_hashes):
        raise ValueError("Verified bundle lacks critical executing DID-v1 source")

    hardware = _strict_object(
        attestation.get("hardware"),
        keys={
            "architecture",
            "hostname",
            "cuda_device_count",
            "device_name",
            "compute_capability",
            "total_memory_bytes",
            "total_memory_gib",
            "nvidia_smi_query",
        },
        label="bootstrap hardware",
    )
    capability = hardware.get("compute_capability")
    memory_bytes = hardware.get("total_memory_bytes")
    memory_gib = hardware.get("total_memory_gib")
    nvidia_rows = hardware.get("nvidia_smi_query")
    if (
        hardware.get("architecture") != "aarch64"
        or not isinstance(hardware.get("hostname"), str)
        or not hardware["hostname"]
        or hardware.get("cuda_device_count") != 1
        or type(hardware.get("cuda_device_count")) is not int
        or hardware.get("device_name") != "NVIDIA GH200 480GB"
        or not isinstance(capability, list)
        or len(capability) != 2
        or any(type(value) is not int for value in capability)
        or capability[0] != 9
        or type(memory_bytes) is not int
        or memory_bytes < 90 * 1024**3
        or not isinstance(memory_gib, (int, float))
        or isinstance(memory_gib, bool)
        or memory_gib < 90
        or abs(float(memory_gib) - memory_bytes / (1024**3)) > 1e-6
        or not isinstance(nvidia_rows, list)
        or len(nvidia_rows) != 1
        or not isinstance(nvidia_rows[0], str)
        or "NVIDIA GH200 480GB" not in nvidia_rows[0]
    ):
        raise ValueError("Bootstrap hardware is not exact 1x GH200/aarch64/CC9/>=90GiB")

    runtime = _strict_object(
        attestation.get("runtime"),
        keys={
            "python",
            "python_executable",
            "venv_root",
            "torch",
            "torch_path",
            "cuda",
            "cudnn",
            "numpy",
            "transformers",
            "peft",
            "qwen_text_loader",
            "module_origins",
        },
        label="bootstrap runtime",
    )
    torch_match = re.match(r"^(\d+)\.(\d+)(?:\.|$)", str(runtime.get("torch", "")))
    venv_root = runtime.get("venv_root")
    module_origins = runtime.get("module_origins")
    if (
        torch_match is None
        or not (2, 5) <= tuple(map(int, torch_match.groups())) < (3, 0)
        or not _is_posix_descendant(runtime.get("python_executable"), venv_root)
        or _is_posix_descendant(runtime.get("torch_path"), venv_root)
        or runtime.get("qwen_text_loader") != "Qwen3_5ForCausalLM"
        or not isinstance(runtime.get("cuda"), str)
        or not runtime["cuda"]
        or not isinstance(runtime.get("cudnn"), int)
        or not isinstance(module_origins, Mapping)
        or not module_origins
        or any(
            not _is_posix_descendant(path_value, venv_root)
            for path_value in module_origins.values()
        )
    ):
        raise ValueError("Bootstrap provider/venv runtime identity differs")

    closure = _strict_object(
        attestation.get("dependency_closure"),
        keys={
            "lock_sha256",
            "schema_version",
            "policy",
            "experiment_roots",
            "provider_torch_root",
            "provider_torch_support_allowlist",
            "experiment_closure",
            "provider_torch_closure",
            "installed_provider_allowlist_snapshot",
            "checks",
        },
        label="bootstrap dependency closure",
    )
    expected_checks = {
        "all_locked_roots_version_matched": True,
        "all_non_torch_experiment_dependencies_inside_venv": True,
        "provider_torch_is_explicit_experiment_boundary": True,
        "provider_torch_closure_fully_traversed": True,
        "every_external_torch_support_distribution_allowlisted": True,
    }
    experiment = closure.get("experiment_closure")
    provider = closure.get("provider_torch_closure")
    if (
        closure.get("lock_sha256")
        != inventory["project/requirements/h100-cu12x.lock"]["sha256"]
        or closure.get("schema_version") != "1.0"
        or closure.get("policy") != DEPENDENCY_CLOSURE_POLICY
        or not isinstance(closure.get("experiment_roots"), list)
        or not closure["experiment_roots"]
        or closure.get("provider_torch_root") != "torch>=2.5,<3"
        or closure.get("provider_torch_support_allowlist")
        != sorted(PROVIDER_TORCH_SUPPORT_ALLOWLIST)
        or closure.get("checks") != expected_checks
        or not isinstance(experiment, Mapping)
        or "torch" not in experiment
        or not isinstance(provider, Mapping)
        or "torch" not in provider
        or not isinstance(closure.get("installed_provider_allowlist_snapshot"), Mapping)
    ):
        raise ValueError("Bootstrap dependency-closure contract differs")
    for name, row in experiment.items():
        if not isinstance(name, str):
            raise ValueError("Bootstrap experiment closure key is malformed")
        _validate_dependency_row(name, row, venv_root=str(venv_root), kind="experiment")
    for name, row in provider.items():
        if not isinstance(name, str):
            raise ValueError("Bootstrap provider closure key is malformed")
        _validate_dependency_row(name, row, venv_root=str(venv_root), kind="provider_torch")
    runtime_distribution_fields = {
        "numpy": "numpy",
        "transformers": "transformers",
        "peft": "peft",
    }
    if any(
        name not in experiment or runtime.get(field) != experiment[name].get("version")
        for field, name in runtime_distribution_fields.items()
    ):
        raise ValueError("Bootstrap runtime versions differ from the audited experiment closure")
    if str(runtime["torch"]).split("+", 1)[0] != provider["torch"].get("version"):
        raise ValueError("Bootstrap provider Torch runtime/metadata versions differ")
    for name, row in closure["installed_provider_allowlist_snapshot"].items():
        if name not in PROVIDER_TORCH_SUPPORT_ALLOWLIST:
            raise ValueError("Bootstrap provider snapshot contains an unallowlisted package")
        snapshot = _strict_object(
            row,
            keys={"name", "canonical_name", "version", "location", "inside_venv"},
            label=f"bootstrap provider snapshot {name}",
        )
        if snapshot.get("canonical_name") != name or type(snapshot.get("inside_venv")) is not bool:
            raise ValueError(f"Bootstrap provider snapshot identity differs for {name}")
        if snapshot["inside_venv"] != _is_posix_descendant(
            snapshot.get("location"), str(venv_root)
        ):
            raise ValueError(f"Bootstrap provider snapshot origin differs for {name}")

    kernels = _strict_object(
        attestation.get("kernel_contract"),
        keys={
            "delta_net_policy",
            "optional_delta_net_packages_present",
            "torch_numpy_cpu_abi_roundtrip",
            "cuda_tensor_probe",
            "cuda_bfloat16_matmul_probe",
            "cuda_sdpa_probe",
        },
        label="bootstrap kernel contract",
    )
    if (
        kernels.get("delta_net_policy") != "torch_fallback_required"
        or kernels.get("optional_delta_net_packages_present")
        != {"causal-conv1d": False, "fla-core": False, "kernels": False}
        or any(
            kernels.get(key) is not True
            for key in (
                "torch_numpy_cpu_abi_roundtrip",
                "cuda_tensor_probe",
                "cuda_bfloat16_matmul_probe",
                "cuda_sdpa_probe",
            )
        )
    ):
        raise ValueError("Bootstrap ABI/kernel probes differ")

    source = _strict_object(
        attestation.get("source"),
        keys={
            "package_path",
            "diagnostic_import_probe",
            "python_file_count",
            "python_file_sha256",
            "python_inventory_sha256",
        },
        label="bootstrap source",
    )
    python_hashes = {
        path: digest
        for path, digest in project_hashes.items()
        if path.startswith("project/src/") and path.endswith(".py")
    }
    if (
        not isinstance(source.get("package_path"), str)
        or "/project/src/under_extinction/" not in source["package_path"].replace(
            "\\", "/"
        )
        or source.get("diagnostic_import_probe") is not True
        or source.get("python_file_count") != len(python_hashes)
        or source.get("python_file_sha256") != python_hashes
        or source.get("python_inventory_sha256")
        != _sha256_bytes(_canonical_json(python_hashes).encode("utf-8"))
    ):
        raise ValueError("Bootstrap executing Python source differs from the bundle")

    return {
        "schema_version": BOOTSTRAP_ATTESTATION_SCHEMA_VERSION,
        "kind": VERIFIED_BOOTSTRAP_BINDING_KIND,
        "attestation_sha256": _sha256_path(attestation_file),
        "bundle_manifest_sha256": manifest_sha256,
        "bundle_inventory_sha256": manifest["inventory_sha256"],
        "source_identity_sha256": _sha256_bytes(
            _canonical_json(source_identity).encode("utf-8")
        ),
        "executing_project_payload_inventory_sha256": expected_project_hash_inventory,
        "git_head_commit": source_identity["git"]["head_commit"],
        "gpu_uuid_query": list(nvidia_rows),
        "checks": {
            "bundle_reverified": True,
            "source_hashes_match_bundle": True,
            "git_source_identity_bound": True,
            "exact_gh200_runtime": True,
            "dependency_closures_valid": True,
            "torch_numpy_abi_valid": True,
            "kernel_probes_valid": True,
        },
    }


def _result_inputs(run_root: Path) -> list[_ArchiveInput]:
    selected: list[_ArchiveInput] = []
    for candidate in sorted(run_root.rglob("*"), key=lambda value: value.as_posix()):
        if candidate.is_symlink():
            raise ValueError(f"Result tree contains a symlink: {candidate}")
        if candidate.is_file():
            relative = candidate.relative_to(run_root).as_posix()
            relative_path = _safe_relative_path(relative, label="result path")
            if relative_path.parts[0] not in {"inference", "logs"}:
                raise ValueError(
                    f"Result collection accepts only inference/ and logs/: {relative}"
                )
            suffix = candidate.suffix.lower()
            if suffix in _FORBIDDEN_RESULT_SUFFIXES or suffix not in _ALLOWED_RESULT_SUFFIXES:
                raise ValueError(f"Result collection refuses non-evidence file: {relative}")
            if {part.casefold() for part in PurePosixPath(relative).parts} & _FORBIDDEN_PATH_PARTS:
                raise ValueError(f"Result collection refuses a cache/private path: {relative}")
            checked = _checked_regular_file(candidate, label="result evidence")
            _assert_no_secret_bytes(checked, label="result evidence")
            selected.append(_ArchiveInput(checked, f"run/{relative}", "result_evidence"))
        elif not candidate.is_dir():
            raise ValueError(f"Result tree contains a non-regular entry: {candidate}")
    if not selected:
        raise ValueError("Result collection received an empty evidence directory")
    _assert_no_casefold_collisions(
        [item.path for item in selected], label="DID-v1 result inventory"
    )
    return selected


def _results_manifest(
    inventory: Sequence[Mapping[str, Any]], run_manifest: Mapping[str, Any] | None
) -> dict[str, Any]:
    inventory_value = [dict(entry) for entry in inventory]
    reported_state = run_manifest.get("state") if isinstance(run_manifest, Mapping) else None
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "kind": RESULTS_KIND,
        "archive_root": RESULTS_ROOT,
        "inventory": inventory_value,
        "inventory_sha256": _sha256_bytes(_canonical_json(inventory_value).encode("utf-8")),
        "file_count": len(inventory_value),
        "payload_bytes": sum(int(entry["bytes"]) for entry in inventory_value),
        "run_status": {
            "run_manifest_present": run_manifest is not None,
            "reported_state": reported_state,
            "complete_claimed_by_run": reported_state == "COMPLETE",
            "partial_and_failed_runs_are_retrievable": True,
            "resume_supported": False,
            "completion_not_revalidated_by_collector": True,
        },
        "contract": {
            "weights_included": False,
            "checkpoint_state_included": False,
            "caches_included": False,
            "secrets_included": False,
            "allowed_suffixes": sorted(_ALLOWED_RESULT_SUFFIXES),
        },
    }


def _validate_results_manifest(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    if (
        manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION
        or manifest.get("kind") != RESULTS_KIND
        or manifest.get("archive_root") != RESULTS_ROOT
    ):
        raise ValueError("DID-v1 results manifest identity differs")
    contract = manifest.get("contract")
    expected_contract = _results_manifest([], None)["contract"]
    if not isinstance(contract, Mapping) or dict(contract) != expected_contract:
        raise ValueError("DID-v1 result exclusion contract differs")
    status_value = manifest.get("run_status")
    if not isinstance(status_value, Mapping) or (
        type(status_value.get("run_manifest_present")) is not bool
        or (
            status_value.get("reported_state") is not None
            and not isinstance(status_value.get("reported_state"), str)
        )
        or type(status_value.get("complete_claimed_by_run")) is not bool
        or status_value.get("complete_claimed_by_run")
        != (status_value.get("reported_state") == "COMPLETE")
        or status_value.get("partial_and_failed_runs_are_retrievable") is not True
        or status_value.get("resume_supported") is not False
        or status_value.get("completion_not_revalidated_by_collector") is not True
    ):
        raise ValueError("DID-v1 result status contract differs")
    raw_inventory = manifest.get("inventory")
    if not isinstance(raw_inventory, list) or not raw_inventory:
        raise ValueError("DID-v1 result inventory is empty")
    inventory: list[dict[str, Any]] = []
    paths: list[str] = []
    for index, raw in enumerate(raw_inventory):
        if not isinstance(raw, Mapping) or set(raw) != {
            "path",
            "role",
            "bytes",
            "sha256",
            "mode",
        }:
            raise ValueError(f"Malformed DID-v1 result inventory row {index}")
        row = dict(raw)
        path = row.get("path")
        if not isinstance(path, str) or not path.startswith("run/"):
            raise ValueError(f"Invalid DID-v1 result path at row {index}")
        relative = _safe_relative_path(path, label="result inventory path")
        if len(relative.parts) < 3 or relative.parts[1] not in {"inference", "logs"}:
            raise ValueError(f"DID-v1 result path leaves inference/logs: {path}")
        suffix = Path(relative.name).suffix.lower()
        if suffix not in _ALLOWED_RESULT_SUFFIXES or suffix in _FORBIDDEN_RESULT_SUFFIXES:
            raise ValueError(f"Forbidden DID-v1 result artifact: {path}")
        if (
            not isinstance(row.get("role"), str)
            or type(row.get("bytes")) is not int
            or row["bytes"] < 0
            or not isinstance(row.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
            or row.get("mode") != "0644"
        ):
            raise ValueError(f"Invalid DID-v1 result metadata at row {index}")
        inventory.append(row)
        paths.append(path)
    _assert_no_casefold_collisions(paths, label="DID-v1 result manifest")
    if (
        manifest.get("file_count") != len(inventory)
        or manifest.get("payload_bytes") != sum(row["bytes"] for row in inventory)
        or manifest.get("inventory_sha256")
        != _sha256_bytes(_canonical_json(inventory).encode("utf-8"))
    ):
        raise ValueError("DID-v1 result inventory totals/hash differ")
    return inventory


def collect_dev_diag_results(run_root: str | Path, destination: str | Path) -> Path:
    """Archive JSON/JSONL/log evidence from a complete, failed, or partial run."""

    root = _checked_directory(run_root, label="DID-v1 evidence root")
    destination_path = Path(destination).absolute()
    if destination_path.is_relative_to(root):
        raise ValueError("Result archive destination must be outside the evidence root")
    if destination_path.exists() or Path(f"{destination_path}.sha256").exists():
        raise FileExistsError(f"Refusing to overwrite result archive at {destination_path}")
    selected = _result_inputs(root)
    run_manifest_path = root / "inference" / "run_manifest.json"
    run_manifest = (
        _read_json(run_manifest_path, label="inference run manifest")
        if run_manifest_path.is_file()
        else None
    )
    inventory = _inventory(selected)
    manifest = _results_manifest(inventory, run_manifest)
    _validate_results_manifest(manifest)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.", suffix=".tmp", dir=destination_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        _write_tar_gz(
            temporary,
            archive_root=RESULTS_ROOT,
            manifest_name=RESULTS_MANIFEST,
            manifest=manifest,
            inputs=selected,
        )
        _verify_tar_inventory(
            temporary,
            archive_root=RESULTS_ROOT,
            manifest_name=RESULTS_MANIFEST,
            expected_kind=RESULTS_KIND,
        )
        os.replace(temporary, destination_path)
        _write_checksum(destination_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        destination_path.unlink(missing_ok=True)
        Path(f"{destination_path}.sha256").unlink(missing_ok=True)
        raise
    return destination_path


def verify_dev_diag_results(path: str | Path) -> dict[str, Any]:
    """Verify a retrieved result tarball or extracted results directory."""

    source = Path(path).absolute()
    if source.is_file():
        return _verify_tar_inventory(
            source,
            archive_root=RESULTS_ROOT,
            manifest_name=RESULTS_MANIFEST,
            expected_kind=RESULTS_KIND,
        )
    root = _resolve_extracted_root(
        _checked_directory(source, label="extracted results"), RESULTS_ROOT, RESULTS_MANIFEST
    )
    manifest = _read_json(root / RESULTS_MANIFEST, label="DID-v1 results manifest")
    inventory = _validate_results_manifest(manifest)
    _verify_extracted_inventory(root, manifest, inventory)
    return manifest


__all__ = [
    "BOOTSTRAP_ATTESTATION_KIND",
    "BOOTSTRAP_ATTESTATION_SCHEMA_VERSION",
    "BUNDLE_MANIFEST",
    "BUNDLE_ROOT",
    "DevDiagnosticBundleInputs",
    "RESULTS_MANIFEST",
    "RESULTS_ROOT",
    "VERIFIED_BOOTSTRAP_BINDING_KIND",
    "collect_dev_diag_results",
    "create_dev_diag_bundle",
    "verify_dev_diag_bootstrap_attestation",
    "verify_dev_diag_bundle",
    "verify_dev_diag_results",
]
