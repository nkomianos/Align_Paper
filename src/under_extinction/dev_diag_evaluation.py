"""Inference-only execution for the post-hoc DID-v1 development diagnostic.

This module is deliberately separate from the registered Stage-1 evaluator.  It
loads no bridge environment state, never parses a frozen data split, and cannot
unlock TEST.  Its only use of the historical DEV corpus is to verify the bytes
to which the diagnostic generator was bound.

The evaluator is intentionally strict:

* the historical config, DEV data, runtime attestation, and three adapter
  checkpoints must match the released failed Stage-1 artifacts byte for byte;
* the diagnostic cases and answer-key commitment must match their frozen
  manifest before any model is loaded;
* unchanged base, checkpoint-zero, genuine-final, and proxy-final each receive a
  fresh base-model load;
* all forward passes run under ``torch.inference_mode`` and every parameter is
  checked to be frozen;
* policy outputs are committed as atomic directories and an existing
  destination is never reused.

Nothing emitted here is registered Stage-1 evidence.  In particular, a DID-v1
result cannot reverse the failed gate or authorize access to the locked split.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import yaml

from .bridge_evaluation import (
    LEGAL_CHOICE_LOG_MASS_TOLERANCE,
    legal_choice_diagnostics,
    parse_unconstrained_choice,
)
from .bridge_training import differentiable_choice_log_probs
from .config import config_hash, validate_bridge_config
from .dev_diag_deployment import (
    BUNDLE_MANIFEST,
    BUNDLE_ROOT,
    verify_dev_diag_bootstrap_attestation,
)
from .io import canonical_json, write_json, write_jsonl
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


POSTHOC_STATUS = "post_hoc_exploratory_failure_localization"
POSTHOC_EVIDENCE_KIND = "did_v1_posthoc_dev_diagnostic_not_stage1_evidence"
POLICY_CONDITIONS = (
    "unchanged_base",
    "checkpoint_zero",
    "genuine_final",
    "proxy_final",
)

# These constants are a second trust anchor, independent of the YAML spec.  A
# jointly modified spec and artifact set must not silently become "historical
# Stage 1".  Tests replace the dictionaries with synthetic fixtures.
FROZEN_HISTORICAL_CONTRACT: dict[str, Any] = {
    "archive_sha256": (
        "322770769e99e8de9c1913a0f3b506831e0123beed19213d53279e13f56cfec9"
    ),
    "stage1_release_tag": "stage1-dev-20260819-failed",
    "stage1_release_commit": "45650ed87720b32f797ec098d5b62b135aacfeec",
    "historical_training_commit": "1d113fe181820c8741e73f42b2dcfcb045b17185",
    "stage1_report_sha256": (
        "16c1ef0af82849d2bab1cc2ed2c9261902d387b5506df91cd567c79af27bd4c0"
    ),
    "bridge_config_file_sha256": (
        "5daceb097afbc0f3733653201c1c32cfb1a3202c4e2c5a9eb7003d8ec5fd424a"
    ),
    "bridge_config_canonical_sha256": (
        "3d1084bfe5358343e9ad0581895e63129cad42acadcaa38713b476d2e74b5389"
    ),
    "data_manifest_sha256": (
        "694e6e7927be06993611a2d8a4b0ff0a01248ca7742fc93bae5832c503135ff5"
    ),
    "dev_file_sha256": (
        "45b101815ad651d1fa2e68ab34cd37e5b50be405e1faae0208feb9b4d119e76b"
    ),
    "pair_seed": 11,
    "initial_environment_state_sha256": (
        "784db50f7f9a7285115f645085de8e074914165eaa387c9196814e9f7226a323"
    ),
    "model_runtime_attestation_sha256": (
        "b8a0a16d861c2208c9d2581a084ec5a1847f7156aec9a71c4e3c0c3ce714e0ec"
    ),
    "model_id": "Qwen/Qwen3.5-9B",
    "model_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    "bridge_spec_sha256": (
        "cd50d12ed76bae287f15bb94028e67ae5909d0a927e7d795d2d87474761facd4"
    ),
}

FROZEN_CHECKPOINT_CONTRACTS: dict[str, dict[str, Any]] = {
    "checkpoint_zero": {
        "arm": "genuine",
        "update": 0,
        "checkpoint_manifest_sha256": (
            "c8cdd66d0e486fa64eeb5e1e47fb14510864587b94821cb7ca3d728585805ca0"
        ),
        "adapter_config_sha256": (
            "bf04315fa3ac4bdf5e12bb939e54ae92a39052be6775b89f4b80f2ee08992649"
        ),
        "adapter_model_sha256": (
            "31805f350f6cddab6c8325944273ce78b01335f92275115dce4db78a8e8a754c"
        ),
    },
    "genuine_final": {
        "arm": "genuine",
        "update": 300,
        "checkpoint_manifest_sha256": (
            "fb1180cd94f9e94b3f3715c1b8fb858f884335d35abf50e4ad6a927a7291d075"
        ),
        "adapter_config_sha256": (
            "bf04315fa3ac4bdf5e12bb939e54ae92a39052be6775b89f4b80f2ee08992649"
        ),
        "adapter_model_sha256": (
            "a99d73b12df9fcc999d03917234d523fe6388a24007596c032b5ef7cd7b7d247"
        ),
    },
    "proxy_final": {
        "arm": "proxy",
        "update": 300,
        "checkpoint_manifest_sha256": (
            "1279c4a809024c6cbe8707579767fe1552cc91c22ac061fb96e9993e88a8857a"
        ),
        "adapter_config_sha256": (
            "bf04315fa3ac4bdf5e12bb939e54ae92a39052be6775b89f4b80f2ee08992649"
        ),
        "adapter_model_sha256": (
            "52b5d70c999ca054ad095cbe32b45e915b65c83e8cf8ce1969e4345fb05bfc5f"
        ),
    },
}

FROZEN_INFERENCE_CONTRACT: dict[str, Any] = {
    "training": False,
    "optimizer": False,
    "reward_feedback": False,
    "parameter_updates": False,
    "batch_size": 32,
    "generation_subset_size": 256,
    "generation_batch_size": 8,
    "max_new_tokens": 1,
    "checkpoint_zero_base_probability_tolerance": 0.001,
    "token_length_audit": {
        "required_before_model_load": True,
        "truncation_allowed": False,
        "expected_prompt_count": 19_200,
        "expected_max_prompt_tokens": 745,
        "expected_all_prompt_token_counts_sha256": (
            "19987209a6211f3152ae6f8f8a4fbfc56242792354989a11f7934c49d11383ff"
        ),
        "expected_generation_subset_token_counts_sha256": (
            "6e161c3503e74e914e07b3aa179c6beffa71dc9814815fbea44e03abe6710d88"
        ),
        "expected_chat_template_sha256": (
            "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"
        ),
        "expected_ordered_case_candidate_token_ids_sha256": (
            "9272c58e8a64e25384a1af6b0ba91d3663638b65aa1258293c80152e330586f4"
        ),
        "require_exact_max": True,
        "require_choice_boundary_single_token": True,
        "require_prompt_plus_generation_within_max_length": True,
    },
}
FROZEN_EXPECTED_PROMPT_COUNT = 19_200

_ANSWER_KEY_COMMITMENT_KEYS = {
    "schema_version",
    "kind",
    "diagnostic_id",
    "case_set_sha256",
    "answer_key_sha256",
    "record_count",
    "answer_key_external_to_model_visible_bundle",
}
_MANIFEST_ANSWER_KEY_KEYS = {"sha256", "count", "external", "path_disclosed"}
_CASE_MANIFEST_KEYS = {
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
}
_CASE_MANIFEST_COUNT_KEYS = {
    "static_prompts",
    "static_semantic_units",
    "update_prompts",
    "update_semantic_units",
    "total_prompts",
}
_CASE_MANIFEST_SOURCE_PARENT_KEYS = {
    "data_manifest_sha256",
    "dev_file_sha256",
    "dev_file_bytes",
    "dev_record_count",
}
_CASE_MANIFEST_GENERATION_SUBSET_KEYS = {
    "method",
    "size",
    "ordered_case_ids_sha256",
    "case_ids",
}
_CASE_MANIFEST_TEMPLATE_PROVENANCE_KEYS = {
    "audit_renderer_ids",
    "calibration_renderer_ids",
    "renderer_template_sha256",
    "calibration_and_audit_renderer_sets_disjoint",
    "calibration_not_model_scored",
}
_FORMAL_LICENSE_OUTCOME = "LICENSES_NEW_PREREGISTERED_E1B_DEV2_ONLY"
_BUNDLE_ROOT_ENV = "UE_DEV_DIAG_BUNDLE_ROOT"
_BOOTSTRAP_ATTESTATION_ENV = "UE_DEV_DIAG_BOOTSTRAP_ATTESTATION"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_without_symlink(path: str | Path, *, label: str) -> Path:
    lexical = Path(path)
    if lexical.is_symlink():
        raise ValueError(f"{label} may not be a symlink: {lexical.absolute()}")
    return lexical.resolve()


def _resolve_extracted_deployment_root(path: str | Path) -> Path:
    """Resolve only the extracted immutable DID-v1 bundle, never a tarball."""

    supplied = _resolve_without_symlink(path, label="DID-v1 deployment root")
    candidates = (supplied, supplied / BUNDLE_ROOT)
    for candidate in candidates:
        if candidate.name == BUNDLE_ROOT and (candidate / BUNDLE_MANIFEST).is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Cannot locate extracted {BUNDLE_ROOT}/{BUNDLE_MANIFEST} below {supplied}"
    )


def _deployment_context_paths(
    *,
    deployment_root: str | Path | None,
    bootstrap_attestation_path: str | Path | None,
) -> tuple[Path, Path]:
    root_value = deployment_root or os.environ.get(_BUNDLE_ROOT_ENV)
    attestation_value = bootstrap_attestation_path or os.environ.get(
        _BOOTSTRAP_ATTESTATION_ENV
    )
    if not root_value or not attestation_value:
        raise ValueError(
            "DID-v1 requires the verified deployment root and bootstrap attestation; "
            f"set {_BUNDLE_ROOT_ENV} and {_BOOTSTRAP_ATTESTATION_ENV}"
        )
    root = _resolve_extracted_deployment_root(root_value)
    attestation = _resolve_without_symlink(
        attestation_value, label="DID-v1 bootstrap attestation"
    )
    if not attestation.is_file():
        raise FileNotFoundError(
            f"DID-v1 bootstrap attestation is missing: {attestation}"
        )
    return root, attestation


def _require_bundle_payload_paths(
    deployment_root: Path,
    supplied: Mapping[str, str | Path],
) -> None:
    expected_relatives = {
        "bridge_config": "project/configs/bridge_pilot.yaml",
        "spec": "project/configs/stage1_dev_diag_v1.yaml",
        "case_manifest": "inputs/public/MANIFEST.json",
        "cases": "inputs/public/cases.jsonl",
        "answer_key_commitment": "inputs/public/ANSWER_KEY_COMMITMENT.json",
        "data_manifest": "inputs/historical/MANIFEST.json",
        "dev_data": "inputs/historical/dev.jsonl",
        "checkpoint_zero": "inputs/checkpoints/checkpoint_zero",
        "genuine_final": "inputs/checkpoints/genuine_final",
        "proxy_final": "inputs/checkpoints/proxy_final",
    }
    unknown = set(supplied) - set(expected_relatives)
    if unknown:
        raise ValueError(f"Unknown DID-v1 deployment payload labels: {sorted(unknown)}")
    for label, value in supplied.items():
        observed = _resolve_without_symlink(value, label=f"DID-v1 {label}")
        expected = (deployment_root / PurePosixPath(expected_relatives[label])).resolve()
        if observed != expected:
            raise ValueError(
                f"DID-v1 {label} is not the attested deployment payload: "
                f"observed={observed}, expected={expected}"
            )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], *, label: str
) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"{label} has the wrong fields (missing={missing}, extra={extra})"
        )


def _valid_messages(messages: Any) -> bool:
    return (
        isinstance(messages, list)
        and bool(messages)
        and all(
            isinstance(message, dict)
            and set(message) == {"role", "content"}
            and message["role"] in {"system", "user", "assistant"}
            and isinstance(message["content"], str)
            and bool(message["content"])
            for message in messages
        )
        and messages[-1]["role"] == "user"
    )


class _ScientificInputLedger:
    """Allowlist and attest every scientific file this module may open."""

    def __init__(self, allowed: Sequence[str | Path]) -> None:
        resolved = [
            _resolve_without_symlink(path, label="Scientific input") for path in allowed
        ]
        if len(resolved) != len(set(resolved)):
            raise ValueError("Scientific input allowlist contains duplicate paths")
        forbidden_names = {"test.jsonl", "locked_test.jsonl", "test.json"}
        if any(path.name.lower() in forbidden_names for path in resolved):
            raise PermissionError("DID-v1 scientific input allowlist may not contain TEST")
        self._allowed = frozenset(resolved)
        self._opened: set[Path] = set()
        self._hashes: dict[Path, str] = {}
        for path in resolved:
            if not path.is_file():
                raise FileNotFoundError(f"Missing allowlisted scientific input: {path}")
            if path.is_symlink():
                raise ValueError(f"Scientific input may not be a symlink: {path}")

    def _record(self, path: str | Path) -> Path:
        resolved = _resolve_without_symlink(path, label="Scientific input")
        if resolved not in self._allowed:
            raise PermissionError(f"Scientific input is outside the frozen allowlist: {resolved}")
        self._opened.add(resolved)
        return resolved

    def read_text(self, path: str | Path) -> str:
        return self._record(path).read_text(encoding="utf-8")

    def sha256(self, path: str | Path, *, refresh: bool = False) -> str:
        resolved = self._record(path)
        if not refresh and resolved in self._hashes:
            return self._hashes[resolved]
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        value = digest.hexdigest()
        if refresh and resolved in self._hashes and self._hashes[resolved] != value:
            raise ValueError(f"Scientific input changed during evaluation: {resolved}")
        self._hashes[resolved] = value
        return value

    def read_json(self, path: str | Path) -> Any:
        try:
            return json.loads(self.read_text(path))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON scientific input: {Path(path).resolve()}") from exc

    def read_jsonl(self, path: str | Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(self.read_text(path).splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL scientific input at {Path(path).resolve()}:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"Diagnostic case {line_number} is not a JSON object")
            rows.append(row)
        return rows

    def attestation(self) -> dict[str, Any]:
        unopened = self._allowed - self._opened
        if unopened:
            raise RuntimeError(
                "Not every frozen scientific input was attested: "
                + ", ".join(str(path) for path in sorted(unopened))
            )
        return {
            "schema_version": "1.0",
            "access_policy": "exact_path_allowlist_no_locked_split",
            "allowed_paths": [str(path) for path in sorted(self._allowed)],
            "opened_paths": [str(path) for path in sorted(self._opened)],
            "opened_equals_allowlist": self._opened == set(self._allowed),
            "opened_file_sha256": {
                str(path): self.sha256(path) for path in sorted(self._allowed)
            },
            "locked_test_accessed": False,
            "dev_content_parsed": False,
            "dev_use": "hash_verification_only",
        }


@dataclass(frozen=True)
class VerifiedCheckpoint:
    policy_condition: str
    path: Path
    manifest: dict[str, Any]
    checkpoint_manifest_sha256: str
    adapter_config_sha256: str
    adapter_model_sha256: str
    runtime_attestation: dict[str, Any]

    def public_attestation(self) -> dict[str, Any]:
        return {
            "policy_condition": self.policy_condition,
            "path": str(self.path),
            "arm": self.manifest["arm"],
            "completed_updates": int(self.manifest["completed_updates"]),
            "pair_seed": int(self.manifest["pair_seed"]),
            "initial_environment_state_sha256": self.manifest[
                "initial_environment_state_sha256"
            ],
            "checkpoint_manifest_sha256": self.checkpoint_manifest_sha256,
            "adapter_config_sha256": self.adapter_config_sha256,
            "adapter_model_sha256": self.adapter_model_sha256,
            "model_runtime_attestation_sha256": self.runtime_attestation[
                "attestation_sha256"
            ],
            "bridge_state_opened": False,
        }


@dataclass(frozen=True)
class TokenLengthAudit:
    """Frozen no-truncation proof produced before any policy model is loaded."""

    report: dict[str, Any]
    report_sha256: str
    prompt_token_counts: dict[str, int]

    def binding(self) -> dict[str, Any]:
        return {
            "sha256": self.report_sha256,
            "report": json.loads(canonical_json(self.report)),
        }


@dataclass
class VerifiedDevDiagnosticInputs:
    bridge_config: dict[str, Any]
    spec: dict[str, Any]
    case_manifest: dict[str, Any]
    cases: list[dict[str, Any]]
    generation_case_ids: list[str]
    checkpoints: dict[str, VerifiedCheckpoint]
    ledger: _ScientificInputLedger
    bindings: dict[str, Any]
    deployment_root: Path
    bootstrap_attestation_path: Path
    bootstrap_verification: dict[str, Any]
    token_length_audit: "TokenLengthAudit | None" = None


@dataclass(frozen=True)
class VerifiedCompletedDevDiagnosticRun:
    """A COMPLETE run plus independently reverified public analysis inputs."""

    run_dir: Path
    run_manifest_path: Path
    run_manifest_sha256: str
    prediction_paths: tuple[Path, ...]
    input_artifact_verification: dict[str, Any]
    deployment_root: Path
    bootstrap_attestation_path: Path
    bootstrap_verification: dict[str, Any]


def _core_load_spec(path: str | Path) -> dict[str, Any]:
    from .dev_diag import load_dev_diag_spec

    return load_dev_diag_spec(path)


def _core_validate_cases(
    records: Sequence[Mapping[str, Any]], spec: Mapping[str, Any]
) -> list[dict[str, Any]]:
    from .dev_diag import validate_dev_diag_cases

    return [dict(row) for row in validate_dev_diag_cases(records, spec)]


def _core_generation_subset_case_ids(
    records: Sequence[Mapping[str, Any]], spec: Mapping[str, Any]
) -> list[str]:
    from .dev_diag import generation_subset_case_ids

    return [str(value) for value in generation_subset_case_ids(records, spec)]


def _validate_frozen_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    if spec.get("kind") != "bridge_posthoc_dev_diagnostic_spec":
        raise ValueError("DID-v1 spec has the wrong kind")
    if spec.get("scientific_status") != POSTHOC_STATUS:
        raise ValueError("DID-v1 spec is not marked post-hoc exploratory")
    if spec.get("diagnostic_id") != "stage1_dev_diag_v1":
        raise ValueError("Unexpected diagnostic_id")
    access = spec.get("access_contract")
    if not isinstance(access, Mapping) or (
        access.get("allowed_split") != "dev"
        or access.get("other_split_access") != "forbidden"
        or access.get("locked_test_accessed") is not False
    ):
        raise PermissionError("DID-v1 spec does not enforce the DEV-only access boundary")
    if list(spec.get("policy_conditions") or []) != list(POLICY_CONDITIONS):
        raise ValueError("DID-v1 policy conditions differ from the frozen four-policy design")
    if dict(spec.get("inference_contract") or {}) != FROZEN_INFERENCE_CONTRACT:
        raise ValueError("DID-v1 inference contract differs from the frozen design")
    generation = spec.get("generation")
    if (
        not isinstance(generation, Mapping)
        or int(generation.get("expected_total_prompt_count", -1))
        != FROZEN_EXPECTED_PROMPT_COUNT
    ):
        raise ValueError("DID-v1 prompt count differs from the frozen full grid")
    model = spec.get("model")
    if not isinstance(model, Mapping) or (
        model.get("id") != FROZEN_HISTORICAL_CONTRACT["model_id"]
        or model.get("revision") != FROZEN_HISTORICAL_CONTRACT["model_revision"]
        or model.get("max_length") != 768
        or model.get("choice_labels") != ["A", "B"]
        or model.get("enable_thinking") is not False
        or model.get("use_kernels") is not False
    ):
        raise ValueError("DID-v1 model contract differs from frozen Qwen3.5-9B")
    parents = spec.get("parents")
    if not isinstance(parents, Mapping):
        raise ValueError("DID-v1 spec lacks parent artifact bindings")
    for key in (
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
    ):
        if parents.get(key) != FROZEN_HISTORICAL_CONTRACT[key]:
            raise ValueError(f"DID-v1 historical parent differs for {key}")
    for policy, expected in FROZEN_CHECKPOINT_CONTRACTS.items():
        if dict(parents.get(policy) or {}) != expected:
            raise ValueError(f"DID-v1 checkpoint contract differs for {policy}")
    decision = spec.get("decision_contract")
    if not isinstance(decision, Mapping) or (
        decision.get("cannot_reverse_stage1") is not True
        or decision.get("cannot_open_locked_test") is not True
        or decision.get("cannot_authorize_replication") is not True
    ):
        raise ValueError("DID-v1 interpretation boundary is not frozen")
    return json.loads(canonical_json(dict(spec)))


def _checkpoint_paths(checkpoint: str | Path) -> dict[str, Path]:
    root = Path(checkpoint).resolve()
    return {
        "manifest": root / "checkpoint_manifest.json",
        "adapter_config": root / "adapter_config.json",
        "adapter_model": root / "adapter_model.safetensors",
    }


def _verify_checkpoint(
    *,
    policy_condition: str,
    checkpoint: str | Path,
    expected: Mapping[str, Any],
    config: Mapping[str, Any],
    ledger: _ScientificInputLedger,
) -> VerifiedCheckpoint:
    root = Path(checkpoint).resolve()
    paths = _checkpoint_paths(root)
    observed = {
        "checkpoint_manifest_sha256": ledger.sha256(paths["manifest"]),
        "adapter_config_sha256": ledger.sha256(paths["adapter_config"]),
        "adapter_model_sha256": ledger.sha256(paths["adapter_model"]),
    }
    for key, digest in observed.items():
        if digest != expected[key]:
            raise ValueError(f"{policy_condition} differs from frozen {key}")
    manifest = ledger.read_json(paths["manifest"])
    adapter_config = ledger.read_json(paths["adapter_config"])
    if not isinstance(manifest, dict) or manifest.get("kind") != "bridge_policy_checkpoint":
        raise ValueError(f"Malformed checkpoint manifest for {policy_condition}")
    if not isinstance(adapter_config, dict):
        raise ValueError(f"Malformed adapter config for {policy_condition}")
    expected_fields = {
        "arm": expected["arm"],
        "completed_updates": int(expected["update"]),
        "pair_seed": int(FROZEN_HISTORICAL_CONTRACT["pair_seed"]),
        "config_sha256": FROZEN_HISTORICAL_CONTRACT[
            "bridge_config_canonical_sha256"
        ],
        "bridge_spec_sha256": FROZEN_HISTORICAL_CONTRACT["bridge_spec_sha256"],
        "initial_environment_state_sha256": FROZEN_HISTORICAL_CONTRACT[
            "initial_environment_state_sha256"
        ],
        "model_runtime_attestation_sha256": FROZEN_HISTORICAL_CONTRACT[
            "model_runtime_attestation_sha256"
        ],
    }
    for key, value in expected_fields.items():
        if manifest.get(key) != value or type(manifest.get(key)) is not type(value):
            raise ValueError(f"{policy_condition} checkpoint differs for {key}")
    declared_files = manifest.get("file_sha256")
    if not isinstance(declared_files, Mapping) or (
        declared_files.get("adapter_config.json") != observed["adapter_config_sha256"]
        or declared_files.get("adapter_model.safetensors")
        != observed["adapter_model_sha256"]
    ):
        raise ValueError(f"{policy_condition} manifest does not bind its exact adapter")
    provenance = manifest.get("environment_provenance")
    if not isinstance(provenance, Mapping) or (
        provenance.get("config_sha256")
        != FROZEN_HISTORICAL_CONTRACT["bridge_config_canonical_sha256"]
        or provenance.get("data_manifest_sha256")
        != FROZEN_HISTORICAL_CONTRACT["data_manifest_sha256"]
        or dict(provenance.get("file_sha256") or {}).get("dev")
        != FROZEN_HISTORICAL_CONTRACT["dev_file_sha256"]
    ):
        raise ValueError(f"{policy_condition} has different historical data provenance")
    runtime = manifest.get("model_runtime_attestation")
    if not isinstance(runtime, Mapping):
        raise ValueError(f"{policy_condition} lacks a runtime attestation")
    verified_runtime = verify_model_runtime_attestation(config, runtime)
    if (
        verified_runtime.get("attestation_sha256")
        != FROZEN_HISTORICAL_CONTRACT["model_runtime_attestation_sha256"]
    ):
        raise ValueError(f"{policy_condition} runtime attestation differs")
    return VerifiedCheckpoint(
        policy_condition=policy_condition,
        path=root,
        manifest=json.loads(canonical_json(manifest)),
        checkpoint_manifest_sha256=observed["checkpoint_manifest_sha256"],
        adapter_config_sha256=observed["adapter_config_sha256"],
        adapter_model_sha256=observed["adapter_model_sha256"],
        runtime_attestation=verified_runtime,
    )


def _reject_recursive_manifest_answer_material(
    value: Any, *, path: tuple[str, ...] = ()
) -> None:
    """Reject answer/target payloads anywhere in the public case manifest."""

    forbidden = {
        "answer",
        "answers",
        "answer_key_records",
        "correct",
        "correct_action",
        "expected_action",
        "expected_actions",
        "expected_answer",
        "expected_by_policy",
        "oracle_action",
        "oracle_actions",
        "target",
        "target_action",
    }
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            if key.casefold() in forbidden:
                location = ".".join((*path, key))
                raise ValueError(
                    f"Public DID-v1 case manifest contains answer/target material at {location}"
                )
            _reject_recursive_manifest_answer_material(
                child, path=(*path, key)
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_recursive_manifest_answer_material(
                child, path=(*path, str(index))
            )


def _validate_case_manifest_exact_schema(
    manifest: Mapping[str, Any], spec: Mapping[str, Any]
) -> None:
    _require_exact_keys(
        manifest, _CASE_MANIFEST_KEYS, label="DID-v1 public case manifest"
    )
    _reject_recursive_manifest_answer_material(manifest)
    if (
        manifest.get("schema_version") != "DID-v1"
        or manifest.get("kind") != "did_v1_model_visible_case_manifest"
        or manifest.get("diagnostic_id") != spec.get("diagnostic_id")
        or manifest.get("scientific_status") != spec.get("scientific_status")
        or manifest.get("generator_version")
        != dict(spec.get("generation") or {}).get("generator_version")
        or manifest.get("split") != "dev"
        or manifest.get("locked_test_opened_or_parsed") is not False
        or manifest.get("existing_dev_prompts_reused") is not False
    ):
        raise ValueError("DID-v1 public case-manifest identity differs")

    access = manifest.get("access_contract")
    if not isinstance(access, Mapping) or dict(access) != dict(
        spec.get("access_contract") or {}
    ):
        raise ValueError("DID-v1 case-manifest access contract differs")
    if manifest.get("parents") != spec.get("parents"):
        raise ValueError("DID-v1 case-manifest historical parents differ")

    source_parent = manifest.get("verified_source_parent")
    if not isinstance(source_parent, Mapping):
        raise ValueError("DID-v1 case manifest lacks verified source provenance")
    _require_exact_keys(
        source_parent,
        _CASE_MANIFEST_SOURCE_PARENT_KEYS,
        label="DID-v1 case-manifest verified_source_parent",
    )
    if (
        not _is_sha256(source_parent.get("data_manifest_sha256"))
        or not _is_sha256(source_parent.get("dev_file_sha256"))
        or source_parent.get("data_manifest_sha256")
        != dict(spec.get("parents") or {}).get("data_manifest_sha256")
        or source_parent.get("dev_file_sha256")
        != dict(spec.get("parents") or {}).get("dev_file_sha256")
        or type(source_parent.get("dev_file_bytes")) is not int
        or source_parent["dev_file_bytes"] <= 0
        or type(source_parent.get("dev_record_count")) is not int
        or source_parent["dev_record_count"] <= 0
    ):
        raise ValueError("DID-v1 verified source-parent provenance is malformed")

    counts = manifest.get("counts")
    if not isinstance(counts, Mapping):
        raise ValueError("DID-v1 case manifest lacks exact counts")
    _require_exact_keys(
        counts, _CASE_MANIFEST_COUNT_KEYS, label="DID-v1 case-manifest counts"
    )
    generation = dict(spec.get("generation") or {})
    static = dict(generation.get("static") or {})
    update = dict(generation.get("update") or {})
    expected_counts = {
        "static_prompts": static.get("expected_prompt_count"),
        "static_semantic_units": static.get("world_count"),
        "update_prompts": update.get("expected_prompt_count"),
        "update_semantic_units": update.get("semantic_unit_count"),
        "total_prompts": generation.get("expected_total_prompt_count"),
    }
    if any(
        type(expected) is not int
        or counts.get(key) != expected
        or type(counts.get(key)) is not int
        for key, expected in expected_counts.items()
    ):
        raise ValueError("DID-v1 case-manifest factorial counts differ")

    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("DID-v1 case manifest lacks files")
    _require_exact_keys(
        files,
        {"cases", "answer_key_commitment"},
        label="DID-v1 case-manifest files",
    )

    subset = manifest.get("generation_subset")
    if not isinstance(subset, Mapping):
        raise ValueError("DID-v1 case manifest lacks generation subset")
    _require_exact_keys(
        subset,
        _CASE_MANIFEST_GENERATION_SUBSET_KEYS,
        label="DID-v1 case-manifest generation_subset",
    )
    if subset.get("method") != (
        "four_hash_ranked_cases_per_panel_module_cue_renderer_label_stratum_v1"
    ):
        raise ValueError("DID-v1 generation-subset method differs")

    provenance = manifest.get("template_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("DID-v1 case manifest lacks template provenance")
    _require_exact_keys(
        provenance,
        _CASE_MANIFEST_TEMPLATE_PROVENANCE_KEYS,
        label="DID-v1 case-manifest template_provenance",
    )
    audit_ids = generation.get("audit_renderer_ids")
    calibration_ids = generation.get("calibration_renderer_ids")
    renderer_hashes = provenance.get("renderer_template_sha256")
    if (
        not isinstance(audit_ids, list)
        or not isinstance(calibration_ids, list)
        or provenance.get("audit_renderer_ids") != audit_ids
        or provenance.get("calibration_renderer_ids") != calibration_ids
        or provenance.get("calibration_and_audit_renderer_sets_disjoint") is not True
        or provenance.get("calibration_not_model_scored") is not True
        or not isinstance(renderer_hashes, Mapping)
        or set(renderer_hashes) != set(audit_ids + calibration_ids)
        or not all(_is_sha256(digest) for digest in renderer_hashes.values())
    ):
        raise ValueError("DID-v1 case-manifest template provenance differs")


def _validate_answer_key_commitment(
    commitment: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    spec: Mapping[str, Any],
    cases_sha256: str,
    case_count: int,
) -> dict[str, Any]:
    """Validate the sealed-key commitment without opening the hidden key."""
    _require_exact_keys(
        commitment,
        _ANSWER_KEY_COMMITMENT_KEYS,
        label="DID-v1 answer-key commitment",
    )
    expected_values: dict[str, Any] = {
        "schema_version": "DID-v1",
        "kind": "did_v1_hidden_answer_key_commitment",
        "diagnostic_id": spec.get("diagnostic_id"),
        "case_set_sha256": cases_sha256,
        "record_count": case_count,
        "answer_key_external_to_model_visible_bundle": True,
    }
    for key, expected in expected_values.items():
        if commitment.get(key) != expected or type(commitment.get(key)) is not type(
            expected
        ):
            raise ValueError(f"DID-v1 answer-key commitment differs for {key}")
    answer_key_sha256 = commitment.get("answer_key_sha256")
    if not _is_sha256(answer_key_sha256):
        raise ValueError("DID-v1 answer-key commitment has an invalid key hash")

    if manifest.get("schema_version") != "DID-v1":
        raise ValueError("Case manifest has the wrong DID-v1 schema version")
    if manifest.get("kind") != "did_v1_model_visible_case_manifest":
        raise ValueError("Case manifest has the wrong DID-v1 kind")
    if manifest.get("diagnostic_id") != spec.get("diagnostic_id"):
        raise ValueError("Case manifest has a different diagnostic ID")
    answer_key_entry = manifest.get("answer_key")
    if not isinstance(answer_key_entry, Mapping):
        raise ValueError("Case manifest lacks answer-key metadata")
    _require_exact_keys(
        answer_key_entry,
        _MANIFEST_ANSWER_KEY_KEYS,
        label="DID-v1 case-manifest answer_key",
    )
    expected_answer_key_entry: dict[str, Any] = {
        "sha256": answer_key_sha256,
        "count": case_count,
        "external": True,
        "path_disclosed": False,
    }
    for key, expected in expected_answer_key_entry.items():
        if answer_key_entry.get(key) != expected or type(
            answer_key_entry.get(key)
        ) is not type(expected):
            raise ValueError(f"Case manifest answer_key differs for {key}")

    counts = manifest.get("counts")
    if not isinstance(counts, Mapping) or (
        counts.get("total_prompts") != case_count
        or type(counts.get("total_prompts")) is not int
    ):
        raise ValueError("Case manifest total prompt count differs from commitment")
    return json.loads(canonical_json(dict(commitment)))


def _validate_case_manifest(
    manifest: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
    spec_sha256: str,
    spec_file_sha256: str,
    cases_path: Path,
    cases_sha256: str,
    cases_bytes: int,
    commitment_path: Path,
    commitment_sha256: str,
    commitment_bytes: int,
    commitment: Mapping[str, Any],
    generation_case_ids: Sequence[str],
) -> dict[str, Any]:
    _validate_case_manifest_exact_schema(manifest, spec)
    if manifest.get("diagnostic_spec_sha256") != spec_sha256:
        raise ValueError("Case manifest is not bound to the exact diagnostic spec")
    if manifest.get("diagnostic_spec_file_sha256") != spec_file_sha256:
        raise ValueError("Case manifest is not bound to the exact diagnostic spec file")
    if manifest.get("parents") != spec.get("parents"):
        raise ValueError("Case manifest historical parents differ from the spec")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("Case manifest lacks a files mapping")
    cases_entry = files.get("cases")
    commitment_entry = files.get("answer_key_commitment")
    if not isinstance(cases_entry, Mapping) or not isinstance(commitment_entry, Mapping):
        raise ValueError("Case manifest lacks cases or answer-key commitment metadata")
    _require_exact_keys(
        cases_entry,
        {"path", "sha256", "bytes", "count"},
        label="DID-v1 case-manifest files.cases",
    )
    _require_exact_keys(
        commitment_entry,
        {"path", "sha256", "bytes"},
        label="DID-v1 case-manifest files.answer_key_commitment",
    )
    expected_cases = {
        "sha256": cases_sha256,
        "bytes": cases_bytes,
        "count": FROZEN_EXPECTED_PROMPT_COUNT,
    }
    for key, value in expected_cases.items():
        if cases_entry.get(key) != value or type(cases_entry.get(key)) is not type(value):
            raise ValueError(f"Case manifest differs for files.cases.{key}")
    if cases_entry.get("path") != "cases.jsonl" or cases_path.name != "cases.jsonl":
        raise ValueError("Case manifest names a different cases file")
    for key, value in {
        "sha256": commitment_sha256,
        "bytes": commitment_bytes,
    }.items():
        if commitment_entry.get(key) != value or type(commitment_entry.get(key)) is not type(value):
            raise ValueError(
                f"Case manifest differs for files.answer_key_commitment.{key}"
            )
    if (
        commitment_entry.get("path") != "ANSWER_KEY_COMMITMENT.json"
        or commitment_path.name != "ANSWER_KEY_COMMITMENT.json"
    ):
        raise ValueError("Case manifest names a different answer-key commitment")
    _validate_answer_key_commitment(
        commitment,
        manifest=manifest,
        spec=spec,
        cases_sha256=cases_sha256,
        case_count=FROZEN_EXPECTED_PROMPT_COUNT,
    )
    declared_generation = manifest.get("generation_subset")
    if not isinstance(declared_generation, Mapping):
        raise ValueError("Case manifest lacks the precommitted generation subset")
    expected_size = int(FROZEN_INFERENCE_CONTRACT["generation_subset_size"])
    if len(generation_case_ids) != expected_size or len(set(generation_case_ids)) != expected_size:
        raise ValueError("Core generator returned an invalid generation subset")
    subset_hash = _sha256_json(list(generation_case_ids))
    declared_size = declared_generation.get(
        "size", declared_generation.get("count", declared_generation.get("selected_case_count"))
    )
    if declared_size != expected_size or type(declared_size) is not int:
        raise ValueError("Case manifest generation-subset size differs")
    declared_hash = declared_generation.get(
        "ordered_case_ids_sha256",
        declared_generation.get("case_ids_sha256"),
    )
    if declared_hash != subset_hash:
        raise ValueError("Case manifest generation subset differs from the frozen helper")
    declared_ids = declared_generation.get("case_ids")
    if declared_ids is not None and list(declared_ids) != list(generation_case_ids):
        raise ValueError("Case manifest contains different generation case IDs")
    return json.loads(canonical_json(dict(manifest)))


def _assert_blinded_cases(cases: Sequence[Mapping[str, Any]]) -> None:
    forbidden_keys = {
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
    seen: set[str] = set()
    for index, case in enumerate(cases):
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError(f"Diagnostic case {index} has an invalid or duplicate case_id")
        seen.add(case_id)
        if case.get("split") != "dev" or case.get("namespace") != "AUDIT":
            raise PermissionError(f"Diagnostic case {case_id} is not an AUDIT/DEV case")
        if forbidden_keys & set(case):
            raise ValueError(f"Diagnostic case {case_id} leaks an answer field")
        if not _valid_messages(case.get("messages")):
            raise ValueError(f"Diagnostic case {case_id} has invalid chat messages")
        messages_hash = hashlib.sha256(
            canonical_json(case["messages"]).encode("utf-8")
        ).hexdigest()
        if case.get("messages_sha256") != messages_hash:
            raise ValueError(f"Diagnostic case {case_id} has a bad message hash")


def verify_dev_diagnostic_inputs(
    bridge_config: Mapping[str, Any],
    *,
    spec_path: str | Path,
    case_manifest_path: str | Path,
    cases_path: str | Path,
    answer_key_commitment_path: str | Path,
    data_manifest_path: str | Path,
    dev_data_path: str | Path,
    checkpoint_zero: str | Path,
    genuine_final: str | Path,
    proxy_final: str | Path,
    deployment_root: str | Path | None = None,
    bootstrap_attestation_path: str | Path | None = None,
) -> VerifiedDevDiagnosticInputs:
    """Verify every frozen input without opening bridge state or parsing DEV/TEST."""
    config_path_value = bridge_config.get("_config_path")
    if not isinstance(config_path_value, str) or not config_path_value:
        raise ValueError("bridge_config must come from load_config and retain _config_path")
    config_path = _resolve_without_symlink(config_path_value, label="Bridge config")
    spec_path = _resolve_without_symlink(spec_path, label="DID-v1 spec")
    case_manifest_path = _resolve_without_symlink(
        case_manifest_path, label="DID-v1 case manifest"
    )
    cases_path = _resolve_without_symlink(cases_path, label="DID-v1 cases")
    answer_key_commitment_path = _resolve_without_symlink(
        answer_key_commitment_path, label="DID-v1 answer-key commitment"
    )
    data_manifest_path = _resolve_without_symlink(
        data_manifest_path, label="Historical data manifest"
    )
    dev_data_path = _resolve_without_symlink(
        dev_data_path, label="Historical DEV data"
    )
    checkpoint_roots = {
        "checkpoint_zero": _resolve_without_symlink(
            checkpoint_zero, label="checkpoint_zero"
        ),
        "genuine_final": _resolve_without_symlink(
            genuine_final, label="genuine_final"
        ),
        "proxy_final": _resolve_without_symlink(proxy_final, label="proxy_final"),
    }
    resolved_deployment_root, resolved_bootstrap_attestation = (
        _deployment_context_paths(
            deployment_root=deployment_root,
            bootstrap_attestation_path=bootstrap_attestation_path,
        )
    )
    _require_bundle_payload_paths(
        resolved_deployment_root,
        {
            "bridge_config": config_path,
            "spec": spec_path,
            "case_manifest": case_manifest_path,
            "cases": cases_path,
            "answer_key_commitment": answer_key_commitment_path,
            "data_manifest": data_manifest_path,
            "dev_data": dev_data_path,
            **checkpoint_roots,
        },
    )
    bootstrap_verification = verify_dev_diag_bootstrap_attestation(
        resolved_bootstrap_attestation, resolved_deployment_root
    )
    allowed: list[Path] = [
        config_path,
        spec_path,
        case_manifest_path,
        cases_path,
        answer_key_commitment_path,
        data_manifest_path,
        dev_data_path,
    ]
    for root in checkpoint_roots.values():
        allowed.extend(_checkpoint_paths(root).values())
    ledger = _ScientificInputLedger(allowed)

    raw_config_sha256 = ledger.sha256(config_path)
    if raw_config_sha256 != FROZEN_HISTORICAL_CONTRACT["bridge_config_file_sha256"]:
        raise ValueError("Historical bridge config file hash differs")
    config = json.loads(canonical_json(dict(bridge_config)))
    config["_config_path"] = str(config_path)
    validate_bridge_config({key: value for key, value in config.items() if not key.startswith("_")})
    canonical_config_sha256 = config_hash(config)
    if canonical_config_sha256 != FROZEN_HISTORICAL_CONTRACT[
        "bridge_config_canonical_sha256"
    ]:
        raise ValueError("Historical bridge config canonical hash differs")
    config["_config_sha256"] = canonical_config_sha256

    # Record the spec through the strict ledger, then independently require the
    # core loader to return the same semantic YAML value.
    raw_spec = yaml.safe_load(ledger.read_text(spec_path))
    core_spec = _core_load_spec(spec_path)
    core_public = {
        str(key): value
        for key, value in core_spec.items()
        if not str(key).startswith("_")
    }
    if (
        not isinstance(raw_spec, dict)
        or canonical_json(raw_spec) != canonical_json(core_public)
    ):
        raise ValueError("Core spec loader disagrees with the allowlisted spec bytes")
    spec = _validate_frozen_spec(core_public)
    spec_file_sha256 = ledger.sha256(spec_path)
    spec_sha256 = _sha256_json(spec)
    if core_spec.get("_spec_sha256") not in (None, spec_sha256):
        raise ValueError("Core spec loader reported a different canonical spec hash")
    if core_spec.get("_spec_file_sha256") not in (None, spec_file_sha256):
        raise ValueError("Core spec loader reported a different spec file hash")

    data_manifest_sha256 = ledger.sha256(data_manifest_path)
    if data_manifest_sha256 != FROZEN_HISTORICAL_CONTRACT["data_manifest_sha256"]:
        raise ValueError("Frozen data manifest differs from historical Stage 1")
    data_manifest = ledger.read_json(data_manifest_path)
    if not isinstance(data_manifest, Mapping):
        raise ValueError("Frozen data manifest is malformed")
    dev_entry = dict(dict(data_manifest.get("files") or {}).get("dev") or {})
    dev_sha256 = ledger.sha256(dev_data_path)
    if (
        dev_sha256 != FROZEN_HISTORICAL_CONTRACT["dev_file_sha256"]
        or dev_entry.get("sha256") != dev_sha256
        or int(dev_entry.get("bytes", -1)) != dev_data_path.stat().st_size
        or Path(str(dev_entry.get("path", ""))).name != dev_data_path.name
        or data_manifest.get("config_sha256") != canonical_config_sha256
    ):
        raise ValueError("Frozen DEV bytes do not match the historical data manifest")

    commitment_sha256 = ledger.sha256(answer_key_commitment_path)
    raw_commitment = ledger.read_json(answer_key_commitment_path)
    if not isinstance(raw_commitment, Mapping):
        raise ValueError("Diagnostic answer-key commitment is malformed")
    cases_sha256 = ledger.sha256(cases_path)
    raw_cases = ledger.read_jsonl(cases_path)
    cases = _core_validate_cases(raw_cases, spec)
    if len(cases) != FROZEN_EXPECTED_PROMPT_COUNT:
        raise ValueError("Core validator returned a partial diagnostic grid")
    _assert_blinded_cases(cases)
    generation_case_ids = _core_generation_subset_case_ids(cases, spec)
    case_ids = {str(case["case_id"]) for case in cases}
    if not set(generation_case_ids) <= case_ids:
        raise ValueError("Generation subset contains a case outside DID-v1")
    raw_case_manifest = ledger.read_json(case_manifest_path)
    if not isinstance(raw_case_manifest, Mapping):
        raise ValueError("Diagnostic case manifest is malformed")
    case_manifest = _validate_case_manifest(
        raw_case_manifest,
        spec=spec,
        spec_sha256=spec_sha256,
        spec_file_sha256=spec_file_sha256,
        cases_path=cases_path,
        cases_sha256=cases_sha256,
        cases_bytes=cases_path.stat().st_size,
        commitment_path=answer_key_commitment_path,
        commitment_sha256=commitment_sha256,
        commitment_bytes=answer_key_commitment_path.stat().st_size,
        commitment=raw_commitment,
        generation_case_ids=generation_case_ids,
    )
    source_parent = case_manifest["verified_source_parent"]
    declared_dev_count = dict(data_manifest.get("counts") or {}).get("dev")
    if (
        source_parent["data_manifest_sha256"] != data_manifest_sha256
        or source_parent["dev_file_sha256"] != dev_sha256
        or source_parent["dev_file_bytes"] != dev_data_path.stat().st_size
        or type(declared_dev_count) is not int
        or source_parent["dev_record_count"] != declared_dev_count
    ):
        raise ValueError("DID-v1 case manifest source-parent binding differs")
    commitment = _validate_answer_key_commitment(
        raw_commitment,
        manifest=case_manifest,
        spec=spec,
        cases_sha256=cases_sha256,
        case_count=len(cases),
    )

    checkpoints = {
        policy: _verify_checkpoint(
            policy_condition=policy,
            checkpoint=root,
            expected=FROZEN_CHECKPOINT_CONTRACTS[policy],
            config=config,
            ledger=ledger,
        )
        for policy, root in checkpoint_roots.items()
    }
    reference = checkpoints["checkpoint_zero"].manifest
    shared_fields = (
        "pair_seed",
        "config_sha256",
        "bridge_spec_sha256",
        "environment_provenance",
        "initial_environment_state_sha256",
        "model_runtime_attestation",
        "model_runtime_attestation_sha256",
    )
    for policy, checkpoint in checkpoints.items():
        for field in shared_fields:
            if checkpoint.manifest.get(field) != reference.get(field):
                raise ValueError(f"{policy} violates the shared seed/init/runtime contract")
    if config["bridge"].get("paired_initialization") is not True:
        raise ValueError("Historical bridge config does not require paired initialization")

    # Complete the ledger before model loading.  The checkpoint adapter files are
    # rehashed after PEFT consumes them to detect in-run replacement.
    opened = ledger.attestation()
    bindings = {
        "schema_version": "1.0",
        "diagnostic_spec_path": str(spec_path),
        "diagnostic_spec_sha256": spec_sha256,
        "diagnostic_spec_file_sha256": spec_file_sha256,
        "case_manifest_path": str(case_manifest_path),
        "case_manifest_sha256": ledger.sha256(case_manifest_path),
        "cases_path": str(cases_path),
        "cases_sha256": cases_sha256,
        "answer_key_commitment_path": str(answer_key_commitment_path),
        "answer_key_commitment_sha256": commitment_sha256,
        "answer_key_commitment": commitment,
        "answer_key_sha256": commitment["answer_key_sha256"],
        "bridge_config_path": str(config_path),
        "bridge_config_file_sha256": raw_config_sha256,
        "bridge_config_canonical_sha256": canonical_config_sha256,
        "data_manifest_path": str(data_manifest_path),
        "data_manifest_sha256": data_manifest_sha256,
        "dev_data_path": str(dev_data_path),
        "dev_file_sha256": dev_sha256,
        "case_count": len(cases),
        "generation_subset": {
            "size": len(generation_case_ids),
            "ordered_case_ids_sha256": _sha256_json(generation_case_ids),
            "selected_case_ids_sha256": _sha256_json(sorted(generation_case_ids)),
            "all_case_ids_sha256": _sha256_json(sorted(case_ids)),
            "case_ids": list(generation_case_ids),
        },
        "historical_parents": json.loads(canonical_json(spec["parents"])),
        "bootstrap_verification": json.loads(
            canonical_json(bootstrap_verification)
        ),
        "checkpoints": {
            policy: checkpoint.public_attestation()
            for policy, checkpoint in checkpoints.items()
        },
        "shared_contract": {
            "pair_seed": int(reference["pair_seed"]),
            "paired_initialization": True,
            "initial_environment_state_sha256": reference[
                "initial_environment_state_sha256"
            ],
            "model_runtime_attestation_sha256": reference[
                "model_runtime_attestation_sha256"
            ],
            "model_id": config["model"]["id"],
            "model_revision": config["model"]["revision"],
        },
        "scientific_input_access": opened,
        "bridge_state_opened": False,
        "locked_test_accessed": False,
    }
    return VerifiedDevDiagnosticInputs(
        bridge_config=config,
        spec=spec,
        case_manifest=case_manifest,
        cases=cases,
        generation_case_ids=generation_case_ids,
        checkpoints=checkpoints,
        ledger=ledger,
        bindings=bindings,
        deployment_root=resolved_deployment_root,
        bootstrap_attestation_path=resolved_bootstrap_attestation,
        bootstrap_verification=json.loads(canonical_json(bootstrap_verification)),
    )


def _batches(values: Sequence[Any], size: int) -> list[Sequence[Any]]:
    if size <= 0:
        raise ValueError("Batch size must be positive")
    return [values[start : start + size] for start in range(0, len(values), size)]


def _token_ids_without_truncation(tokenizer: Any, text: str) -> list[int]:
    """Tokenize one exact string while making truncation explicitly impossible."""
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
    )
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise ValueError("DID-v1 tokenizer did not return input_ids")
    raw_ids = encoded["input_ids"]
    if hasattr(raw_ids, "tolist"):
        raw_ids = raw_ids.tolist()
    if (
        isinstance(raw_ids, list)
        and len(raw_ids) == 1
        and isinstance(raw_ids[0], list)
    ):
        raw_ids = raw_ids[0]
    if not isinstance(raw_ids, list) or not raw_ids or any(
        isinstance(value, bool) or not isinstance(value, int) for value in raw_ids
    ):
        raise ValueError("DID-v1 tokenizer returned invalid or empty input_ids")
    return [int(value) for value in raw_ids]


def _strict_prompt_token_counts(
    tokenizer: Any,
    cases: Sequence[Mapping[str, Any]],
    *,
    max_length: int,
    max_new_tokens: int,
    labels: Sequence[str] = ("A", "B"),
) -> tuple[dict[str, int], list[list[Any]]]:
    """Audit prompts and A/B boundaries without ever enabling truncation.

    The prompt is the complete next-token model input.  DID-v1 additionally
    requires the prompt plus its one generated token to fit inside the same
    768-token diagnostic ceiling, so neither scorer nor generator can hide an
    overlength case behind left truncation.
    """
    if max_length <= 0 or max_new_tokens <= 0:
        raise ValueError("DID-v1 token limits must be positive")
    if tuple(labels) != ("A", "B"):
        raise ValueError("DID-v1 token audit requires the frozen A/B choices")
    counts: dict[str, int] = {}
    candidate_rows: list[list[Any]] = []
    for index, case in enumerate(cases):
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in counts:
            raise ValueError(f"DID-v1 token audit found invalid case ID at row {index}")
        messages = case.get("messages")
        if not _valid_messages(messages):
            raise ValueError(f"DID-v1 token audit found invalid messages in {case_id}")
        prompt_text = chat_prompt_text(tokenizer, list(messages))
        prompt_ids = _token_ids_without_truncation(tokenizer, prompt_text)
        prompt_length = len(prompt_ids)
        if prompt_length > max_length:
            raise ValueError(
                f"DID-v1 prompt {case_id} has {prompt_length} tokens and would "
                f"require truncation at max_length={max_length}"
            )
        if prompt_length + max_new_tokens > max_length:
            raise ValueError(
                f"DID-v1 prompt {case_id} plus {max_new_tokens} generated token(s) "
                f"would exceed max_length={max_length}"
            )
        candidate_ids: list[int] = []
        for label in labels:
            full_ids = _token_ids_without_truncation(tokenizer, prompt_text + label)
            if full_ids[:prompt_length] != prompt_ids:
                raise ValueError(
                    f"DID-v1 tokenizer changed the prompt boundary for {case_id}/{label}"
                )
            completion = full_ids[prompt_length:]
            if len(completion) != 1:
                raise ValueError(
                    f"DID-v1 choice {label!r} is not one token in {case_id}"
                )
            if len(full_ids) > max_length:
                raise ValueError(
                    f"DID-v1 prompt-plus-choice for {case_id}/{label} would exceed "
                    f"max_length={max_length}"
                )
            candidate_ids.append(int(completion[0]))
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError(f"DID-v1 A/B choices collide in {case_id}")
        counts[case_id] = prompt_length
        candidate_rows.append([case_id, *candidate_ids])
    if not counts:
        raise ValueError("DID-v1 token audit received no cases")
    return counts, candidate_rows


def _nearest_rank(values: Sequence[int], fraction: float) -> int:
    if not values or not 0.0 < fraction <= 1.0:
        raise ValueError("Invalid nearest-rank input")
    ordered = sorted(int(value) for value in values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _token_length_stats(
    case_ids: Sequence[str], counts: Mapping[str, int], *, max_new_tokens: int
) -> dict[str, Any]:
    lengths = [int(counts[case_id]) for case_id in case_ids]
    ordered_pairs = [[case_id, int(counts[case_id])] for case_id in case_ids]
    return {
        "count": len(lengths),
        "minimum_prompt_tokens": min(lengths),
        "maximum_prompt_tokens": max(lengths),
        "sum_prompt_tokens": sum(lengths),
        "mean_prompt_tokens_numerator": sum(lengths),
        "mean_prompt_tokens_denominator": len(lengths),
        "p50_prompt_tokens_nearest_rank": _nearest_rank(lengths, 0.50),
        "p95_prompt_tokens_nearest_rank": _nearest_rank(lengths, 0.95),
        "p99_prompt_tokens_nearest_rank": _nearest_rank(lengths, 0.99),
        "maximum_total_tokens_after_generation": max(lengths) + max_new_tokens,
        "ordered_case_token_counts_sha256": _sha256_json(ordered_pairs),
    }


def _ordered_generation_token_count_rows(
    prompt_token_counts_by_case: Mapping[str, Any],
    committed_case_ids: Sequence[Any],
    *,
    expected_size: int,
) -> list[list[Any]]:
    """Reindex selected token counts by the precommitted subset order."""

    case_ids = list(committed_case_ids)
    if (
        len(case_ids) != expected_size
        or len(case_ids) != len(set(case_ids))
        or any(not isinstance(case_id, str) or not case_id for case_id in case_ids)
    ):
        raise ValueError("DID-v1 committed generation subset is malformed")
    if set(prompt_token_counts_by_case) != set(case_ids):
        raise ValueError("DID-v1 prediction generation subset differs from commitment")
    rows: list[list[Any]] = []
    for case_id in case_ids:
        count = prompt_token_counts_by_case[case_id]
        if type(count) is not int or count <= 0:
            raise ValueError("DID-v1 selected prompt token count is malformed")
        rows.append([case_id, count])
    return rows


def audit_dev_diag_token_lengths(
    tokenizer: Any,
    cases: Sequence[Mapping[str, Any]],
    generation_case_ids: Sequence[str],
    spec: Mapping[str, Any],
) -> TokenLengthAudit:
    """Create the full-corpus no-truncation proof before loading a policy."""
    model = spec.get("model")
    inference = spec.get("inference_contract")
    if not isinstance(model, Mapping) or not isinstance(inference, Mapping):
        raise ValueError("DID-v1 tokenizer audit requires model/inference contracts")
    audit_contract = inference.get("token_length_audit")
    if not isinstance(audit_contract, Mapping):
        raise ValueError("DID-v1 tokenizer-audit contract is missing")
    if (
        audit_contract.get("required_before_model_load") is not True
        or audit_contract.get("truncation_allowed") is not False
        or audit_contract.get("require_choice_boundary_single_token") is not True
        or audit_contract.get("require_prompt_plus_generation_within_max_length")
        is not True
    ):
        raise ValueError("DID-v1 tokenizer-audit safeguards are not frozen")
    max_length = int(model.get("max_length", -1))
    max_new_tokens = int(inference.get("max_new_tokens", -1))
    expected_count = int(audit_contract.get("expected_prompt_count", -1))
    expected_maximum = int(audit_contract.get("expected_max_prompt_tokens", -1))
    ordered_cases = list(cases)
    if len(ordered_cases) != expected_count:
        raise ValueError(
            f"DID-v1 tokenizer audit received {len(ordered_cases)} prompts; "
            f"expected {expected_count}"
        )
    all_ids = [str(case.get("case_id", "")) for case in ordered_cases]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("DID-v1 tokenizer audit requires unique case IDs")
    generation_ids = [str(case_id) for case_id in generation_case_ids]
    if (
        len(generation_ids) != int(inference.get("generation_subset_size", -1))
        or len(generation_ids) != len(set(generation_ids))
        or not set(generation_ids) <= set(all_ids)
    ):
        raise ValueError("DID-v1 tokenizer audit received the wrong generation subset")
    counts, candidate_rows = _strict_prompt_token_counts(
        tokenizer,
        ordered_cases,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        labels=list(model.get("choice_labels") or []),
    )
    observed_maximum = max(counts.values())
    if audit_contract.get("require_exact_max") is not True:
        raise ValueError("DID-v1 must require its exact tokenizer maximum")
    if observed_maximum != expected_maximum:
        raise ValueError(
            f"DID-v1 tokenizer maximum is {observed_maximum}; expected exact "
            f"frozen maximum {expected_maximum}"
        )
    maximum_case_ids = sorted(
        case_id for case_id, length in counts.items() if length == observed_maximum
    )
    all_stats = _token_length_stats(all_ids, counts, max_new_tokens=max_new_tokens)
    generation_stats = _token_length_stats(
        generation_ids, counts, max_new_tokens=max_new_tokens
    )
    if (
        all_stats["ordered_case_token_counts_sha256"]
        != audit_contract.get("expected_all_prompt_token_counts_sha256")
        or generation_stats["ordered_case_token_counts_sha256"]
        != audit_contract.get("expected_generation_subset_token_counts_sha256")
    ):
        raise ValueError(
            "DID-v1 tokenizer length vector differs from the frozen full corpus"
        )
    chat_template_sha256 = hashlib.sha256(
        str(getattr(tokenizer, "chat_template", "")).encode("utf-8")
    ).hexdigest()
    candidate_token_ids_sha256 = _sha256_json(candidate_rows)
    if chat_template_sha256 != audit_contract.get(
        "expected_chat_template_sha256"
    ):
        raise ValueError("DID-v1 tokenizer chat template differs from the frozen hash")
    if candidate_token_ids_sha256 != audit_contract.get(
        "expected_ordered_case_candidate_token_ids_sha256"
    ):
        raise ValueError(
            "DID-v1 ordered A/B candidate token IDs differ from the frozen hash"
        )
    report = {
        "schema_version": "1.0",
        "kind": "did_v1_no_truncation_tokenizer_audit",
        "diagnostic_id": str(spec.get("diagnostic_id", "")),
        "model_id": str(model.get("id", "")),
        "model_revision": str(model.get("revision", "")),
        "tokenizer_class": type(tokenizer).__name__,
        "chat_template_sha256": chat_template_sha256,
        "max_length": max_length,
        "max_new_tokens": max_new_tokens,
        "truncation_allowed": False,
        "expected_max_prompt_tokens": expected_maximum,
        "observed_max_prompt_tokens": observed_maximum,
        "maximum_prompt_case_ids": maximum_case_ids,
        "all_prompts": all_stats,
        "generation_subset": generation_stats,
        "candidate_boundary": {
            "labels": list(model.get("choice_labels") or []),
            "single_token_for_every_prompt": True,
            "distinct_for_every_prompt": True,
            "ordered_case_candidate_token_ids_sha256": candidate_token_ids_sha256,
        },
        "checks": {
            "full_prompt_grid_audited": len(counts) == expected_count,
            "generation_subset_audited": len(generation_ids)
            == int(inference["generation_subset_size"]),
            "exact_maximum_matches": observed_maximum == expected_maximum,
            "all_prompt_counts_hash_matches": all_stats[
                "ordered_case_token_counts_sha256"
            ]
            == audit_contract["expected_all_prompt_token_counts_sha256"],
            "generation_subset_counts_hash_matches": generation_stats[
                "ordered_case_token_counts_sha256"
            ]
            == audit_contract[
                "expected_generation_subset_token_counts_sha256"
            ],
            "chat_template_hash_matches": chat_template_sha256
            == audit_contract["expected_chat_template_sha256"],
            "ordered_candidate_token_ids_hash_matches": candidate_token_ids_sha256
            == audit_contract[
                "expected_ordered_case_candidate_token_ids_sha256"
            ],
            "every_prompt_within_max_length": observed_maximum <= max_length,
            "every_prompt_plus_generation_within_max_length": observed_maximum
            + max_new_tokens
            <= max_length,
            "truncation_disabled": True,
        },
    }
    if not all(report["checks"].values()):
        raise RuntimeError("DID-v1 no-truncation tokenizer audit did not pass")
    digest = _sha256_json(report)
    return TokenLengthAudit(
        report=json.loads(canonical_json(report)),
        report_sha256=digest,
        prompt_token_counts=dict(counts),
    )


def _validate_choice_score_record(
    row: Mapping[str, Any], *, label: str
) -> dict[str, float]:
    """Independently validate raw and derived A/B probability diagnostics."""
    names = (
        "logp_A",
        "logp_B",
        "probability_A",
        "probability_B",
        "log_legal_choice_mass",
        "legal_choice_mass",
    )
    values: dict[str, float] = {}
    for name in names:
        raw = row.get(name)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"{label} has a missing/non-numeric {name}")
        numeric = float(raw)
        if not math.isfinite(numeric):
            raise ValueError(f"{label} has a non-finite {name}")
        values[name] = numeric

    tolerance = float(LEGAL_CHOICE_LOG_MASS_TOLERANCE)
    if values["logp_A"] > tolerance or values["logp_B"] > tolerance:
        raise ValueError(f"{label} has a raw token log probability above zero tolerance")
    if values["log_legal_choice_mass"] > tolerance:
        raise ValueError(f"{label} has legal-choice log mass above zero tolerance")
    if not 0.0 <= values["legal_choice_mass"] <= math.exp(tolerance):
        raise ValueError(f"{label} has legal-choice mass outside tolerated [0, 1]")
    if not 0.0 <= values["probability_A"] <= 1.0 or not 0.0 <= values[
        "probability_B"
    ] <= 1.0:
        raise ValueError(f"{label} has a normalized probability outside [0, 1]")

    expected = legal_choice_diagnostics(values["logp_A"], values["logp_B"])
    for name in (
        "probability_A",
        "probability_B",
        "log_legal_choice_mass",
        "legal_choice_mass",
    ):
        if not math.isclose(
            values[name],
            float(expected[name]),
            rel_tol=tolerance,
            abs_tol=tolerance,
        ):
            raise ValueError(f"{label} has inconsistent derived field {name}")
    return values


def _score_model(
    model: Any,
    tokenizer: Any,
    cases: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    max_length: int,
    expected_prompt_token_counts: Mapping[str, int] | None = None,
) -> dict[str, dict[str, float]]:
    import torch

    observed_counts, _ = _strict_prompt_token_counts(
        tokenizer,
        cases,
        max_length=max_length,
        max_new_tokens=1,
    )
    if expected_prompt_token_counts is not None and observed_counts != {
        str(case["case_id"]): int(expected_prompt_token_counts[str(case["case_id"])])
        for case in cases
    }:
        raise RuntimeError("DID-v1 scorer tokenizer lengths differ from pre-inference audit")
    scores: dict[str, dict[str, float]] = {}
    for batch in _batches(list(cases), batch_size):
        with torch.inference_mode():
            _, raw = differentiable_choice_log_probs(
                model,
                tokenizer,
                batch,
                max_length=max_length,
            )
        for case, values in zip(batch, raw.detach().float().cpu(), strict=True):
            logp_a, logp_b = map(float, values.tolist())
            diagnostic = legal_choice_diagnostics(logp_a, logp_b)
            score = {
                "logp_A": logp_a,
                "logp_B": logp_b,
                **diagnostic,
            }
            _validate_choice_score_record(
                score,
                label=f"DID-v1 scorer output for {case['case_id']}",
            )
            scores[str(case["case_id"])] = score
    if set(scores) != {str(case["case_id"]) for case in cases}:
        raise RuntimeError("A/B scorer did not return every diagnostic case")
    return scores


def _generate_model(
    model: Any,
    tokenizer: Any,
    cases: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
    expected_prompt_token_counts: Mapping[str, int] | None = None,
) -> dict[str, dict[str, Any]]:
    import torch

    if max_new_tokens != 1:
        raise ValueError("DID-v1 exact generation is frozen to one first token")
    observed_counts, _ = _strict_prompt_token_counts(
        tokenizer,
        cases,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
    )
    if expected_prompt_token_counts is not None and observed_counts != {
        str(case["case_id"]): int(expected_prompt_token_counts[str(case["case_id"])])
        for case in cases
    }:
        raise RuntimeError("DID-v1 generator tokenizer lengths differ from pre-inference audit")
    generated: dict[str, dict[str, Any]] = {}
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
                truncation=False,
                return_tensors="pt",
            )
            encoded_lengths = encoded["attention_mask"].sum(dim=1).tolist()
            expected_lengths = [observed_counts[str(case["case_id"])] for case in batch]
            if [int(value) for value in encoded_lengths] != expected_lengths:
                raise RuntimeError(
                    "DID-v1 generator encoding differs from its no-truncation audit"
                )
            device = next(model.parameters()).device
            encoded = {key: value.to(device) for key, value in encoded.items()}
            input_width = int(encoded["input_ids"].shape[1])
            with torch.inference_mode():
                output = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=1,
                    pad_token_id=int(tokenizer.pad_token_id),
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            completions = tokenizer.batch_decode(
                output[:, input_width:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for case, completion in zip(batch, completions, strict=True):
                parsed, status = parse_unconstrained_choice(completion)
                generated[str(case["case_id"])] = {
                    "generated_output": completion,
                    "parsed_action": parsed,
                    "parse_status": status,
                }
    finally:
        tokenizer.padding_side = original_padding_side
        tokenizer.truncation_side = original_truncation_side
    if set(generated) != {str(case["case_id"]) for case in cases}:
        raise RuntimeError("Exact generation did not return every precommitted case")
    return generated


def _freeze_and_validate_model(model: Any, *, require_already_frozen: bool) -> None:
    model.eval()
    parameters = list(model.parameters())
    if not parameters:
        raise ValueError("Loaded policy has no parameters")
    if require_already_frozen and any(parameter.requires_grad for parameter in parameters):
        raise RuntimeError(
            "PEFT adapter was not attached inference-only with is_trainable=False"
        )
    for parameter in parameters:
        parameter.requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in parameters):
        raise RuntimeError("DID-v1 policy was not frozen for inference")


def _seed_inference(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed & 0xFFFF_FFFF)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _policy_artifact(
    policy_condition: str,
    verified: VerifiedDevDiagnosticInputs,
) -> tuple[VerifiedCheckpoint, bool]:
    if policy_condition == "unchanged_base":
        return verified.checkpoints["checkpoint_zero"], True
    return verified.checkpoints[policy_condition], False


def _evaluate_policy(
    policy_condition: str,
    verified: VerifiedDevDiagnosticInputs,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    started = time.monotonic()
    config = verified.bridge_config
    inference = verified.spec["inference_contract"]
    token_audit = verified.token_length_audit
    if token_audit is None:
        raise RuntimeError("DID-v1 policy inference lacks its pre-inference token audit")
    max_length = int(verified.spec["model"]["max_length"])
    checkpoint, base_policy = _policy_artifact(policy_condition, verified)
    seed = int(verified.spec["generation"]["seed"])
    _seed_inference(seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    tokenizer = None
    model = None
    try:
        tokenizer_started = time.monotonic()
        tokenizer = load_tokenizer(config)
        tokenizer_seconds = time.monotonic() - tokenizer_started
        model_started = time.monotonic()
        model = (
            load_base_model(config, training=False)
            if base_policy
            else load_adapter_model(config, checkpoint.path)
        )
        model_load_seconds = time.monotonic() - model_started
        _freeze_and_validate_model(model, require_already_frozen=not base_policy)
        runtime_started = time.monotonic()
        if base_policy:
            runtime_contract = validate_loaded_base_runtime(
                config,
                model,
                tokenizer,
                checkpoint.runtime_attestation,
            )
        else:
            runtime_contract = validate_loaded_lora_runtime(
                config,
                model,
                tokenizer,
                getattr(model, "_ue_lora_target_inventory", None),
                checkpoint.runtime_attestation,
            )
        runtime_seconds = time.monotonic() - runtime_started
        _freeze_and_validate_model(model, require_already_frozen=True)

        scoring_started = time.monotonic()
        scores = _score_model(
            model,
            tokenizer,
            verified.cases,
            batch_size=int(inference["batch_size"]),
            max_length=max_length,
            expected_prompt_token_counts=token_audit.prompt_token_counts,
        )
        scoring_seconds = time.monotonic() - scoring_started
        by_id = {str(case["case_id"]): case for case in verified.cases}
        generation_cases = [by_id[case_id] for case_id in verified.generation_case_ids]
        generation_started = time.monotonic()
        generated = _generate_model(
            model,
            tokenizer,
            generation_cases,
            batch_size=int(inference["generation_batch_size"]),
            max_length=max_length,
            max_new_tokens=int(inference["max_new_tokens"]),
            expected_prompt_token_counts=token_audit.prompt_token_counts,
        )
        generation_seconds = time.monotonic() - generation_started

        # PEFT has consumed these files.  Detect replacement between verification
        # and load; bridge_state.pt is intentionally neither opened nor hashed.
        if not base_policy:
            paths = _checkpoint_paths(checkpoint.path)
            verified.ledger.sha256(paths["adapter_config"], refresh=True)
            verified.ledger.sha256(paths["adapter_model"], refresh=True)
            verified.ledger.sha256(paths["manifest"], refresh=True)

        generation_ids = set(verified.generation_case_ids)
        binding = {
            "diagnostic_spec_sha256": verified.bindings["diagnostic_spec_sha256"],
            "diagnostic_spec_file_sha256": verified.bindings[
                "diagnostic_spec_file_sha256"
            ],
            "case_manifest_sha256": verified.bindings["case_manifest_sha256"],
            "cases_sha256": verified.bindings["cases_sha256"],
            "answer_key_commitment_sha256": verified.bindings[
                "answer_key_commitment_sha256"
            ],
            "answer_key_sha256": verified.bindings["answer_key_sha256"],
            "policy_checkpoint_manifest_sha256": checkpoint.checkpoint_manifest_sha256,
            "model_runtime_attestation_sha256": checkpoint.runtime_attestation[
                "attestation_sha256"
            ],
            "token_length_audit_sha256": token_audit.report_sha256,
            "bootstrap_verification": json.loads(
                canonical_json(verified.bootstrap_verification)
            ),
        }
        rows: list[dict[str, Any]] = []
        for case in verified.cases:
            case_id = str(case["case_id"])
            score = scores[case_id]
            predicted = "A" if float(score["probability_A"]) >= 0.5 else "B"
            generation = generated.get(
                case_id,
                {
                    "generated_output": None,
                    "parsed_action": None,
                    "parse_status": "not_sampled",
                },
            )
            rows.append(
                {
                    "schema_version": "1.0",
                    "evidence_kind": POSTHOC_EVIDENCE_KIND,
                    "scientific_status": POSTHOC_STATUS,
                    "stage1_evidence": False,
                    "can_change_stage1_decision": False,
                    "locked_test_accessed": False,
                    "policy_condition": policy_condition,
                    "case_id": case_id,
                    "messages_sha256": case["messages_sha256"],
                    "prompt_token_count": token_audit.prompt_token_counts[case_id],
                    "prompt_plus_max_new_tokens": token_audit.prompt_token_counts[
                        case_id
                    ]
                    + int(inference["max_new_tokens"]),
                    "inference_max_length": max_length,
                    "truncation_applied": False,
                    "input_bindings": binding,
                    "case_metadata": {
                        key: value for key, value in case.items() if key != "messages"
                    },
                    "predicted_action": predicted,
                    "tie_break_rule": "probability_A_greater_than_or_equal_to_0.5",
                    **score,
                    "generation_subset_selected": case_id in generation_ids,
                    **generation,
                }
            )
        sampled = [row for row in rows if row["generation_subset_selected"]]
        unsampled = [row for row in rows if not row["generation_subset_selected"]]
        if (
            len(sampled) != int(inference["generation_subset_size"])
            or any(row["parse_status"] == "not_sampled" for row in sampled)
            or any(
                row["parse_status"] != "not_sampled"
                or row["generated_output"] is not None
                or row["parsed_action"] is not None
                for row in unsampled
            )
        ):
            raise RuntimeError("Generation outputs do not match the precommitted subset")
        legal_masses = [float(row["legal_choice_mass"]) for row in rows]
        probabilities = [float(row["probability_A"]) for row in rows]
        parse_rate = float(np.mean([row["parse_status"] == "exact" for row in sampled]))
        peak_allocated = (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        )
        peak_reserved = (
            int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0
        )
        summary = {
            "schema_version": "1.0",
            "kind": "did_v1_policy_inference_summary",
            "evidence_kind": POSTHOC_EVIDENCE_KIND,
            "scientific_status": POSTHOC_STATUS,
            "stage1_evidence": False,
            "can_change_stage1_decision": False,
            "can_authorize_locked_test": False,
            "policy_condition": policy_condition,
            "fresh_base_model_load": True,
            "adapter_loaded": not base_policy,
            "inference_only": True,
            "parameters_trainable": False,
            "optimizer_constructed": False,
            "reward_feedback": False,
            "record_count": len(rows),
            "generation_sample_count": len(sampled),
            "generation_subset": verified.bindings["generation_subset"],
            "token_length_audit": token_audit.binding(),
            "exact_parse_rate": parse_rate,
            "mean_legal_choice_mass": float(np.mean(legal_masses)),
            "minimum_legal_choice_mass": min(legal_masses),
            "mean_probability_A": float(np.mean(probabilities)),
            "predicted_action_counts": {
                label: sum(row["predicted_action"] == label for row in rows)
                for label in ("A", "B")
            },
            "output_integrity_checks": {
                "record_count_complete": len(rows) == FROZEN_EXPECTED_PROMPT_COUNT,
                "generation_subset_complete": len(sampled)
                == int(inference["generation_subset_size"]),
                "minimum_legal_choice_mass": min(legal_masses)
                >= float(verified.spec["analysis"]["minimum_legal_choice_mass"]),
                "unconstrained_exact_parse": parse_rate
                >= float(verified.spec["analysis"]["minimum_exact_parse_rate"]),
                "full_corpus_token_audit": token_audit.report["checks"][
                    "full_prompt_grid_audited"
                ],
                "no_prompt_truncated": token_audit.report["checks"][
                    "truncation_disabled"
                ]
                and token_audit.report["checks"][
                    "every_prompt_plus_generation_within_max_length"
                ],
                "exact_frozen_max_prompt_tokens": token_audit.report[
                    "observed_max_prompt_tokens"
                ]
                == int(
                    inference["token_length_audit"]["expected_max_prompt_tokens"]
                ),
            },
            "policy_artifact": {
                **checkpoint.public_attestation(),
                "adapter_loaded": not base_policy,
                "unchanged_base_anchor_only": base_policy,
            },
            "input_bindings": binding,
            "model_runtime_contract": runtime_contract,
            "timing": {
                "tokenizer_load_wall_seconds": tokenizer_seconds,
                "fresh_model_load_wall_seconds": model_load_seconds,
                "runtime_validation_wall_seconds": runtime_seconds,
                "forced_scoring_wall_seconds": scoring_seconds,
                "generation_wall_seconds": generation_seconds,
                "total_wall_seconds": time.monotonic() - started,
                "peak_vram_allocated_bytes": peak_allocated,
                "peak_vram_reserved_bytes": peak_reserved,
            },
        }
        return rows, summary
    finally:
        model = None
        tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _atomic_policy_output(
    destination_dir: Path,
    policy_condition: str,
    rows: Sequence[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    final = destination_dir / policy_condition
    if final.exists():
        raise FileExistsError(f"Refusing to overwrite DID-v1 policy output: {final}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{policy_condition}.", dir=destination_dir)
    )
    predictions_relative = PurePosixPath(
        policy_condition, "predictions.jsonl"
    ).as_posix()
    summary_relative = PurePosixPath(policy_condition, "summary.json").as_posix()
    try:
        predictions = temporary / "predictions.jsonl"
        summary_path = temporary / "summary.json"
        write_jsonl(predictions, rows)
        predictions_sha256 = _sha256_path(predictions)
        completed = dict(summary)
        completed["created_at_utc"] = _utc_now()
        completed["predictions_path"] = predictions_relative
        completed["predictions_sha256"] = predictions_sha256
        write_json(summary_path, completed)
        summary_sha256 = _sha256_path(summary_path)
        os.replace(temporary, final)
        return {
            "policy_condition": policy_condition,
            "predictions_path": predictions_relative,
            "predictions_sha256": predictions_sha256,
            "summary_path": summary_relative,
            "summary_sha256": summary_sha256,
            "record_count": len(rows),
        }
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _module_sha256() -> str:
    return _sha256_path(__file__)


def _portable_run_file(run_dir: Path, value: Any, *, expected: str) -> Path:
    """Resolve one exact POSIX run-relative path without accepting traversal."""
    if not isinstance(value, str) or value != expected or "\\" in value:
        raise ValueError(f"Completed DID-v1 run has a nonportable path: {value!r}")
    relative = PurePosixPath(value)
    if relative.is_absolute() or not relative.parts or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError(f"Completed DID-v1 run has an unsafe relative path: {value!r}")
    lexical = run_dir.joinpath(*relative.parts)
    resolved = lexical.resolve()
    try:
        resolved.relative_to(run_dir)
    except ValueError as exc:
        raise ValueError(f"Completed DID-v1 path escapes its run: {value!r}") from exc
    if lexical.is_symlink() or any(
        (run_dir.joinpath(*relative.parts[:index])).is_symlink()
        for index in range(1, len(relative.parts))
    ):
        raise ValueError(f"Completed DID-v1 path traverses a symlink: {value!r}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Completed DID-v1 output is missing: {resolved}")
    return resolved


def _read_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON mapping: {path}")
    return value


def _verify_completed_public_input_artifacts(
    bindings: Mapping[str, Any],
    *,
    spec_path: str | Path,
    case_manifest_path: str | Path,
    cases_path: str | Path,
    answer_key_commitment_path: str | Path,
) -> dict[str, Any]:
    """Reopen public artifacts and bind them to one completed inference run."""
    supplied = {
        "spec": Path(spec_path),
        "case_manifest": Path(case_manifest_path),
        "cases": Path(cases_path),
        "answer_key_commitment": Path(answer_key_commitment_path),
    }
    resolved = {
        key: _resolve_without_symlink(path, label=f"Analysis {key}")
        for key, path in supplied.items()
    }
    ledger = _ScientificInputLedger(list(supplied.values()))

    raw_spec = yaml.safe_load(ledger.read_text(resolved["spec"]))
    core_spec = _core_load_spec(resolved["spec"])
    core_public = {
        str(key): value
        for key, value in core_spec.items()
        if not str(key).startswith("_")
    }
    if (
        not isinstance(raw_spec, dict)
        or canonical_json(raw_spec) != canonical_json(core_public)
    ):
        raise ValueError("Analysis spec loader disagrees with supplied spec bytes")
    spec = _validate_frozen_spec(core_public)
    spec_file_sha256 = ledger.sha256(resolved["spec"])
    spec_sha256 = _sha256_json(spec)
    if core_spec.get("_spec_sha256") not in (None, spec_sha256):
        raise ValueError("Analysis spec loader reported a different semantic hash")
    if core_spec.get("_spec_file_sha256") not in (None, spec_file_sha256):
        raise ValueError("Analysis spec loader reported a different file hash")

    cases_sha256 = ledger.sha256(resolved["cases"])
    cases = _core_validate_cases(ledger.read_jsonl(resolved["cases"]), spec)
    if len(cases) != FROZEN_EXPECTED_PROMPT_COUNT:
        raise ValueError("Analysis cases do not contain the complete DID-v1 grid")
    _assert_blinded_cases(cases)
    generation_case_ids = _core_generation_subset_case_ids(cases, spec)

    commitment_sha256 = ledger.sha256(resolved["answer_key_commitment"])
    commitment_value = ledger.read_json(resolved["answer_key_commitment"])
    if not isinstance(commitment_value, Mapping):
        raise ValueError("Analysis answer-key commitment is malformed")
    manifest_sha256 = ledger.sha256(resolved["case_manifest"])
    manifest_value = ledger.read_json(resolved["case_manifest"])
    if not isinstance(manifest_value, Mapping):
        raise ValueError("Analysis case manifest is malformed")
    manifest = _validate_case_manifest(
        manifest_value,
        spec=spec,
        spec_sha256=spec_sha256,
        spec_file_sha256=spec_file_sha256,
        cases_path=resolved["cases"],
        cases_sha256=cases_sha256,
        cases_bytes=resolved["cases"].stat().st_size,
        commitment_path=resolved["answer_key_commitment"],
        commitment_sha256=commitment_sha256,
        commitment_bytes=resolved["answer_key_commitment"].stat().st_size,
        commitment=commitment_value,
        generation_case_ids=generation_case_ids,
    )
    commitment = _validate_answer_key_commitment(
        commitment_value,
        manifest=manifest,
        spec=spec,
        cases_sha256=cases_sha256,
        case_count=len(cases),
    )

    observed_bindings: dict[str, Any] = {
        "diagnostic_spec_file_sha256": spec_file_sha256,
        "diagnostic_spec_sha256": spec_sha256,
        "case_manifest_sha256": manifest_sha256,
        "cases_sha256": cases_sha256,
        "answer_key_commitment_sha256": commitment_sha256,
        "answer_key_sha256": commitment["answer_key_sha256"],
    }
    for key, observed in observed_bindings.items():
        if bindings.get(key) != observed or type(bindings.get(key)) is not type(observed):
            raise ValueError(f"Supplied analysis artifact differs from run binding {key}")
    if bindings.get("answer_key_commitment") != commitment:
        raise ValueError("Run input binding contains a different answer-key commitment")
    if bindings.get("historical_parents") != spec.get("parents"):
        raise ValueError("Run input binding contains different historical parents")
    if bindings.get("case_count") != len(cases) or type(bindings.get("case_count")) is not int:
        raise ValueError("Run input binding contains a different case count")
    generation_subset = {
        "size": len(generation_case_ids),
        "ordered_case_ids_sha256": _sha256_json(generation_case_ids),
        "selected_case_ids_sha256": _sha256_json(sorted(generation_case_ids)),
        "all_case_ids_sha256": _sha256_json(
            sorted(str(case["case_id"]) for case in cases)
        ),
        "case_ids": list(generation_case_ids),
    }
    if bindings.get("generation_subset") != generation_subset:
        raise ValueError("Run input binding contains a different generation subset")
    for path in resolved.values():
        ledger.sha256(path, refresh=True)

    verification: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": "did_v1_completed_run_public_input_verification",
        "status": "PASS",
        "diagnostic_id": spec["diagnostic_id"],
        "artifact_paths": {key: str(path) for key, path in resolved.items()},
        "artifact_sha256": observed_bindings,
        "answer_key_commitment": commitment,
        "historical_parents": json.loads(canonical_json(spec["parents"])),
        "case_count": len(cases),
        "generation_subset": generation_subset,
        "checks": {
            "diagnostic_spec_file_sha256_matches_run_binding": True,
            "diagnostic_spec_semantic_sha256_matches_run_binding": True,
            "case_manifest_sha256_matches_run_binding": True,
            "cases_sha256_matches_run_binding": True,
            "answer_key_commitment_sha256_matches_run_binding": True,
            "answer_key_sha256_matches_run_binding": True,
            "commitment_exact_schema_and_relations_valid": True,
            "all_historical_parent_identities_match_frozen_evaluator_constants": True,
            "complete_blinded_case_grid_valid": True,
            "generation_subset_order_matches_public_commitment": True,
        },
        "scientific_input_access": ledger.attestation(),
        "answer_key_revealed": False,
        "locked_test_accessed": False,
    }
    verification["verification_sha256"] = _sha256_json(verification)
    return verification


def verify_dev_diagnostic_analysis_inputs(
    completed_run: VerifiedCompletedDevDiagnosticRun,
    *,
    answer_key_path: str | Path,
) -> dict[str, Any]:
    """Verify the separately held answer key only at analysis/reveal time."""
    if not isinstance(completed_run, VerifiedCompletedDevDiagnosticRun):
        raise TypeError("completed_run must come from verify_completed_dev_diagnostic_run")
    path = _resolve_without_symlink(answer_key_path, label="DID-v1 revealed answer key")
    if path.name.lower() in {"test.jsonl", "locked_test.jsonl", "test.json"}:
        raise PermissionError("DID-v1 analysis answer key may not be a locked TEST artifact")
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"DID-v1 revealed answer key is missing: {path}")
    observed_sha256 = _sha256_path(path)
    public = completed_run.input_artifact_verification
    commitment = public.get("answer_key_commitment")
    if not isinstance(commitment, Mapping):
        raise ValueError("Completed-run verification lacks the answer-key commitment")
    expected_sha256 = commitment.get("answer_key_sha256")
    if observed_sha256 != expected_sha256:
        raise ValueError("Revealed answer key does not match the committed hash")
    record_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid DID-v1 answer-key JSONL at line {line_number}"
                ) from exc
            if not isinstance(row, Mapping):
                raise ValueError(
                    f"DID-v1 answer-key line {line_number} is not a mapping"
                )
            record_count += 1
    if record_count != commitment.get("record_count"):
        raise ValueError("Revealed answer-key record count differs from commitment")
    if _sha256_path(path) != observed_sha256:
        raise ValueError("Revealed answer key changed during analysis verification")
    verification: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": "did_v1_revealed_answer_key_verification",
        "status": "PASS",
        "path": str(path),
        "sha256": observed_sha256,
        "record_count": record_count,
        "checks": {
            "answer_key_sha256_matches_commitment": True,
            "answer_key_record_count_matches_commitment": True,
            "answer_key_revealed_only_for_analysis": True,
            "locked_test_accessed": False,
        },
    }
    verification["verification_sha256"] = _sha256_json(verification)
    return verification


def verify_completed_dev_diagnostic_run(
    run_dir: str | Path,
    *,
    spec_path: str | Path,
    case_manifest_path: str | Path,
    cases_path: str | Path,
    answer_key_commitment_path: str | Path,
    deployment_root: str | Path | None = None,
    bootstrap_attestation_path: str | Path | None = None,
) -> VerifiedCompletedDevDiagnosticRun:
    """Verify and resolve a movable COMPLETE DID-v1 inference run.

    Prediction paths are absolute paths under the supplied ``run_dir`` in
    frozen policy order.  The required public input paths are rehashed and
    semantically checked against the run before any result is accepted.  The
    private answer key is intentionally not needed here.
    """
    root = _resolve_without_symlink(run_dir, label="Completed DID-v1 run directory")
    if not root.is_dir():
        raise FileNotFoundError(f"Completed DID-v1 run directory is missing: {root}")
    manifest_path = root / "run_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError(f"Completed DID-v1 run manifest is missing: {manifest_path}")
    resolved_deployment_root, resolved_bootstrap_attestation = (
        _deployment_context_paths(
            deployment_root=deployment_root,
            bootstrap_attestation_path=bootstrap_attestation_path,
        )
    )
    _require_bundle_payload_paths(
        resolved_deployment_root,
        {
            "spec": spec_path,
            "case_manifest": case_manifest_path,
            "cases": cases_path,
            "answer_key_commitment": answer_key_commitment_path,
        },
    )
    bootstrap_verification = verify_dev_diag_bootstrap_attestation(
        resolved_bootstrap_attestation, resolved_deployment_root
    )
    initial_manifest_sha256 = _sha256_path(manifest_path)
    manifest = _read_json_mapping(manifest_path, label="DID-v1 run manifest")
    expected_manifest_values = {
        "kind": "did_v1_posthoc_inference_run",
        "diagnostic_id": "stage1_dev_diag_v1",
        "evidence_kind": POSTHOC_EVIDENCE_KIND,
        "scientific_status": POSTHOC_STATUS,
        "stage1_evidence": False,
        "can_change_stage1_decision": False,
        "can_authorize_locked_test": False,
        "locked_test_accessed": False,
        "state": "COMPLETE",
        "error": None,
        "complete_four_policy_grid": True,
        "output_count": len(POLICY_CONDITIONS),
    }
    for key, expected in expected_manifest_values.items():
        if manifest.get(key) != expected or type(manifest.get(key)) is not type(expected):
            raise ValueError(f"Completed DID-v1 run manifest differs for {key}")
    if manifest.get("policy_conditions") != list(POLICY_CONDITIONS):
        raise ValueError("Completed DID-v1 run has a different policy order")
    inference = manifest.get("inference_contract")
    if not isinstance(inference, Mapping) or dict(inference) != FROZEN_INFERENCE_CONTRACT:
        raise ValueError("Completed DID-v1 run has a different inference contract")
    token_audit_binding = manifest.get("token_length_audit")
    if not isinstance(token_audit_binding, Mapping):
        raise ValueError("Completed DID-v1 run lacks its no-truncation audit")
    token_audit_report = token_audit_binding.get("report")
    token_audit_sha256 = token_audit_binding.get("sha256")
    audit_contract = inference["token_length_audit"]
    expected_audit_checks = {
        "full_prompt_grid_audited",
        "generation_subset_audited",
        "exact_maximum_matches",
        "all_prompt_counts_hash_matches",
        "generation_subset_counts_hash_matches",
        "chat_template_hash_matches",
        "ordered_candidate_token_ids_hash_matches",
        "every_prompt_within_max_length",
        "every_prompt_plus_generation_within_max_length",
        "truncation_disabled",
    }
    audit_checks = (
        token_audit_report.get("checks")
        if isinstance(token_audit_report, Mapping)
        else None
    )
    all_prompt_stats = (
        token_audit_report.get("all_prompts")
        if isinstance(token_audit_report, Mapping)
        else None
    )
    generation_stats = (
        token_audit_report.get("generation_subset")
        if isinstance(token_audit_report, Mapping)
        else None
    )
    boundary = (
        token_audit_report.get("candidate_boundary")
        if isinstance(token_audit_report, Mapping)
        else None
    )
    if (
        not isinstance(token_audit_report, Mapping)
        or not isinstance(token_audit_sha256, str)
        or _sha256_json(token_audit_report) != token_audit_sha256
        or token_audit_report.get("kind")
        != "did_v1_no_truncation_tokenizer_audit"
        or token_audit_report.get("diagnostic_id") != "stage1_dev_diag_v1"
        or token_audit_report.get("model_id")
        != FROZEN_HISTORICAL_CONTRACT["model_id"]
        or token_audit_report.get("model_revision")
        != FROZEN_HISTORICAL_CONTRACT["model_revision"]
        or token_audit_report.get("chat_template_sha256")
        != audit_contract["expected_chat_template_sha256"]
        or token_audit_report.get("expected_max_prompt_tokens")
        != audit_contract["expected_max_prompt_tokens"]
        or token_audit_report.get("observed_max_prompt_tokens")
        != audit_contract["expected_max_prompt_tokens"]
        or token_audit_report.get("max_length") != 768
        or token_audit_report.get("max_new_tokens") != inference["max_new_tokens"]
        or token_audit_report.get("truncation_allowed") is not False
        or not isinstance(all_prompt_stats, Mapping)
        or all_prompt_stats.get("count") != FROZEN_EXPECTED_PROMPT_COUNT
        or all_prompt_stats.get("maximum_prompt_tokens")
        != audit_contract["expected_max_prompt_tokens"]
        or all_prompt_stats.get("ordered_case_token_counts_sha256")
        != audit_contract["expected_all_prompt_token_counts_sha256"]
        or not isinstance(generation_stats, Mapping)
        or generation_stats.get("count") != inference["generation_subset_size"]
        or generation_stats.get("ordered_case_token_counts_sha256")
        != audit_contract["expected_generation_subset_token_counts_sha256"]
        or not isinstance(boundary, Mapping)
        or boundary.get("labels") != ["A", "B"]
        or boundary.get("single_token_for_every_prompt") is not True
        or boundary.get("distinct_for_every_prompt") is not True
        or boundary.get("ordered_case_candidate_token_ids_sha256")
        != audit_contract[
            "expected_ordered_case_candidate_token_ids_sha256"
        ]
        or not isinstance(audit_checks, Mapping)
        or set(audit_checks) != expected_audit_checks
        or not all(value is True for value in audit_checks.values())
    ):
        raise ValueError("Completed DID-v1 run has an invalid no-truncation audit")
    bindings = manifest.get("input_bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("Completed DID-v1 run lacks input bindings")
    if (
        int(bindings.get("case_count", -1)) != FROZEN_EXPECTED_PROMPT_COUNT
        or bindings.get("bridge_state_opened") is not False
        or bindings.get("locked_test_accessed") is not False
        or bindings.get("token_length_audit") != token_audit_binding
        or bindings.get("token_length_audit_sha256") != token_audit_sha256
        or bindings.get("bootstrap_verification") != bootstrap_verification
    ):
        raise ValueError("Completed DID-v1 run has invalid case/access bindings")
    if (
        manifest.get("bootstrap_preflight_verification") != bootstrap_verification
        or manifest.get("bootstrap_postflight_verification")
        != bootstrap_verification
    ):
        raise ValueError(
            "Completed DID-v1 run lacks exact pre/post bootstrap verification"
        )
    input_artifact_verification = _verify_completed_public_input_artifacts(
        bindings,
        spec_path=spec_path,
        case_manifest_path=case_manifest_path,
        cases_path=cases_path,
        answer_key_commitment_path=answer_key_commitment_path,
    )
    generation_binding = input_artifact_verification.get("generation_subset")
    if not isinstance(generation_binding, Mapping):
        raise ValueError("Completed DID-v1 run lacks a verified generation subset")
    committed_generation_case_ids = generation_binding.get("case_ids")
    if not isinstance(committed_generation_case_ids, list):
        raise ValueError("Completed DID-v1 generation-subset order is malformed")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != len(POLICY_CONDITIONS):
        raise ValueError("Completed DID-v1 run lacks all four policy outputs")

    expected_topology = {"run_manifest.json", *POLICY_CONDITIONS}
    if {path.name for path in root.iterdir()} != expected_topology:
        raise ValueError("Completed DID-v1 run contains an unexpected top-level entry")

    prediction_paths: list[Path] = []
    completed_output_hashes: list[tuple[Path, str]] = []
    reference_case_ids: list[str] | None = None
    reference_generation_ids: list[str] | None = None
    for policy_condition, output_value in zip(POLICY_CONDITIONS, outputs, strict=True):
        if not isinstance(output_value, Mapping):
            raise ValueError(f"Completed output for {policy_condition} is malformed")
        output = dict(output_value)
        expected_output_keys = {
            "policy_condition",
            "predictions_path",
            "predictions_sha256",
            "summary_path",
            "summary_sha256",
            "record_count",
        }
        if set(output) != expected_output_keys:
            raise ValueError(f"Completed output for {policy_condition} has wrong fields")
        if output["policy_condition"] != policy_condition:
            raise ValueError("Completed DID-v1 output policy order differs")
        if (
            type(output["record_count"]) is not int
            or output["record_count"] != FROZEN_EXPECTED_PROMPT_COUNT
        ):
            raise ValueError(f"Completed output count differs for {policy_condition}")
        predictions_relative = PurePosixPath(
            policy_condition, "predictions.jsonl"
        ).as_posix()
        summary_relative = PurePosixPath(policy_condition, "summary.json").as_posix()
        predictions_path = _portable_run_file(
            root,
            output["predictions_path"],
            expected=predictions_relative,
        )
        summary_path = _portable_run_file(
            root,
            output["summary_path"],
            expected=summary_relative,
        )
        policy_directory = root / policy_condition
        if not policy_directory.is_dir() or {
            path.name for path in policy_directory.iterdir()
        } != {"predictions.jsonl", "summary.json"}:
            raise ValueError(f"Completed output topology differs for {policy_condition}")
        predictions_sha256 = _sha256_path(predictions_path)
        summary_sha256 = _sha256_path(summary_path)
        if output["predictions_sha256"] != predictions_sha256:
            raise ValueError(f"Prediction hash mismatch for {policy_condition}")
        if output["summary_sha256"] != summary_sha256:
            raise ValueError(f"Summary hash mismatch for {policy_condition}")
        completed_output_hashes.extend(
            ((predictions_path, predictions_sha256), (summary_path, summary_sha256))
        )

        summary = _read_json_mapping(
            summary_path,
            label=f"DID-v1 {policy_condition} summary",
        )
        expected_summary_values = {
            "kind": "did_v1_policy_inference_summary",
            "evidence_kind": POSTHOC_EVIDENCE_KIND,
            "scientific_status": POSTHOC_STATUS,
            "stage1_evidence": False,
            "can_change_stage1_decision": False,
            "can_authorize_locked_test": False,
            "policy_condition": policy_condition,
            "fresh_base_model_load": True,
            "inference_only": True,
            "parameters_trainable": False,
            "optimizer_constructed": False,
            "reward_feedback": False,
            "record_count": FROZEN_EXPECTED_PROMPT_COUNT,
            "generation_sample_count": int(inference["generation_subset_size"]),
            "predictions_path": predictions_relative,
            "predictions_sha256": predictions_sha256,
        }
        for key, expected in expected_summary_values.items():
            if summary.get(key) != expected or type(summary.get(key)) is not type(expected):
                raise ValueError(f"Completed {policy_condition} summary differs for {key}")
        checks = summary.get("output_integrity_checks")
        if not isinstance(checks, Mapping) or (
            checks.get("record_count_complete") is not True
            or checks.get("generation_subset_complete") is not True
            or checks.get("full_corpus_token_audit") is not True
            or checks.get("no_prompt_truncated") is not True
            or checks.get("exact_frozen_max_prompt_tokens") is not True
        ):
            raise ValueError(f"Completed {policy_condition} summary is structurally incomplete")
        summary_binding = summary.get("input_bindings")
        if not isinstance(summary_binding, Mapping):
            raise ValueError(f"Completed {policy_condition} summary lacks input bindings")
        for key in (
            "diagnostic_spec_sha256",
            "diagnostic_spec_file_sha256",
            "case_manifest_sha256",
            "cases_sha256",
            "answer_key_commitment_sha256",
            "answer_key_sha256",
            "token_length_audit_sha256",
        ):
            if summary_binding.get(key) != bindings.get(key):
                raise ValueError(
                    f"Completed {policy_condition} summary differs from run binding {key}"
                )
        if summary_binding.get("bootstrap_verification") != bootstrap_verification:
            raise ValueError(
                f"Completed {policy_condition} summary has wrong bootstrap binding"
            )
        artifact_policy = (
            "checkpoint_zero" if policy_condition == "unchanged_base" else policy_condition
        )
        if summary.get("generation_subset") != bindings.get("generation_subset"):
            raise ValueError(
                f"Completed {policy_condition} summary has a different generation subset"
            )
        if summary.get("token_length_audit") != token_audit_binding:
            raise ValueError(
                f"Completed {policy_condition} summary has a different token audit"
            )
        checkpoint_bindings = bindings.get("checkpoints")
        if not isinstance(checkpoint_bindings, Mapping) or not isinstance(
            checkpoint_bindings.get(artifact_policy), Mapping
        ):
            raise ValueError(f"Completed {policy_condition} lacks checkpoint binding")
        expected_checkpoint_sha = checkpoint_bindings[artifact_policy].get(
            "checkpoint_manifest_sha256"
        )
        if summary_binding.get("policy_checkpoint_manifest_sha256") != expected_checkpoint_sha:
            raise ValueError(f"Completed {policy_condition} summary has wrong checkpoint")
        expected_runtime_sha = checkpoint_bindings[artifact_policy].get(
            "model_runtime_attestation_sha256"
        )
        if summary_binding.get("model_runtime_attestation_sha256") != expected_runtime_sha:
            raise ValueError(f"Completed {policy_condition} summary has wrong runtime")
        policy_artifact = summary.get("policy_artifact")
        if not isinstance(policy_artifact, Mapping) or (
            policy_artifact.get("checkpoint_manifest_sha256") != expected_checkpoint_sha
            or policy_artifact.get("model_runtime_attestation_sha256")
            != expected_runtime_sha
        ):
            raise ValueError(f"Completed {policy_condition} summary has wrong policy artifact")

        ordered_case_ids: list[str] = []
        generation_case_ids: list[str] = []
        ordered_case_token_counts: list[list[Any]] = []
        generation_case_token_counts: dict[str, int] = {}
        seen: set[str] = set()
        with predictions_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid {policy_condition} prediction JSONL at line {line_number}"
                    ) from exc
                if not isinstance(row, Mapping):
                    raise ValueError(
                        f"{policy_condition} prediction line {line_number} is not a mapping"
                    )
                case_id = row.get("case_id")
                if not isinstance(case_id, str) or not case_id or case_id in seen:
                    raise ValueError(f"{policy_condition} has duplicate/invalid case IDs")
                seen.add(case_id)
                ordered_case_ids.append(case_id)
                expected_row_values = {
                    "evidence_kind": POSTHOC_EVIDENCE_KIND,
                    "scientific_status": POSTHOC_STATUS,
                    "stage1_evidence": False,
                    "can_change_stage1_decision": False,
                    "locked_test_accessed": False,
                    "policy_condition": policy_condition,
                }
                for key, expected in expected_row_values.items():
                    if row.get(key) != expected or type(row.get(key)) is not type(expected):
                        raise ValueError(
                            f"{policy_condition} prediction {case_id} differs for {key}"
                        )
                if row.get("input_bindings") != summary_binding:
                    raise ValueError(
                        f"{policy_condition} prediction {case_id} has different bindings"
                    )
                score = _validate_choice_score_record(
                    row,
                    label=f"{policy_condition} prediction {case_id}",
                )
                expected_action = "A" if score["probability_A"] >= 0.5 else "B"
                if (
                    row.get("predicted_action") != expected_action
                    or row.get("tie_break_rule")
                    != "probability_A_greater_than_or_equal_to_0.5"
                ):
                    raise ValueError(
                        f"{policy_condition} prediction {case_id} has an invalid action"
                    )
                prompt_token_count = row.get("prompt_token_count")
                if (
                    type(prompt_token_count) is not int
                    or prompt_token_count <= 0
                    or prompt_token_count > 745
                    or row.get("prompt_plus_max_new_tokens")
                    != prompt_token_count + int(inference["max_new_tokens"])
                    or row.get("inference_max_length") != 768
                    or row.get("truncation_applied") is not False
                ):
                    raise ValueError(
                        f"{policy_condition} prediction {case_id} has invalid token provenance"
                    )
                ordered_case_token_counts.append([case_id, prompt_token_count])
                selected = row.get("generation_subset_selected")
                if type(selected) is not bool:
                    raise ValueError(
                        f"{policy_condition} prediction {case_id} has invalid subset flag"
                    )
                if selected:
                    generation_case_ids.append(case_id)
                    generation_case_token_counts[case_id] = prompt_token_count
        if len(ordered_case_ids) != FROZEN_EXPECTED_PROMPT_COUNT:
            raise ValueError(f"Prediction record count mismatch for {policy_condition}")
        if (
            _sha256_json(ordered_case_token_counts)
            != token_audit_report["all_prompts"][
                "ordered_case_token_counts_sha256"
            ]
            or max(value for _, value in ordered_case_token_counts) != 745
        ):
            raise ValueError(
                f"Prediction token-length binding mismatch for {policy_condition}"
            )
        ordered_generation_token_counts = _ordered_generation_token_count_rows(
            generation_case_token_counts,
            committed_generation_case_ids,
            expected_size=int(inference["generation_subset_size"]),
        )
        if (
            _sha256_json(ordered_generation_token_counts)
            != token_audit_report["generation_subset"][
                "ordered_case_token_counts_sha256"
            ]
        ):
            raise ValueError(
                f"Generation token-length binding mismatch for {policy_condition}"
            )
        if (
            _sha256_json(sorted(ordered_case_ids))
            != generation_binding.get("all_case_ids_sha256")
            or _sha256_json(sorted(generation_case_ids))
            != generation_binding.get("selected_case_ids_sha256")
        ):
            raise ValueError(f"Prediction case-set binding mismatch for {policy_condition}")
        if reference_case_ids is None:
            reference_case_ids = ordered_case_ids
            reference_generation_ids = generation_case_ids
        elif (
            ordered_case_ids != reference_case_ids
            or generation_case_ids != reference_generation_ids
        ):
            raise ValueError("Completed DID-v1 policies do not cover the same ordered cases")
        prediction_paths.append(predictions_path)
    for path, expected_sha256 in completed_output_hashes:
        if _sha256_path(path) != expected_sha256:
            raise ValueError(
                f"Completed DID-v1 output changed during verification: {path}"
            )
    final_bootstrap_verification = verify_dev_diag_bootstrap_attestation(
        resolved_bootstrap_attestation, resolved_deployment_root
    )
    if final_bootstrap_verification != bootstrap_verification:
        raise ValueError(
            "DID-v1 bundle/bootstrap/source changed during completed-run verification"
        )
    final_manifest_sha256 = _sha256_path(manifest_path)
    if final_manifest_sha256 != initial_manifest_sha256:
        raise ValueError("Completed DID-v1 run manifest changed during verification")
    return VerifiedCompletedDevDiagnosticRun(
        run_dir=root,
        run_manifest_path=manifest_path,
        run_manifest_sha256=final_manifest_sha256,
        prediction_paths=tuple(prediction_paths),
        input_artifact_verification=input_artifact_verification,
        deployment_root=resolved_deployment_root,
        bootstrap_attestation_path=resolved_bootstrap_attestation,
        bootstrap_verification=json.loads(canonical_json(bootstrap_verification)),
    )


def _stable_file_snapshot(path: str | Path, *, label: str) -> tuple[Path, bytes, str]:
    resolved = _resolve_without_symlink(path, label=label)
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    before = resolved.stat()
    payload = resolved.read_bytes()
    after = resolved.stat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after or len(payload) != after.st_size:
        raise ValueError(f"{label} changed while it was snapshotted: {resolved}")
    return resolved, payload, hashlib.sha256(payload).hexdigest()


def _json_mapping_from_snapshot(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} snapshot is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} snapshot must be a JSON mapping")
    return value


def _jsonl_from_snapshot(payload: bytes, *, label: str) -> list[dict[str, Any]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} snapshot is not UTF-8") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{label} snapshot has invalid JSONL at line {line_number}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(
                f"{label} snapshot line {line_number} is not a mapping"
            )
        rows.append(row)
    return rows


def _completed_run_identity(
    completed: VerifiedCompletedDevDiagnosticRun,
) -> dict[str, Any]:
    return {
        "run_dir": str(completed.run_dir),
        "run_manifest_path": str(completed.run_manifest_path),
        "run_manifest_sha256": completed.run_manifest_sha256,
        "prediction_paths": [str(path) for path in completed.prediction_paths],
        "input_artifact_verification": completed.input_artifact_verification,
        "deployment_root": str(completed.deployment_root),
        "bootstrap_attestation_path": str(completed.bootstrap_attestation_path),
        "bootstrap_verification": completed.bootstrap_verification,
    }


def finalize_verified_dev_diagnostic_analysis(
    *,
    run_dir: str | Path,
    deployment_root: str | Path,
    bootstrap_attestation_path: str | Path,
    spec_path: str | Path,
    case_manifest_path: str | Path,
    cases_path: str | Path,
    answer_key_commitment_path: str | Path,
    answer_key_path: str | Path,
    destination: str | Path,
) -> Path:
    """Formally analyze one run after repeated path-derived verification.

    This is the only API allowed to convert the core analyzer's explicitly
    unverified localization outcome into a licensing decision.  It accepts no
    caller-constructed verifier object: all evidence is rederived from paths,
    analyzed from stable byte snapshots, and reverified before the report is
    atomically published.
    """

    target = Path(destination).resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite DID-v1 analysis: {target}")

    verifier_kwargs = {
        "spec_path": spec_path,
        "case_manifest_path": case_manifest_path,
        "cases_path": cases_path,
        "answer_key_commitment_path": answer_key_commitment_path,
        "deployment_root": deployment_root,
        "bootstrap_attestation_path": bootstrap_attestation_path,
    }
    first = verify_completed_dev_diagnostic_run(run_dir, **verifier_kwargs)
    first_key = verify_dev_diagnostic_analysis_inputs(
        first, answer_key_path=answer_key_path
    )

    snapshot_sources: dict[str, Path] = {
        "run_manifest": first.run_manifest_path,
        "spec": Path(spec_path),
        "case_manifest": Path(case_manifest_path),
        "cases": Path(cases_path),
        "answer_key_commitment": Path(answer_key_commitment_path),
        "answer_key": Path(answer_key_path),
    }
    for policy, path in zip(POLICY_CONDITIONS, first.prediction_paths, strict=True):
        snapshot_sources[f"predictions/{policy}"] = path
    snapshots: dict[str, tuple[Path, bytes, str]] = {
        name: _stable_file_snapshot(path, label=f"DID-v1 formal {name}")
        for name, path in snapshot_sources.items()
    }
    if snapshots["run_manifest"][2] != first.run_manifest_sha256:
        raise ValueError("DID-v1 run manifest changed before formal analysis")
    committed = first.input_artifact_verification["artifact_sha256"]
    expected_public_hashes = {
        "spec": committed["diagnostic_spec_file_sha256"],
        "case_manifest": committed["case_manifest_sha256"],
        "cases": committed["cases_sha256"],
        "answer_key_commitment": committed["answer_key_commitment_sha256"],
        "answer_key": first_key["sha256"],
    }
    for name, expected_sha256 in expected_public_hashes.items():
        if snapshots[name][2] != expected_sha256:
            raise ValueError(f"DID-v1 formal snapshot differs for {name}")

    try:
        spec_value = yaml.safe_load(snapshots["spec"][1].decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError("DID-v1 formal spec snapshot is malformed") from exc
    if not isinstance(spec_value, dict):
        raise ValueError("DID-v1 formal spec snapshot is not a mapping")
    validated_spec = _validate_frozen_spec(spec_value)
    if _sha256_json(validated_spec) != committed["diagnostic_spec_sha256"]:
        raise ValueError("DID-v1 formal spec semantic hash differs")
    case_manifest = _json_mapping_from_snapshot(
        snapshots["case_manifest"][1], label="DID-v1 formal case manifest"
    )
    commitment = _json_mapping_from_snapshot(
        snapshots["answer_key_commitment"][1],
        label="DID-v1 formal answer-key commitment",
    )
    cases = _jsonl_from_snapshot(
        snapshots["cases"][1], label="DID-v1 formal cases"
    )
    answer_key = _jsonl_from_snapshot(
        snapshots["answer_key"][1], label="DID-v1 formal answer key"
    )
    predictions: list[dict[str, Any]] = []
    for policy in POLICY_CONDITIONS:
        predictions.extend(
            _jsonl_from_snapshot(
                snapshots[f"predictions/{policy}"][1],
                label=f"DID-v1 formal {policy} predictions",
            )
        )

    # Recheck the exact public schemas against the byte snapshots before the
    # core computation.  The core separately regenerates every case and answer.
    _validate_case_manifest(
        case_manifest,
        spec=validated_spec,
        spec_sha256=committed["diagnostic_spec_sha256"],
        spec_file_sha256=committed["diagnostic_spec_file_sha256"],
        cases_path=snapshots["cases"][0],
        cases_sha256=snapshots["cases"][2],
        cases_bytes=len(snapshots["cases"][1]),
        commitment_path=snapshots["answer_key_commitment"][0],
        commitment_sha256=snapshots["answer_key_commitment"][2],
        commitment_bytes=len(snapshots["answer_key_commitment"][1]),
        commitment=commitment,
        generation_case_ids=_core_generation_subset_case_ids(cases, validated_spec),
    )
    _validate_answer_key_commitment(
        commitment,
        manifest=case_manifest,
        spec=validated_spec,
        cases_sha256=snapshots["cases"][2],
        case_count=len(cases),
    )

    from .dev_diag import analyze_dev_diag_predictions

    report = analyze_dev_diag_predictions(
        validated_spec,
        cases,
        predictions,
        answer_key=answer_key,
        case_manifest=case_manifest,
    )
    localization_outcome = report.get("localization_outcome")
    if (
        not isinstance(localization_outcome, str)
        or report.get("decision")
        != f"UNVERIFIED_DIRECT_API_{localization_outcome}"
        or report.get("verification_status") != "unverified_direct_api"
        or dict(report.get("interpretation_contract") or {}).get(
            "can_license_e1b"
        )
        is not False
    ):
        raise RuntimeError("DID-v1 core analyzer violated the direct-API boundary")

    # Never trust or accept a caller-constructed dataclass.  Rederive the whole
    # completed-run verification from immutable paths after the computation.
    second = verify_completed_dev_diagnostic_run(run_dir, **verifier_kwargs)
    second_key = verify_dev_diagnostic_analysis_inputs(
        second, answer_key_path=answer_key_path
    )
    if canonical_json(_completed_run_identity(first)) != canonical_json(
        _completed_run_identity(second)
    ):
        raise ValueError("DID-v1 verified run identity changed during analysis")
    if canonical_json(first_key) != canonical_json(second_key):
        raise ValueError("DID-v1 answer-key verification changed during analysis")
    for name, (path, _payload, digest) in snapshots.items():
        if _sha256_path(path) != digest:
            raise ValueError(f"DID-v1 formal input changed during analysis: {name}")

    all_gates_pass = report.get("all_gates_pass") is True
    if (localization_outcome == _FORMAL_LICENSE_OUTCOME) is not all_gates_pass:
        raise RuntimeError(
            "DID-v1 localization outcome is inconsistent with the all-gates result"
        )
    can_license = bool(
        all_gates_pass and localization_outcome == _FORMAL_LICENSE_OUTCOME
    )
    report["decision"] = localization_outcome
    report["verification_status"] = "verified_completed_run"
    interpretation = dict(report["interpretation_contract"])
    interpretation["can_license_e1b"] = can_license
    interpretation["verified_inference_run"] = True
    interpretation.pop("conditional_outcome_after_verified_finalize", None)
    report["interpretation_contract"] = interpretation
    report["inference_run"] = {
        "run_manifest_sha256": second.run_manifest_sha256,
        "prediction_files": [
            {
                "policy_condition": policy,
                "path": path.relative_to(second.run_dir).as_posix(),
                "sha256": snapshots[f"predictions/{policy}"][2],
            }
            for policy, path in zip(
                POLICY_CONDITIONS, second.prediction_paths, strict=True
            )
        ],
        "input_artifact_verification": second.input_artifact_verification,
        "answer_key_verification": second_key,
        "bootstrap_verification": second.bootstrap_verification,
        "analysis_inputs_fail_closed": True,
        "verified_complete_posthoc_run": True,
        "independent_completed_run_verification_passes": 2,
        "caller_constructed_verifier_objects_accepted": False,
    }
    report["formal_analysis_provenance"] = {
        "schema_version": "1.0",
        "kind": "did_v1_verified_formal_analysis_provenance",
        "status": "PASS",
        "snapshot_sha256": {
            name: digest for name, (_path, _payload, digest) in snapshots.items()
        },
        "checks": {
            "analysis_used_stable_byte_snapshots": True,
            "public_inputs_match_run_bindings": True,
            "answer_key_matches_pre_inference_commitment": True,
            "completed_run_rederived_before_and_after_analysis": True,
            "bootstrap_bundle_source_reverified_before_and_after_analysis": True,
            "all_snapshots_unchanged_after_analysis": True,
            "direct_api_could_not_license": True,
        },
        "locked_test_accessed": False,
    }

    # One last byte check immediately before publishing.  The report is first
    # fully materialized off-path, then linked into place without overwrite.
    for name, (path, _payload, digest) in snapshots.items():
        if _sha256_path(path) != digest:
            raise ValueError(f"DID-v1 formal input changed before write: {name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        payload = (
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if json.loads(temporary.read_text(encoding="utf-8")) != report:
            raise RuntimeError("DID-v1 formal report write verification failed")
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Refusing to overwrite DID-v1 analysis: {target}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return target


def evaluate_dev_diagnostic(
    bridge_config: Mapping[str, Any],
    *,
    spec_path: str | Path,
    case_manifest_path: str | Path,
    cases_path: str | Path,
    answer_key_commitment_path: str | Path,
    data_manifest_path: str | Path,
    dev_data_path: str | Path,
    checkpoint_zero: str | Path,
    genuine_final: str | Path,
    proxy_final: str | Path,
    destination_dir: str | Path,
    deployment_root: str | Path | None = None,
    bootstrap_attestation_path: str | Path | None = None,
) -> Path:
    """Run the exact frozen DID-v1 four-policy inference contract.

    There are intentionally no batch-size, subset, checkpoint, split, or model
    overrides.  Those values come only from ``stage1_dev_diag_v1.yaml``.
    """
    started_at = _utc_now()
    started = time.monotonic()
    destination = Path(destination_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite DID-v1 destination: {destination}")

    # Verify before making an output directory or loading a model.  A provenance
    # failure therefore leaves no result that could be mistaken for a partial run.
    verified = verify_dev_diagnostic_inputs(
        bridge_config,
        spec_path=spec_path,
        case_manifest_path=case_manifest_path,
        cases_path=cases_path,
        answer_key_commitment_path=answer_key_commitment_path,
        data_manifest_path=data_manifest_path,
        dev_data_path=dev_data_path,
        checkpoint_zero=checkpoint_zero,
        genuine_final=genuine_final,
        proxy_final=proxy_final,
        deployment_root=deployment_root,
        bootstrap_attestation_path=bootstrap_attestation_path,
    )

    # Tokenize the complete 19,200-prompt grid before any policy model is
    # loaded.  This is deliberately a separate tokenizer load: a failure leaves
    # no output directory and cannot be confused with partial inference.
    token_audit_started = time.monotonic()
    audit_tokenizer = load_tokenizer(verified.bridge_config)
    try:
        token_audit = audit_dev_diag_token_lengths(
            audit_tokenizer,
            verified.cases,
            verified.generation_case_ids,
            verified.spec,
        )
    finally:
        audit_tokenizer = None
        gc.collect()
    token_audit_seconds = time.monotonic() - token_audit_started
    verified.token_length_audit = token_audit
    verified.bindings["token_length_audit"] = token_audit.binding()
    verified.bindings["token_length_audit_sha256"] = token_audit.report_sha256

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=False)
    project_root = Path(verified.bridge_config["_config_path"]).parent.parent.resolve()
    run_manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": "did_v1_posthoc_inference_run",
        "diagnostic_id": verified.spec["diagnostic_id"],
        "evidence_kind": POSTHOC_EVIDENCE_KIND,
        "scientific_status": POSTHOC_STATUS,
        "stage1_evidence": False,
        "can_change_stage1_decision": False,
        "can_authorize_locked_test": False,
        "locked_test_accessed": False,
        "state": "RUNNING",
        "started_at_utc": started_at,
        "ended_at_utc": None,
        "command_line": list(sys.argv),
        "policy_conditions": list(POLICY_CONDITIONS),
        "inference_contract": verified.spec["inference_contract"],
        "token_length_audit": token_audit.binding(),
        "token_length_audit_wall_seconds": token_audit_seconds,
        "input_bindings": verified.bindings,
        "bootstrap_preflight_verification": verified.bootstrap_verification,
        "source": {
            "project_tree_sha256": project_hash(project_root),
            "evaluator_module_sha256": _module_sha256(),
        },
        "runtime_environment": environment_snapshot(),
        "outputs": [],
        "wall_seconds": None,
        "error": None,
    }
    manifest_path = destination / "run_manifest.json"
    write_json(manifest_path, run_manifest)
    try:
        for policy_condition in POLICY_CONDITIONS:
            rows, summary = _evaluate_policy(policy_condition, verified)
            output = _atomic_policy_output(
                destination,
                policy_condition,
                rows,
                summary,
            )
            run_manifest["outputs"].append(output)
            # Persist progress after every expensive fresh model load.  Completed
            # policy directories remain valid forensic artifacts if a later load
            # fails, while a new destination is still required for any rerun.
            write_json(manifest_path, run_manifest)
            del rows, summary
            gc.collect()
        bootstrap_postflight = verify_dev_diag_bootstrap_attestation(
            verified.bootstrap_attestation_path, verified.deployment_root
        )
        if canonical_json(bootstrap_postflight) != canonical_json(
            verified.bootstrap_verification
        ):
            raise RuntimeError(
                "DID-v1 bundle/bootstrap/source binding changed during inference"
            )
        run_manifest["bootstrap_postflight_verification"] = bootstrap_postflight
        run_manifest["state"] = "COMPLETE"
        run_manifest["ended_at_utc"] = _utc_now()
        run_manifest["wall_seconds"] = time.monotonic() - started
        run_manifest["output_count"] = len(run_manifest["outputs"])
        run_manifest["complete_four_policy_grid"] = (
            [row["policy_condition"] for row in run_manifest["outputs"]]
            == list(POLICY_CONDITIONS)
            and all(
                int(row["record_count"]) == FROZEN_EXPECTED_PROMPT_COUNT
                for row in run_manifest["outputs"]
            )
        )
        write_json(manifest_path, run_manifest)
    except BaseException as exc:
        run_manifest["state"] = "FAILED"
        run_manifest["ended_at_utc"] = _utc_now()
        run_manifest["wall_seconds"] = time.monotonic() - started
        run_manifest["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        write_json(manifest_path, run_manifest)
        raise
    return destination
