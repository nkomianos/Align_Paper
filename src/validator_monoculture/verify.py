"""Offline, fail-closed reconstruction of validator-monoculture G0 evidence.

The GPU runner records raw model completions as phase-separated evidence.  This
module deliberately does not trust any runner-side parse, classification, or
test result: it checks every phase manifest, replays every parser, reclassifies
every patch with the private oracle, re-evaluates generated vectors in the
restricted sandbox, reconstructs the exact crossed analysis table, and only
then calls :func:`validator_monoculture.analysis.evaluate_gate`.

No file below ``evidence_root`` is opened for writing.  The final report must be
written to a new path outside the entire retrieved run root containing it.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import io
import shutil
import subprocess
import sys
import tempfile
import tokenize
from typing import Any

import yaml

from under_extinction.io import read_jsonl, sha256_file, write_json

from .analysis import GateThresholds, evaluate_gate
from .prompts import (
    parse_patch_completion,
    parse_test_completion,
    patch_prompt,
    verifier_prompt,
)
from .runtime import deterministic_seed
from .schema import PrivateOracle, PublicTask, TestVector, canonical_json_bytes
from .serde import (
    bind_corpus,
    deserialize_private_oracles,
    deserialize_public_tasks,
)


FAMILIES = ("qwen3_5", "gemma4")
PROMPT_MODES = ("spec_only", "patch_aware")
PATCH_SCHEMA = "validator-monoculture-patch-completion-v1"
TEST_SCHEMA = "validator-monoculture-test-completion-v1"
ELIGIBLE_SCHEMA = "validator-monoculture-eligible-patch-v1"
PATCH_PHASE_KIND = "validator_monoculture_patch_collection"
CLASSIFICATION_PHASE_KIND = "validator_monoculture_patch_classification"
TEST_PHASE_KIND = "validator_monoculture_test_collection"
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")


PatchClassifier = Callable[..., Mapping[str, Any]]
VectorEvaluator = Callable[..., Mapping[str, Any]]
OracleValidator = Callable[..., Mapping[str, Any]]


_PATCH_FIELDS = {
    "schema_version",
    "task_id",
    "cwe_id",
    "split",
    "patch_family",
    "model_id",
    "model_revision",
    "sample_index",
    "seed",
    "prompt_sha256",
    "completion_sha256",
    "raw_completion",
    "parsed_source",
    "parse_error",
    "patch_id",
}
_TEST_FIELDS = {
    "schema_version",
    "task_id",
    "cwe_id",
    "split",
    "patch_id",
    "verifier_family",
    "prompt_mode",
    "model_id",
    "model_revision",
    "suite_index",
    "seed",
    "requested_tests",
    "prompt_sha256",
    "completion_sha256",
    "raw_completion",
    "parsed_tests",
    "parse_error",
}
_CLASSIFICATION_FIELDS = {
    "patch_id",
    "patch_family",
    "task_id",
    "cwe_id",
    "split",
    "raw_completion_sha256",
    "parsed_source",
    "classification",
}
_ELIGIBLE_FIELDS = {
    "schema_version",
    "patch_id",
    "patch_family",
    "task_id",
    "cwe_id",
    "split",
    "candidate_source",
    "candidate_sha256",
}


@dataclass(frozen=True)
class _Phase:
    name: str
    path: Path
    manifest: dict[str, Any]
    files: dict[str, Path]


@dataclass(frozen=True)
class _Patch:
    patch_id: str
    patch_family: str
    task: PublicTask
    candidate_source: str | None
    classification: dict[str, Any]
    raw: dict[str, Any]


@dataclass(frozen=True)
class _Suite:
    family: str
    mode: str
    task_id: str
    patch_id: str | None
    suite_index: int
    requested_tests: int
    completion_sha256: str
    tests: tuple[dict[str, Any], ...] | None
    parse_error: str | None


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return dict(value)


def _exact_fields(record: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if set(record) != expected:
        difference = sorted(set(record) ^ expected)
        raise ValueError(f"{label} fields differ from the frozen schema: {difference}")


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _text(value: object, *, label: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise ValueError(f"{label} must be {qualifier}")
    return value


def _hash(value: object, *, label: str) -> str:
    text = _text(value, label=label).lower()
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{label} is not a SHA-256 digest")
    return text


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_hash(value: object) -> str:
    # Evidence trees and analysis tables legitimately exceed the deliberately
    # tiny container bounds imposed on model-generated JSON vectors.
    encoded = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}") from exc
    return _mapping(value, label=label)


def _load_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        for index, value in enumerate(read_jsonl(path), start=1):
            records.append(_mapping(value, label=f"{label} row {index}"))
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}") from exc
    return records


def _snapshot(root: Path) -> dict[str, dict[str, Any]]:
    """Hash every evidence file, rejecting links and non-regular entries."""

    snapshot: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError(f"evidence may not contain symbolic links: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"evidence contains a non-regular entry: {path}")
        relative = path.relative_to(root).as_posix()
        snapshot[relative] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    if not snapshot:
        raise ValueError("evidence root is empty")
    return snapshot


def _outside_retrieved_run(output: Path, evidence_root: Path) -> None:
    run_root = evidence_root.parent
    resolved = output.resolve()
    if resolved == run_root or run_root in resolved.parents:
        raise ValueError("output report must be outside the retrieved run root")
    if output.exists():
        raise FileExistsError(f"output report already exists: {output}")


def _formal_environment(config: Mapping[str, Any]) -> dict[str, Any]:
    frozen = _mapping(
        config.get("offline_verification"), label="config offline_verification"
    )
    expected_packages = _mapping(
        frozen.get("package_versions"),
        label="config offline_verification package_versions",
    )
    observed = {
        "os_name": os.name,
        "platform_system": platform.system(),
        "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "pythonsafepath": os.environ.get("PYTHONSAFEPATH"),
        "pythonnousersite": os.environ.get("PYTHONNOUSERSITE"),
        "package_versions": {
            name: importlib.metadata.version(name) for name in expected_packages
        },
    }
    expected = {
        "os_name": str(frozen.get("os_name")),
        "platform_system": str(frozen.get("platform_system")),
        "python_major_minor": str(frozen.get("python_major_minor")),
        "pythonhashseed": str(frozen.get("pythonhashseed")),
        "pythonsafepath": str(frozen.get("pythonsafepath")),
        "pythonnousersite": str(frozen.get("pythonnousersite")),
        "package_versions": {str(k): str(v) for k, v in expected_packages.items()},
    }
    if observed != expected:
        raise ValueError(
            f"formal verification environment differs from frozen contract: "
            f"expected={expected!r}, observed={observed!r}"
        )
    return observed


def _formal_code_attestation(
    *, expected_code_sha256: str, expected_git_commit: str
) -> dict[str, str]:
    from .prepare import _code_inventory

    expected_code = _hash(expected_code_sha256, label="expected code-tree hash")
    if not re.fullmatch(r"[0-9a-f]{40}", expected_git_commit):
        raise ValueError("expected Git commit must be 40 lowercase hex characters")
    code_root = Path(__file__).resolve().parents[1]
    _files, actual_code = _code_inventory(code_root)
    if actual_code != expected_code:
        raise ValueError("local verifier code tree differs from its expected commitment")
    repo_root = Path(__file__).resolve().parents[2]
    actual_commit = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--verify", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_commit != expected_git_commit:
        raise ValueError("local verifier Git commit differs from its expected commitment")
    for args in (("diff", "--quiet"), ("diff", "--cached", "--quiet")):
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            raise ValueError("tracked verifier repository differs from the pinned Git commit")
    untracked = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "--others", "--exclude-standard", "-z"],
        check=True,
        capture_output=True,
    ).stdout.decode("utf-8").split("\0")
    import_relevant = [
        path
        for path in untracked
        if path
        and (
            path.startswith("src/")
            or ("/" not in path and path.endswith((".py", ".pth")))
            or path in {"sitecustomize.py", "usercustomize.py"}
        )
    ]
    if import_relevant:
        raise ValueError(
            f"untracked files could contaminate verifier imports: {import_relevant}"
        )
    return {"code_tree_sha256": actual_code, "git_commit": actual_commit}


def _validate_run_binding(
    phases_root: Path,
    phase_names: set[str],
    *,
    expected_run_binding_sha256: str,
) -> str:
    """Require every closed phase to carry the same out-of-band run binding."""

    expected = _hash(
        expected_run_binding_sha256, label="expected run-binding hash"
    )
    for phase_name in sorted(phase_names):
        manifest = _load_json(
            phases_root / phase_name / "MANIFEST.json",
            label=f"{phase_name} manifest",
        )
        observed = _hash(
            manifest.get("run_binding_sha256"),
            label=f"{phase_name} run-binding hash",
        )
        if observed != expected:
            raise ValueError(
                f"phase {phase_name} is not bound to the expected immutable run"
            )
    return expected


def _read_phase(
    evidence_root: Path,
    name: str,
    *,
    expected_files: set[str],
) -> _Phase:
    phase = evidence_root / "phases" / name
    if not phase.is_dir() or phase.is_symlink():
        raise ValueError(f"missing regular evidence phase: {name}")
    names = {item.name for item in phase.iterdir()}
    wanted = expected_files | {"MANIFEST.json", "COMPLETE"}
    if names != wanted:
        raise ValueError(
            f"phase {name} files differ from its frozen layout: {sorted(names ^ wanted)}"
        )
    if (phase / "COMPLETE").stat().st_size != 0:
        raise ValueError(f"phase {name} has a malformed COMPLETE marker")
    manifest = _load_json(phase / "MANIFEST.json", label=f"{name} manifest")
    if manifest.get("state") != "COMPLETE":
        raise ValueError(f"phase {name} is not complete")
    files = _mapping(manifest.get("files"), label=f"{name} manifest files")
    if set(files) != expected_files:
        raise ValueError(f"phase {name} manifest does not bind exactly its data files")
    paths: dict[str, Path] = {}
    for filename in sorted(expected_files):
        entry = _mapping(files[filename], label=f"{name}/{filename} commitment")
        _exact_fields(entry, {"sha256", "bytes"}, label=f"{name}/{filename} commitment")
        path = phase / filename
        expected_sha = _hash(entry["sha256"], label=f"{name}/{filename} sha256")
        expected_bytes = _integer(entry["bytes"], label=f"{name}/{filename} bytes")
        if expected_bytes < 0 or path.stat().st_size != expected_bytes:
            raise ValueError(f"phase file checksum mismatch (size differs): {name}/{filename}")
        if sha256_file(path) != expected_sha:
            raise ValueError(f"phase file checksum mismatch: {name}/{filename}")
        paths[filename] = path
    return _Phase(name, phase, manifest, paths)


def _deserialize_config(payload: bytes) -> tuple[dict[str, Any], GateThresholds]:
    try:
        value = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError) as exc:
        raise ValueError("frozen config is not valid UTF-8 YAML") from exc
    config = _mapping(value, label="config")
    if config.get("kind") != "validator_monoculture_g0":
        raise ValueError("config is not a validator-monoculture G0 config")
    models = _mapping(config.get("models"), label="config models")
    if set(models) != set(FAMILIES):
        raise ValueError("config must contain exactly the two frozen model families")
    for family in FAMILIES:
        model = _mapping(models[family], label=f"config model {family}")
        _text(model.get("id"), label=f"config model id {family}")
        _text(model.get("revision"), label=f"config model revision {family}")
    generation = _mapping(config.get("generation"), label="config generation")
    required_generation = {
        "patches_per_model_task",
        "spec_only_test_suites_per_verifier_task",
        "patch_aware_test_suites_per_verifier_patch",
        "tests_per_suite",
    }
    if not required_generation.issubset(generation):
        raise ValueError("config lacks frozen generation counts")
    tests_per_suite = _integer(generation["tests_per_suite"], label="tests_per_suite")
    if tests_per_suite <= 0:
        raise ValueError("tests_per_suite must be positive")
    for key in (
        "spec_only_test_suites_per_verifier_task",
        "patch_aware_test_suites_per_verifier_patch",
    ):
        suites = _integer(generation[key], label=key)
        if suites <= 0 or suites * tests_per_suite != 12:
            raise ValueError(f"{key} must define exactly 12 proposal slots")
    if _integer(generation["patches_per_model_task"], label="patches_per_model_task") <= 0:
        raise ValueError("patches_per_model_task must be positive")
    execution = _mapping(config.get("execution"), label="config execution")
    timeout = execution.get("sandbox_timeout_seconds")
    max_bytes = execution.get("max_test_completion_bytes")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0.05 <= float(timeout) <= 10:
        raise ValueError("sandbox_timeout_seconds is invalid")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or not 1024 <= max_bytes <= 1_000_000:
        raise ValueError("max_test_completion_bytes is invalid")
    generation_environment = _mapping(
        config.get("generation_environment"), label="config generation_environment"
    )
    required_environment = {
        "platform_system",
        "python_major_minor",
        "torch_version_prefix",
        "cuda_version",
        "transformers_version",
        "minimum_device_memory_bytes",
    }
    if set(generation_environment) != required_environment:
        raise ValueError("config generation_environment has the wrong fields")
    for key in required_environment - {"minimum_device_memory_bytes"}:
        _text(generation_environment[key], label=f"generation environment {key}")
    if _integer(
        generation_environment["minimum_device_memory_bytes"],
        label="minimum_device_memory_bytes",
    ) < 80 * 1024**3:
        raise ValueError("generation device-memory floor is below 80 GiB")
    analysis = _mapping(config.get("analysis"), label="config analysis")
    thresholds = GateThresholds(**analysis)
    if thresholds.proposal_test_budget != 12:
        raise ValueError("analysis proposal budget must equal the frozen 12 slots")
    return config, thresholds


def _read_frozen_bytes(path: Path, *, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read frozen {label}: {path}") from exc


def _validate_corpus_shape(tasks: Sequence[PublicTask], config: Mapping[str, Any]) -> None:
    corpus = _mapping(config.get("corpus"), label="config corpus")
    cwe_counts = Counter(task.cwe_id for task in tasks)
    expected_families = _integer(corpus.get("cwe_families"), label="corpus cwe_families")
    expected_variants = _integer(
        corpus.get("variants_per_cwe"), label="corpus variants_per_cwe"
    )
    if len(cwe_counts) != expected_families or any(
        count != expected_variants for count in cwe_counts.values()
    ):
        raise ValueError("public corpus does not match frozen CWE/variant counts")
    dev = {task.cwe_id for task in tasks if task.split.value == "development"}
    test = {task.cwe_id for task in tasks if task.split.value == "locked_test"}
    if len(dev) != _integer(corpus.get("dev_cwe_families"), label="dev_cwe_families"):
        raise ValueError("public corpus has the wrong development CWE count")
    if len(test) != _integer(corpus.get("test_cwe_families"), label="test_cwe_families"):
        raise ValueError("public corpus has the wrong locked-test CWE count")
    if dev & test:
        raise ValueError("a CWE family crosses the public split")


def _default_patch_classifier(
    task: PublicTask,
    oracle: PrivateOracle,
    candidate_source: str,
    *,
    timeout_seconds: float,
) -> Mapping[str, Any]:
    """Late-bound adapter so an alternative implementation can be plugged in."""

    try:
        from . import evaluation

        classifier = getattr(evaluation, "classify_patch")
    except (ImportError, AttributeError) as exc:  # pragma: no cover - compatibility path
        raise RuntimeError(
            "validator_monoculture.evaluation.classify_patch is unavailable; "
            "supply a patch_classifier adapter"
        ) from exc
    if not callable(classifier):  # pragma: no cover - defensive compatibility path
        raise RuntimeError("evaluation.classify_patch is not callable")
    return classifier(task, oracle, candidate_source, timeout_seconds=timeout_seconds)


def _default_vector_evaluator(
    task: PublicTask,
    oracle: PrivateOracle,
    candidate_source: str,
    vectors: Sequence[Mapping[str, object]],
    *,
    timeout_seconds: float,
) -> Mapping[str, Any]:
    from .evaluation import evaluate_generated_vectors

    return evaluate_generated_vectors(
        task,
        oracle,
        candidate_source,
        vectors,
        timeout_seconds=timeout_seconds,
    )


def _default_oracle_validator(
    tasks: Mapping[str, PublicTask],
    oracles: Mapping[str, PrivateOracle],
    *,
    timeout_seconds: float,
) -> Mapping[str, Any]:
    from .oracle_preflight import validate_oracle

    return validate_oracle(
        tasks,
        oracles,
        timeout_seconds=timeout_seconds,
    )


def _patch_identifier(record: Mapping[str, Any]) -> str:
    material = "|".join(
        (
            str(record["patch_family"]),
            str(record["task_id"]),
            str(record["sample_index"]),
            str(record["completion_sha256"]),
        )
    )
    return "patch-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def _parse_patch(task: PublicTask, completion: str) -> tuple[str | None, str | None]:
    try:
        source = parse_patch_completion(
            completion, entrypoint=task.entrypoint, signature=task.signature
        )
    except ValueError as exc:
        return None, type(exc).__name__ + ": " + str(exc)
    return source, None


def _validate_stored_canonical_patch(
    task: PublicTask,
    *,
    raw_local_source: str,
    stored_source: object,
) -> str:
    """Accept cross-Python formatting differences but no semantic/style drift."""

    if not isinstance(stored_source, str) or not stored_source:
        raise ValueError("stored parsed patch must be non-empty source")
    try:
        tokens = tokenize.generate_tokens(io.StringIO(stored_source).readline)
        if any(token.type == tokenize.COMMENT for token in tokens):
            raise ValueError("stored canonical patch contains a comment style channel")
        stored_local = parse_patch_completion(
            stored_source, entrypoint=task.entrypoint, signature=task.signature
        )
    except (tokenize.TokenError, ValueError) as exc:
        raise ValueError("stored canonical patch is invalid") from exc
    if stored_local != raw_local_source:
        raise ValueError("stored patch AST does not reconstruct from the raw completion")
    return stored_source


def _parse_suite(
    completion: str, requested_tests: int, max_completion_bytes: int
) -> tuple[tuple[dict[str, Any], ...] | None, str | None]:
    if len(completion.encode("utf-8")) > max_completion_bytes:
        return None, f"ValueError: verifier completion exceeds {max_completion_bytes} bytes"
    try:
        tests = parse_test_completion(completion, requested_tests=requested_tests)
        for index, record in enumerate(tests):
            TestVector.from_record(record, default_id=f"generated-{index:03d}")
    except ValueError as exc:
        return None, type(exc).__name__ + ": " + str(exc)
    return tuple(tests), None


def _validate_manifest_commitments(
    manifest: Mapping[str, Any],
    *,
    phase: str,
    public_sha256: str,
    config_sha256: str,
    private_sha256: str | None = None,
) -> None:
    if _hash(manifest.get("public_corpus_sha256"), label=f"{phase} public hash") != public_sha256:
        raise ValueError(f"phase {phase} is not bound to the expected public corpus")
    if _hash(manifest.get("config_sha256"), label=f"{phase} config hash") != config_sha256:
        raise ValueError(f"phase {phase} is not bound to the expected config")
    if private_sha256 is not None:
        observed = _hash(
            manifest.get("private_oracles_sha256"), label=f"{phase} private hash"
        )
        if observed != private_sha256:
            raise ValueError(f"phase {phase} is not bound to the expected private oracle")


def _validate_runtime_provenance(
    manifest: Mapping[str, Any],
    *,
    phase: str,
    model_id: str,
    revision: str,
    environment: Mapping[str, Any],
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    provenance = _mapping(
        manifest.get("runtime_provenance"), label=f"{phase} runtime provenance"
    )
    required = {
        "model_id", "model_revision", "chat_template_sha256",
        "transformers_version", "torch_version", "python_version",
        "cuda_version", "cuda_available", "device_name",
        "platform_system", "device_memory_bytes", "compute_capability",
    }
    if set(provenance) != required:
        raise ValueError(f"{phase} runtime provenance has the wrong fields")
    if provenance["model_id"] != model_id or provenance["model_revision"] != revision:
        raise ValueError(f"{phase} runtime provenance has the wrong model identity")
    _hash(provenance["chat_template_sha256"], label=f"{phase} chat-template hash")
    if provenance["transformers_version"] != environment["transformers_version"]:
        raise ValueError(f"{phase} used a non-frozen Transformers version")
    for field in ("torch_version", "python_version", "cuda_version", "device_name"):
        _text(provenance[field], label=f"{phase} {field}")
    if provenance["cuda_available"] is not True:
        raise ValueError(f"{phase} was not collected on CUDA")
    if provenance["platform_system"] != environment["platform_system"]:
        raise ValueError(f"{phase} used the wrong generation operating system")
    python_minor = str(environment["python_major_minor"])
    if not str(provenance["python_version"]).startswith(python_minor + "."):
        raise ValueError(f"{phase} used the wrong Python minor version")
    if not str(provenance["torch_version"]).startswith(
        str(environment["torch_version_prefix"])
    ):
        raise ValueError(f"{phase} used the wrong PyTorch version")
    if provenance["cuda_version"] != environment["cuda_version"]:
        raise ValueError(f"{phase} used the wrong CUDA runtime")
    memory = _integer(provenance["device_memory_bytes"], label=f"{phase} device memory")
    if memory < _integer(
        environment["minimum_device_memory_bytes"],
        label="minimum generation device memory",
    ):
        raise ValueError(f"{phase} ran on a GPU below the frozen memory floor")
    capability = provenance["compute_capability"]
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in capability)
    ):
        raise ValueError(f"{phase} has malformed compute capability provenance")
    if expected is not None and provenance != dict(expected):
        raise ValueError(
            f"{phase} runtime provenance differs across phases for family"
        )
    return provenance


def _reconstruct_patches(
    *,
    evidence_root: Path,
    tasks: Mapping[str, PublicTask],
    oracles: Mapping[str, PrivateOracle],
    config: Mapping[str, Any],
    public_sha256: str,
    private_sha256: str,
    config_sha256: str,
    patch_classifier: PatchClassifier,
    timeout_seconds: float,
) -> tuple[
    list[_Patch],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, str],
    dict[str, dict[str, Any]],
]:
    generation = _mapping(config["generation"], label="config generation")
    generation_environment = _mapping(
        config["generation_environment"], label="config generation_environment"
    )
    sample_count = _integer(
        generation["patches_per_model_task"], label="patches_per_model_task"
    )
    task_order = list(tasks)
    patches: list[_Patch] = []
    manifest_hashes: dict[str, str] = {}
    runtime_provenance: dict[str, dict[str, Any]] = {}
    for family in FAMILIES:
        phase_name = f"patches_{family}"
        phase = _read_phase(
            evidence_root, phase_name, expected_files={"raw_patch_completions.jsonl"}
        )
        manifest_hashes[phase_name] = sha256_file(phase.path / "MANIFEST.json")
        manifest = phase.manifest
        if manifest.get("kind") != PATCH_PHASE_KIND or manifest.get("family") != family:
            raise ValueError(f"phase {phase_name} has the wrong kind or family")
        model = _mapping(config["models"][family], label=f"config model {family}")
        model_id = _text(model["id"], label=f"model id {family}")
        revision = _text(model["revision"], label=f"model revision {family}")
        if manifest.get("model_id") != model_id or manifest.get("model_revision") != revision:
            raise ValueError(f"phase {phase_name} model identity differs from config")
        runtime_provenance[family] = _validate_runtime_provenance(
            manifest,
            phase=phase_name,
            model_id=model_id,
            revision=revision,
            environment=generation_environment,
        )
        _validate_manifest_commitments(
            manifest,
            phase=phase_name,
            public_sha256=public_sha256,
            config_sha256=config_sha256,
        )
        records = _load_jsonl(
            phase.files["raw_patch_completions.jsonl"], label=f"{phase_name} patches"
        )
        expected_count = len(tasks) * sample_count
        if (
            len(records) != expected_count
            or _integer(manifest.get("record_count"), label=f"{phase_name} record_count")
            != expected_count
            or _integer(manifest.get("task_count"), label=f"{phase_name} task_count")
            != len(tasks)
        ):
            raise ValueError(f"phase {phase_name} does not contain the exact patch budget")
        seen: set[tuple[str, int]] = set()
        for index, record in enumerate(records):
            label = f"{phase_name} patch row {index + 1}"
            _exact_fields(record, _PATCH_FIELDS, label=label)
            if record["schema_version"] != PATCH_SCHEMA:
                raise ValueError(f"{label} has the wrong schema version")
            task_id = _text(record["task_id"], label=f"{label} task_id")
            if task_id not in tasks:
                raise ValueError(f"{label} references an unknown task")
            task = tasks[task_id]
            sample_index = _integer(record["sample_index"], label=f"{label} sample_index")
            key = (task_id, sample_index)
            if not 0 <= sample_index < sample_count or key in seen:
                raise ValueError(f"{label} has a duplicate or out-of-range sample index")
            seen.add(key)
            expected_metadata = {
                "cwe_id": task.cwe_id,
                "split": task.split.value,
                "patch_family": family,
                "model_id": model_id,
                "model_revision": revision,
            }
            if any(record.get(field) != value for field, value in expected_metadata.items()):
                raise ValueError(f"{label} metadata differs from the frozen task/model")
            completion = _text(
                record["raw_completion"], label=f"{label} raw_completion", allow_empty=True
            )
            completion_sha = _sha256_text(completion)
            if _hash(record["completion_sha256"], label=f"{label} completion hash") != completion_sha:
                raise ValueError(f"{label} completion hash is wrong")
            prompt = patch_prompt(task.patch_prompt_record())
            if _hash(record["prompt_sha256"], label=f"{label} prompt hash") != _sha256_text(prompt):
                raise ValueError(f"{label} prompt commitment is wrong")
            expected_seed = deterministic_seed(task_id, revision, "patch", sample_index)
            if _integer(record["seed"], label=f"{label} seed") != expected_seed:
                raise ValueError(f"{label} seed is wrong")
            local_source, parse_error = _parse_patch(task, completion)
            if local_source is None:
                if record["parsed_source"] is not None:
                    raise ValueError(f"{label} stored patch parse does not reconstruct")
                stored_error = record["parse_error"]
                if (
                    not isinstance(stored_error, str)
                    or parse_error is None
                    or stored_error.split(":", 1)[0] != parse_error.split(":", 1)[0]
                ):
                    raise ValueError(f"{label} stored patch error category does not reconstruct")
                source = None
            else:
                if record["parse_error"] is not None:
                    raise ValueError(f"{label} stores an error for a valid patch")
                source = _validate_stored_canonical_patch(
                    task,
                    raw_local_source=local_source,
                    stored_source=record["parsed_source"],
                )
            patch_id = _patch_identifier(record)
            if record["patch_id"] != patch_id:
                raise ValueError(f"{label} patch_id does not reconstruct")
            if source is None:
                classification: dict[str, Any] = {
                    "schema_version": "validator-monoculture-patch-classification-v2",
                    "task_id": task_id,
                    "cwe_id": task.cwe_id,
                    "split": task.split.value,
                    "status": "REJECTED_COMPLETION_PARSE",
                    "plausible_security_repair": False,
                    "fully_correct": False,
                    "parse_error": parse_error,
                }
            else:
                result = patch_classifier(
                    task,
                    oracles[task_id],
                    source,
                    timeout_seconds=timeout_seconds,
                )
                classification = _mapping(result, label=f"{label} reconstructed classification")
                if classification.get("task_id") != task_id:
                    raise ValueError(f"{label} classifier returned the wrong task_id")
                if not isinstance(classification.get("plausible_security_repair"), bool):
                    raise ValueError(f"{label} classifier omitted a boolean eligibility result")
            patches.append(_Patch(patch_id, family, task, source, classification, record))
        expected_keys = {(task_id, sample) for task_id in task_order for sample in range(sample_count)}
        if seen != expected_keys:
            raise ValueError(f"phase {phase_name} lacks an exact task/sample crossing")

    ids = [patch.patch_id for patch in patches]
    if len(ids) != len(set(ids)):
        raise ValueError("reconstructed patch IDs are not globally unique")

    classification_phase = _read_phase(
        evidence_root,
        "classifications",
        expected_files={"private_classifications.jsonl", "eligible_patches.jsonl"},
    )
    manifest_hashes["classifications"] = sha256_file(
        classification_phase.path / "MANIFEST.json"
    )
    manifest = classification_phase.manifest
    if manifest.get("kind") != CLASSIFICATION_PHASE_KIND:
        raise ValueError("classification phase has the wrong kind")
    frozen_timeout = float(_mapping(config["execution"], label="config execution")["sandbox_timeout_seconds"])
    if float(manifest.get("sandbox_timeout_seconds", -1)) != frozen_timeout:
        raise ValueError("classification phase used a non-frozen sandbox timeout")
    _validate_manifest_commitments(
        manifest,
        phase="classifications",
        public_sha256=public_sha256,
        private_sha256=private_sha256,
        config_sha256=config_sha256,
    )
    reconstructed_classifications: list[dict[str, Any]] = []
    reconstructed_eligible: list[dict[str, Any]] = []
    for patch in patches:
        row = {
            "patch_id": patch.patch_id,
            "patch_family": patch.patch_family,
            "task_id": patch.task.task_id,
            "cwe_id": patch.task.cwe_id,
            "split": patch.task.split.value,
            "raw_completion_sha256": patch.raw["completion_sha256"],
            "parsed_source": patch.candidate_source,
            "classification": patch.classification,
        }
        reconstructed_classifications.append(row)
        if (
            patch.candidate_source is not None
            and patch.classification.get("plausible_security_repair") is True
        ):
            reconstructed_eligible.append(
                {
                    "schema_version": ELIGIBLE_SCHEMA,
                    "patch_id": patch.patch_id,
                    "patch_family": patch.patch_family,
                    "task_id": patch.task.task_id,
                    "cwe_id": patch.task.cwe_id,
                    "split": patch.task.split.value,
                    "candidate_source": patch.candidate_source,
                    "candidate_sha256": _sha256_text(patch.candidate_source),
                }
            )
    recorded_classifications = _load_jsonl(
        classification_phase.files["private_classifications.jsonl"],
        label="stored private classifications",
    )
    recorded_eligible = _load_jsonl(
        classification_phase.files["eligible_patches.jsonl"],
        label="stored eligible patches",
    )
    for index, row in enumerate(recorded_classifications, start=1):
        _exact_fields(row, _CLASSIFICATION_FIELDS, label=f"classification row {index}")
    for index, row in enumerate(recorded_eligible, start=1):
        _exact_fields(row, _ELIGIBLE_FIELDS, label=f"eligible row {index}")
    if recorded_classifications != reconstructed_classifications:
        raise ValueError("stored private classifications do not reconstruct from raw patches")
    if recorded_eligible != reconstructed_eligible:
        raise ValueError("stored eligible registry does not reconstruct from raw patches")
    expected_by_family = {
        family: sum(row["patch_family"] == family for row in reconstructed_eligible)
        for family in FAMILIES
    }
    if (
        _integer(manifest.get("record_count"), label="classification record_count")
        != len(reconstructed_classifications)
        or _integer(manifest.get("eligible_count"), label="classification eligible_count")
        != len(reconstructed_eligible)
        or manifest.get("eligible_by_family") != expected_by_family
        or manifest.get("config_kind") != "validator_monoculture_g0"
    ):
        raise ValueError("classification manifest counts do not reconstruct")
    first_environment = runtime_provenance[FAMILIES[0]]
    second_environment = runtime_provenance[FAMILIES[1]]
    shared_environment_fields = {
        "transformers_version",
        "torch_version",
        "python_version",
        "platform_system",
        "cuda_version",
        "cuda_available",
        "device_name",
        "device_memory_bytes",
        "compute_capability",
    }
    if any(
        first_environment[field] != second_environment[field]
        for field in shared_environment_fields
    ):
        raise ValueError("generation environment changed between model families")
    return (
        patches,
        reconstructed_classifications,
        reconstructed_eligible,
        manifest_hashes,
        runtime_provenance,
    )


def _reconstruct_test_phases(
    *,
    evidence_root: Path,
    tasks: Mapping[str, PublicTask],
    eligible: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    public_sha256: str,
    config_sha256: str,
    runtime_provenance: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[tuple[str, str, str, str | None], tuple[_Suite, ...]], dict[str, str]]:
    generation = _mapping(config["generation"], label="config generation")
    generation_environment = _mapping(
        config["generation_environment"], label="config generation_environment"
    )
    execution = _mapping(config["execution"], label="config execution")
    max_completion_bytes = _integer(
        execution["max_test_completion_bytes"], label="max_test_completion_bytes"
    )
    tests_per_suite = _integer(generation["tests_per_suite"], label="tests_per_suite")
    eligible_by_id = {str(row["patch_id"]): dict(row) for row in eligible}
    if len(eligible_by_id) != len(eligible):
        raise ValueError("eligible patch IDs are not unique")
    suites_by_target: dict[tuple[str, str, str, str | None], list[_Suite]] = {}
    manifest_hashes: dict[str, str] = {}
    for family in FAMILIES:
        model = _mapping(config["models"][family], label=f"config model {family}")
        model_id = _text(model["id"], label=f"model id {family}")
        revision = _text(model["revision"], label=f"model revision {family}")
        for mode in PROMPT_MODES:
            phase_name = f"tests_{family}_{mode}"
            phase = _read_phase(
                evidence_root, phase_name, expected_files={"raw_test_completions.jsonl"}
            )
            manifest_hashes[phase_name] = sha256_file(phase.path / "MANIFEST.json")
            manifest = phase.manifest
            if (
                manifest.get("kind") != TEST_PHASE_KIND
                or manifest.get("family") != family
                or manifest.get("prompt_mode") != mode
                or manifest.get("model_id") != model_id
                or manifest.get("model_revision") != revision
            ):
                raise ValueError(f"phase {phase_name} metadata differs from config")
            _validate_runtime_provenance(
                manifest,
                phase=phase_name,
                model_id=model_id,
                revision=revision,
                environment=generation_environment,
                expected=runtime_provenance[family],
            )
            _validate_manifest_commitments(
                manifest,
                phase=phase_name,
                public_sha256=public_sha256,
                config_sha256=config_sha256,
            )
            suites_per_target_key = (
                "spec_only_test_suites_per_verifier_task"
                if mode == "spec_only"
                else "patch_aware_test_suites_per_verifier_patch"
            )
            suites_per_target = _integer(
                generation[suites_per_target_key], label=suites_per_target_key
            )
            if mode == "spec_only":
                targets = [(task_id, None) for task_id in tasks]
            else:
                targets = [(str(row["task_id"]), str(row["patch_id"])) for row in eligible]
            records = _load_jsonl(
                phase.files["raw_test_completions.jsonl"], label=f"{phase_name} tests"
            )
            expected_count = len(targets) * suites_per_target
            if (
                len(records) != expected_count
                or _integer(manifest.get("record_count"), label=f"{phase_name} record_count")
                != expected_count
                or _integer(manifest.get("target_count"), label=f"{phase_name} target_count")
                != len(targets)
                or _integer(
                    manifest.get("suites_per_target"), label=f"{phase_name} suites_per_target"
                )
                != suites_per_target
                or _integer(
                    manifest.get("tests_per_suite"), label=f"{phase_name} tests_per_suite"
                )
                != tests_per_suite
            ):
                raise ValueError(f"phase {phase_name} does not contain the exact suite budget")
            seen: set[tuple[str, str | None, int]] = set()
            for index, record in enumerate(records):
                label = f"{phase_name} test row {index + 1}"
                _exact_fields(record, _TEST_FIELDS, label=label)
                if record["schema_version"] != TEST_SCHEMA:
                    raise ValueError(f"{label} has the wrong schema version")
                task_id = _text(record["task_id"], label=f"{label} task_id")
                if task_id not in tasks:
                    raise ValueError(f"{label} references an unknown task")
                task = tasks[task_id]
                patch_id_value = record["patch_id"]
                if mode == "spec_only":
                    if patch_id_value is not None:
                        raise ValueError(f"{label} leaks a patch into the spec-only arm")
                    patch_id: str | None = None
                    candidate_source: str | None = None
                else:
                    patch_id = _text(patch_id_value, label=f"{label} patch_id")
                    if patch_id not in eligible_by_id:
                        raise ValueError(f"{label} references a non-eligible patch")
                    patch = eligible_by_id[patch_id]
                    if patch["task_id"] != task_id:
                        raise ValueError(f"{label} patch/task binding is wrong")
                    candidate_source = _text(
                        patch["candidate_source"], label=f"{label} candidate source"
                    )
                suite_index = _integer(record["suite_index"], label=f"{label} suite_index")
                key = (task_id, patch_id, suite_index)
                if not 0 <= suite_index < suites_per_target or key in seen:
                    raise ValueError(f"{label} has a duplicate or out-of-range suite index")
                seen.add(key)
                expected_metadata = {
                    "cwe_id": task.cwe_id,
                    "split": task.split.value,
                    "verifier_family": family,
                    "prompt_mode": mode,
                    "model_id": model_id,
                    "model_revision": revision,
                }
                if any(record.get(field) != value for field, value in expected_metadata.items()):
                    raise ValueError(f"{label} metadata differs from the task/model")
                requested = _integer(record["requested_tests"], label=f"{label} requested_tests")
                if requested != tests_per_suite:
                    raise ValueError(f"{label} requested the wrong suite size")
                completion = _text(
                    record["raw_completion"], label=f"{label} raw_completion", allow_empty=True
                )
                completion_sha = _sha256_text(completion)
                if _hash(record["completion_sha256"], label=f"{label} completion hash") != completion_sha:
                    raise ValueError(f"{label} completion hash is wrong")
                prompt = verifier_prompt(
                    task.verifier_prompt_record(candidate_source),
                    candidate_source,
                    requested_tests=requested,
                )
                if _hash(record["prompt_sha256"], label=f"{label} prompt hash") != _sha256_text(prompt):
                    raise ValueError(f"{label} prompt commitment is wrong")
                expected_seed = deterministic_seed(
                    task_id,
                    revision,
                    "test",
                    mode,
                    patch_id or "all-patches",
                    suite_index,
                )
                if _integer(record["seed"], label=f"{label} seed") != expected_seed:
                    raise ValueError(f"{label} seed is wrong")
                parsed, parse_error = _parse_suite(
                    completion, requested, max_completion_bytes
                )
                parsed_for_comparison = list(parsed) if parsed is not None else None
                if record["parsed_tests"] != parsed_for_comparison or record["parse_error"] != parse_error:
                    raise ValueError(f"{label} stored test parse does not reconstruct")
                target = (family, mode, task_id, patch_id)
                suites_by_target.setdefault(target, []).append(
                    _Suite(
                        family,
                        mode,
                        task_id,
                        patch_id,
                        suite_index,
                        requested,
                        completion_sha,
                        parsed,
                        parse_error,
                    )
                )
            expected_keys = {
                (task_id, patch_id, suite_index)
                for task_id, patch_id in targets
                for suite_index in range(suites_per_target)
            }
            if seen != expected_keys:
                raise ValueError(f"phase {phase_name} lacks an exact target/suite crossing")
    frozen: dict[tuple[str, str, str, str | None], tuple[_Suite, ...]] = {}
    for target, suites in suites_by_target.items():
        frozen[target] = tuple(sorted(suites, key=lambda item: item.suite_index))
    return frozen, manifest_hashes


def _slot_id(
    *,
    family: str,
    mode: str,
    scope: str,
    suite_index: int,
    slot_index: int,
    content_sha256: str,
) -> str:
    namespace = _canonical_hash(
        {"verifier_family": family, "prompt_mode": mode, "scope": scope}
    )[:20]
    return (
        f"test-{family}-{mode}-{namespace}-s{suite_index:02d}-"
        f"i{slot_index:02d}-{content_sha256[:20]}"
    )


def _evaluate_arm(
    *,
    task: PublicTask,
    oracle: PrivateOracle,
    candidate_source: str,
    suites: Sequence[_Suite],
    family: str,
    mode: str,
    scope: str,
    vector_evaluator: VectorEvaluator,
    timeout_seconds: float,
) -> tuple[dict[str, Any], dict[str, int]]:
    proposals: list[str] = []
    unique_vectors: list[dict[str, object]] = []
    seen_content: dict[str, tuple[bytes, str]] = {}
    parsed_count = duplicate_count = malformed_count = 0
    for suite in suites:
        if suite.family != family or suite.mode != mode or suite.task_id != task.task_id:
            raise ValueError("suite metadata changes while constructing an analysis arm")
        if suite.tests is None:
            malformed_count += suite.requested_tests
            marker = _canonical_hash(
                {
                    "parse_error": suite.parse_error,
                    "completion_sha256": suite.completion_sha256,
                }
            )
            for slot_index in range(suite.requested_tests):
                proposals.append(
                    _slot_id(
                        family=family,
                        mode=mode,
                        scope=scope,
                        suite_index=suite.suite_index,
                        slot_index=slot_index,
                        content_sha256=marker,
                    )
                )
            continue
        if len(suite.tests) != suite.requested_tests:
            raise ValueError("a reconstructed suite has the wrong number of tests")
        for slot_index, test in enumerate(suite.tests):
            content = {
                "args": test["args"],
                "kwargs": test["kwargs"],
                "expected": test["expected"],
            }
            encoded = canonical_json_bytes(content)
            content_sha = hashlib.sha256(encoded).hexdigest()
            slot_id = _slot_id(
                family=family,
                mode=mode,
                scope=scope,
                suite_index=suite.suite_index,
                slot_index=slot_index,
                content_sha256=content_sha,
            )
            proposals.append(slot_id)
            parsed_count += 1
            prior = seen_content.get(content_sha)
            if prior is not None:
                if prior[0] != encoded:  # pragma: no cover - cryptographic collision guard
                    raise ValueError("SHA-256 collision while deduplicating generated tests")
                duplicate_count += 1
                continue
            seen_content[content_sha] = (encoded, slot_id)
            unique_vectors.append({"slot_id": slot_id, **test})
    if len(proposals) != 12 or len(set(proposals)) != 12:
        raise ValueError("each verifier arm must reconstruct exactly 12 unique proposal slots")
    evaluation = _mapping(
        vector_evaluator(
            task,
            oracle,
            candidate_source,
            unique_vectors,
            timeout_seconds=timeout_seconds,
        ),
        label="generated-vector sandbox evaluation",
    )
    unique_ids = [str(vector["slot_id"]) for vector in unique_vectors]
    returned_proposals = [str(item) for item in evaluation.get("proposal_test_ids", [])]
    if returned_proposals != unique_ids:
        raise ValueError("vector evaluator did not return the exact canonical unique proposals")
    valid = [str(item) for item in evaluation.get("valid_test_ids", [])]
    killed = [str(item) for item in evaluation.get("kill_test_ids", [])]
    execution_counts = _mapping(
        evaluation.get("counts"), label="generated-vector execution counts"
    )
    indeterminate_execution_count = _integer(
        execution_counts.get("indeterminate_execution_count"),
        label="indeterminate execution count",
    )
    if (
        len(valid) != len(set(valid))
        or len(killed) != len(set(killed))
        or not set(valid).issubset(unique_ids)
        or not set(killed).issubset(valid)
        or not 0 <= indeterminate_execution_count <= len(unique_ids)
    ):
        raise ValueError("vector evaluator violated proposal/valid/kill containment")
    return (
        {
            "proposal_test_ids": proposals,
            "valid_test_ids": valid,
            "kill_test_ids": killed,
            "indeterminate_execution_count": indeterminate_execution_count,
        },
        {
            "proposal_slots": len(proposals),
            "parsed_slots": parsed_count,
            "malformed_slots": malformed_count,
            "duplicate_content_slots": duplicate_count,
            "unique_content_slots": len(unique_vectors),
            "reference_valid_slots": len(valid),
            "kill_slots": len(killed),
            "indeterminate_execution_slots": indeterminate_execution_count,
        },
    )


def _construct_rows(
    *,
    eligible: Sequence[Mapping[str, Any]],
    tasks: Mapping[str, PublicTask],
    oracles: Mapping[str, PrivateOracle],
    suites: Mapping[tuple[str, str, str, str | None], Sequence[_Suite]],
    vector_evaluator: VectorEvaluator,
    timeout_seconds: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    for patch in eligible:
        patch_id = str(patch["patch_id"])
        patch_family = str(patch["patch_family"])
        task_id = str(patch["task_id"])
        task = tasks[task_id]
        candidate_source = str(patch["candidate_source"])
        for mode in PROMPT_MODES:
            for family in FAMILIES:
                suite_patch_id = None if mode == "spec_only" else patch_id
                key = (family, mode, task_id, suite_patch_id)
                if key not in suites:
                    raise ValueError(f"missing exact verifier suite arm: {key}")
                arm, counts = _evaluate_arm(
                    task=task,
                    oracle=oracles[task_id],
                    candidate_source=candidate_source,
                    suites=suites[key],
                    family=family,
                    mode=mode,
                    scope=task_id if suite_patch_id is None else patch_id,
                    vector_evaluator=vector_evaluator,
                    timeout_seconds=timeout_seconds,
                )
                totals.update(counts)
                rows.append(
                    {
                        "task_id": task_id,
                        "split": task.split.value,
                        "cwe": task.cwe_id,
                        "patch_id": patch_id,
                        "patch_family": patch_family,
                        "verifier_family": family,
                        "prompt_mode": mode,
                        **arm,
                    }
                )
    expected = {
        (str(patch["patch_id"]), family, mode)
        for patch in eligible
        for family in FAMILIES
        for mode in PROMPT_MODES
    }
    observed = {
        (str(row["patch_id"]), str(row["verifier_family"]), str(row["prompt_mode"]))
        for row in rows
    }
    if observed != expected or len(rows) != len(expected):
        raise ValueError("analysis rows do not form the exact crossed verifier arms")
    return rows, dict(sorted(totals.items()))


def _planted_control_power(
    *,
    tasks: Mapping[str, PublicTask],
    oracles: Mapping[str, PrivateOracle],
    suites: Mapping[tuple[str, str, str, str | None], Sequence[_Suite]],
    patch_classifier: PatchClassifier,
    vector_evaluator: VectorEvaluator,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Measure absolute test power on one frozen incomplete mutant per TEST task."""

    rows: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    for task_id in sorted(tasks):
        task = tasks[task_id]
        if task.split.value != "locked_test":
            continue
        oracle = oracles[task_id]
        selected = None
        for mutant in sorted(oracle.mutants, key=lambda item: item.mutant_id):
            classification = _mapping(
                patch_classifier(
                    task,
                    oracle,
                    mutant.source,
                    timeout_seconds=timeout_seconds,
                ),
                label=f"planted control classification {task_id}/{mutant.mutant_id}",
            )
            if classification.get("plausible_security_repair") is True:
                selected = mutant
                break
        if selected is None:
            raise ValueError(f"TEST task lacks a plausible planted control: {task_id}")
        for family in FAMILIES:
            key = (family, "spec_only", task_id, None)
            if key not in suites:
                raise ValueError(f"missing specification-only planted-control suite: {key}")
            arm, counts = _evaluate_arm(
                task=task,
                oracle=oracle,
                candidate_source=selected.source,
                suites=suites[key],
                family=family,
                mode="spec_only",
                scope=task_id,
                vector_evaluator=vector_evaluator,
                timeout_seconds=timeout_seconds,
            )
            totals.update(counts)
            rows.append(
                {
                    "task_id": task_id,
                    "cwe": task.cwe_id,
                    "mutant_id": selected.mutant_id,
                    "verifier_family": family,
                    "detected": bool(arm["kill_test_ids"]),
                    "indeterminate_execution_count": int(
                        arm["indeterminate_execution_count"]
                    ),
                }
            )
    expected = len(
        {task.task_id for task in tasks.values() if task.split.value == "locked_test"}
    ) * len(FAMILIES)
    if len(rows) != expected:
        raise ValueError("planted-control rows do not cover every TEST task and verifier")
    cwes = sorted({str(row["cwe"]) for row in rows})
    rates: dict[str, float] = {}
    cell_power: dict[str, dict[str, dict[str, float | int]]] = {}
    for family in FAMILIES:
        per_cwe = []
        cell_power[family] = {}
        for cwe in cwes:
            values = [
                float(row["detected"])
                for row in rows
                if row["verifier_family"] == family and row["cwe"] == cwe
            ]
            if not values:
                raise ValueError("planted controls lack exact CWE/family support")
            detected = int(sum(values))
            total = len(values)
            rate = float(detected / total)
            cell_power[family][cwe] = {
                "detected_count": detected,
                "total_count": total,
                "detection_rate": rate,
            }
            per_cwe.append(rate)
        rates[family] = float(sum(per_cwe) / len(per_cwe))
    covered_cwes = [
        cwe
        for cwe in cwes
        if any(row["cwe"] == cwe and row["detected"] for row in rows)
    ]
    return {
        "kind": "validator_monoculture_planted_control_power",
        "selection_rule": "lexicographically_first_plausible_incomplete_mutant_per_test_task",
        "task_count": expected // len(FAMILIES),
        "cwe_count": len(cwes),
        "cwes": cwes,
        "macro_detection_rate_by_verifier": rates,
        "detection_power_by_verifier_cwe": cell_power,
        "covered_cwes": covered_cwes,
        "cwe_coverage_rate": float(len(covered_cwes) / len(cwes)) if cwes else 0.0,
        "indeterminate_execution_count": sum(
            int(row["indeterminate_execution_count"]) for row in rows
        ),
        "execution_counts": dict(sorted(totals.items())),
        "rows_sha256": _canonical_hash(rows),
    }


def _apply_planted_control_gate(
    gate: dict[str, Any],
    planted_controls: Mapping[str, Any],
    thresholds: GateThresholds,
) -> None:
    """Require absolute power independently in every verifier-by-CWE cell."""

    control_rates = _mapping(
        planted_controls.get("macro_detection_rate_by_verifier"),
        label="planted-control detection rates",
    )
    macro_power_ok = all(
        float(control_rates.get(family, -1.0))
        >= thresholds.minimum_planted_control_detection_rate
        for family in FAMILIES
    )
    cwes_raw = planted_controls.get("cwes")
    if not isinstance(cwes_raw, list) or not cwes_raw:
        raise ValueError("planted controls lack the held-out CWE registry")
    cwes = [str(cwe) for cwe in cwes_raw]
    if len(cwes) != len(set(cwes)):
        raise ValueError("planted-control CWE registry contains duplicates")
    power_by_family = _mapping(
        planted_controls.get("detection_power_by_verifier_cwe"),
        label="planted-control verifier-by-CWE power",
    )
    if set(power_by_family) != set(FAMILIES):
        raise ValueError("planted-control power lacks the exact verifier families")
    every_cell_power_ok = True
    for family in FAMILIES:
        cells = _mapping(
            power_by_family[family],
            label=f"planted-control cells for {family}",
        )
        if set(cells) != set(cwes):
            raise ValueError(
                f"planted-control power lacks exact CWE support for {family}"
            )
        for cwe in cwes:
            cell = _mapping(cells[cwe], label=f"planted-control cell {family}/{cwe}")
            detected = _integer(
                cell.get("detected_count"), label=f"{family}/{cwe} detected count"
            )
            total = _integer(
                cell.get("total_count"), label=f"{family}/{cwe} total count"
            )
            rate = cell.get("detection_rate")
            if (
                total <= 0
                or detected < 0
                or detected > total
                or isinstance(rate, bool)
                or not isinstance(rate, (int, float))
                or abs(float(rate) - detected / total) > 1e-12
            ):
                raise ValueError(f"planted-control cell is malformed: {family}/{cwe}")
            every_cell_power_ok = every_cell_power_ok and (
                detected >= 1
                and float(rate)
                >= thresholds.minimum_planted_control_cell_detection_rate
            )
    cwe_coverage_ok = (
        float(planted_controls.get("cwe_coverage_rate", -1.0))
        >= thresholds.minimum_planted_control_cwe_coverage
    )
    control_execution_ok = (
        _integer(
            planted_controls.get("indeterminate_execution_count"),
            label="planted-control indeterminate execution count",
        )
        == 0
    )
    macro_and_coverage_ok = macro_power_ok and cwe_coverage_ok
    combined_power_ok = macro_and_coverage_ok and every_cell_power_ok
    gate["checks"]["planted_control_macro_detection_power"] = macro_and_coverage_ok
    gate["checks"]["planted_control_every_verifier_cwe_cell_power"] = every_cell_power_ok
    gate["checks"]["planted_control_execution_clean"] = control_execution_ok
    gate["kill_checks"]["planted_control_macro_detection_power"] = macro_and_coverage_ok
    gate["kill_checks"]["planted_control_every_verifier_cwe_cell_power"] = every_cell_power_ok
    gate["kill_checks"]["planted_control_execution_clean"] = control_execution_ok
    if not control_execution_ok:
        gate["decision"] = "INCONCLUSIVE_EXECUTION_ANOMALIES"
    elif not combined_power_ok:
        gate["decision"] = "INCONCLUSIVE_INSUFFICIENT_APPARATUS_POWER"
    gate["pass"] = gate["decision"] == "EXPAND_VALIDATOR_MONOCULTURE"
    gate["reasons"] = [
        name for name, passed in gate["checks"].items() if not passed
    ]


def verify(
    *,
    evidence_root: str | Path,
    public_corpus: str | Path,
    private_oracle: str | Path,
    config: str | Path,
    expected_public_sha256: str,
    expected_private_sha256: str,
    expected_config_sha256: str,
    output_report: str | Path,
    expected_evidence_sha256: str | None = None,
    expected_code_sha256: str | None = None,
    expected_git_commit: str | None = None,
    expected_run_binding_sha256: str | None = None,
    patch_classifier: PatchClassifier | None = None,
    vector_evaluator: VectorEvaluator | None = None,
    oracle_validator: OracleValidator | None = None,
    allow_test_hooks: bool = False,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Verify a complete G0 evidence root and write one external report.

    Production verification requires evidence, input, code, Git, and run-binding
    out-of-band commitments.  The
    adapters are an explicit compatibility-test path only; reports produced
    with that path are marked invalid as scientific evidence.
    """

    root = Path(evidence_root).resolve()
    source_root = root
    public_path = Path(public_corpus).resolve()
    private_path = Path(private_oracle).resolve()
    config_path = Path(config).resolve()
    output = Path(output_report).resolve()
    if not root.is_dir():
        raise ValueError(f"evidence root is not a directory: {root}")
    _outside_retrieved_run(output, root)
    injected_hooks = any(
        hook is not None
        for hook in (patch_classifier, vector_evaluator, oracle_validator)
    )
    if injected_hooks and not allow_test_hooks:
        raise ValueError("injected verifier adapters require allow_test_hooks=True")
    test_hooks_active = allow_test_hooks
    if not test_hooks_active and (
        expected_evidence_sha256 is None
        or expected_code_sha256 is None
        or expected_git_commit is None
        or expected_run_binding_sha256 is None
    ):
        raise ValueError(
            "production verification requires expected evidence, code-tree, Git, and run-binding commitments"
        )
    expected_public = _hash(expected_public_sha256, label="expected public hash")
    expected_private = _hash(expected_private_sha256, label="expected private hash")
    expected_config = _hash(expected_config_sha256, label="expected config hash")
    public_bytes = _read_frozen_bytes(public_path, label="public corpus")
    private_bytes = _read_frozen_bytes(private_path, label="private oracle")
    config_bytes = _read_frozen_bytes(config_path, label="config")
    actual_public = hashlib.sha256(public_bytes).hexdigest()
    actual_private = hashlib.sha256(private_bytes).hexdigest()
    actual_config = hashlib.sha256(config_bytes).hexdigest()
    for label, actual, expected in (
        ("public corpus", actual_public, expected_public),
        ("private oracle", actual_private, expected_private),
        ("config", actual_config, expected_config),
    ):
        if actual != expected:
            raise ValueError(f"{label} checksum differs from its expected commitment")
    before = _snapshot(root)
    evidence_sha = _canonical_hash(before)
    if expected_evidence_sha256 is not None:
        expected_evidence = _hash(
            expected_evidence_sha256, label="expected evidence hash"
        )
        if evidence_sha != expected_evidence:
            raise ValueError("evidence tree checksum differs from its expected commitment")
    captured_evidence: tempfile.TemporaryDirectory[str] | None = None
    if not test_hooks_active:
        captured_evidence = tempfile.TemporaryDirectory(
            prefix="validator-monoculture-verified-snapshot-"
        )
        captured_root = Path(captured_evidence.name) / "evidence"
        shutil.copytree(source_root, captured_root)
        if _snapshot(captured_root) != before:
            raise RuntimeError("evidence changed while creating the immutable analysis snapshot")
        root = captured_root
    phases = root / "phases"
    expected_phase_names = {
        "classifications",
        *(f"patches_{family}" for family in FAMILIES),
        *(
            f"tests_{family}_{mode}"
            for family in FAMILIES
            for mode in PROMPT_MODES
        ),
    }
    if not phases.is_dir() or {item.name for item in phases.iterdir()} != expected_phase_names:
        raise ValueError("evidence root does not contain exactly the seven frozen phases")

    run_binding_sha256: str | None = None
    if not test_hooks_active:
        assert expected_run_binding_sha256 is not None
        run_binding_sha256 = _validate_run_binding(
            phases,
            expected_phase_names,
            expected_run_binding_sha256=expected_run_binding_sha256,
        )

    frozen_config, thresholds = _deserialize_config(config_bytes)
    formal_environment: dict[str, Any] | None = None
    code_attestation: dict[str, str] | None = None
    if not test_hooks_active:
        assert expected_code_sha256 is not None and expected_git_commit is not None
        formal_environment = _formal_environment(frozen_config)
        code_attestation = _formal_code_attestation(
            expected_code_sha256=expected_code_sha256,
            expected_git_commit=expected_git_commit,
        )
    frozen_timeout = float(
        _mapping(frozen_config["execution"], label="config execution")[
            "sandbox_timeout_seconds"
        ]
    )
    timeout_seconds = frozen_timeout if timeout_seconds is None else timeout_seconds
    if timeout_seconds != frozen_timeout:
        raise ValueError("verification timeout differs from the frozen config")
    public_tasks = deserialize_public_tasks(public_bytes)
    private_oracles = deserialize_private_oracles(private_bytes)
    tasks, oracles = bind_corpus(public_tasks, private_oracles)
    _validate_corpus_shape(public_tasks, frozen_config)
    if oracle_validator is None:
        oracle_preflight_value = _default_oracle_validator(
            tasks,
            oracles,
            timeout_seconds=timeout_seconds,
        )
    else:
        oracle_preflight_value = oracle_validator(
            public_path,
            private_path,
            timeout_seconds=timeout_seconds,
        )
    oracle_preflight = _mapping(
        oracle_preflight_value,
        label="reconstructed oracle preflight",
    )
    if (
        oracle_preflight.get("kind") != "validator_monoculture_oracle_preflight"
        or oracle_preflight.get("status") != "PASS"
        or oracle_preflight.get("interpretation") != "apparatus_validation_only"
    ):
        raise ValueError("reconstructed oracle preflight did not pass")
    classifier = patch_classifier or _default_patch_classifier
    evaluator = vector_evaluator or _default_vector_evaluator
    (
        patches,
        classifications,
        eligible,
        patch_manifest_hashes,
        runtime_provenance,
    ) = _reconstruct_patches(
        evidence_root=root,
        tasks=tasks,
        oracles=oracles,
        config=frozen_config,
        public_sha256=actual_public,
        private_sha256=actual_private,
        config_sha256=actual_config,
        patch_classifier=classifier,
        timeout_seconds=timeout_seconds,
    )
    suites, test_manifest_hashes = _reconstruct_test_phases(
        evidence_root=root,
        tasks=tasks,
        eligible=eligible,
        config=frozen_config,
        public_sha256=actual_public,
        config_sha256=actual_config,
        runtime_provenance=runtime_provenance,
    )
    planted_controls = _planted_control_power(
        tasks=tasks,
        oracles=oracles,
        suites=suites,
        patch_classifier=classifier,
        vector_evaluator=evaluator,
        timeout_seconds=timeout_seconds,
    )
    rows, test_counts = _construct_rows(
        eligible=eligible,
        tasks=tasks,
        oracles=oracles,
        suites=suites,
        vector_evaluator=evaluator,
        timeout_seconds=timeout_seconds,
    )
    gate = evaluate_gate(rows, families=FAMILIES, thresholds=thresholds)
    _apply_planted_control_gate(gate, planted_controls, thresholds)
    gate["planted_control_power"] = planted_controls
    status_counts = Counter(str(row["classification"].get("status")) for row in classifications)
    report: dict[str, Any] = dict(gate)
    if test_hooks_active:
        report["decision"] = "INVALID_TEST_HOOKS_ACTIVE"
        report["pass"] = False
    report["verification"] = {
        "status": (
            "TEST_HOOKS_ACTIVE_NOT_SCIENTIFIC_EVIDENCE"
            if test_hooks_active
            else "VERIFIED_FROM_RAW_COMPLETIONS"
        ),
        "evidence_root_sha256": evidence_sha,
        "public_corpus_sha256": actual_public,
        "private_oracle_sha256": actual_private,
        "config_sha256": actual_config,
        "phase_manifest_sha256": {
            **patch_manifest_hashes,
            **test_manifest_hashes,
        },
        "raw_patch_count": len(patches),
        "eligible_patch_count": len(eligible),
        "classification_status_counts": dict(sorted(status_counts.items())),
        "analysis_row_count": len(rows),
        "analysis_rows_sha256": _canonical_hash(rows),
        "test_reconstruction_counts": test_counts,
        "formal_environment": formal_environment,
        "code_attestation": code_attestation,
        "run_binding_sha256": run_binding_sha256,
    }
    report["oracle_preflight"] = oracle_preflight
    report["analysis_rows"] = rows
    after = _snapshot(root)
    source_after = _snapshot(source_root)
    if after != before or source_after != before:
        raise RuntimeError("evidence root changed during offline verification")
    if not test_hooks_active:
        assert expected_code_sha256 is not None and expected_git_commit is not None
        final_code_attestation = _formal_code_attestation(
            expected_code_sha256=expected_code_sha256,
            expected_git_commit=expected_git_commit,
        )
        if final_code_attestation != code_attestation:
            raise RuntimeError("verifier code attestation changed during analysis")
    if captured_evidence is not None:
        captured_evidence.cleanup()
    write_json(output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--public-corpus", type=Path, required=True)
    parser.add_argument(
        "--private-oracle", "--private-oracles", dest="private_oracle", type=Path, required=True
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--expected-public-sha256",
        "--expected-public-corpus-sha256",
        dest="expected_public_sha256",
        required=True,
    )
    parser.add_argument(
        "--expected-private-sha256",
        "--expected-private-oracle-sha256",
        dest="expected_private_sha256",
        required=True,
    )
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-evidence-sha256", required=True)
    parser.add_argument("--expected-code-sha256", required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--expected-run-binding-sha256", required=True)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument(
        "--output-report", "--destination", dest="output_report", type=Path, required=True
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = verify(**vars(args))
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "verification": report["verification"]["status"],
                "output_report": str(Path(args.output_report).resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
