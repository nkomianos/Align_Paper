"""Phase-separated collection for the frozen validator-monoculture G0 gate.

Patch generation and test generation load public material only.  The CPU
classification phase is the sole collection phase that opens the private
oracle, and it emits a separate model-safe eligible-patch registry so hidden
case outcomes never enter a verifier prompt process.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import yaml

from under_extinction.io import canonical_json, sha256_file, write_json, write_jsonl

from .evaluation import classify_patch
from .prompts import patch_prompt, parse_patch_completion, parse_test_completion, verifier_prompt
from .runtime import TextRuntime, deterministic_seed
from .schema import PrivateOracle, PublicTask, TestVector
from .serde import bind_corpus, load_private_oracles, load_public_tasks


FAMILIES = ("qwen3_5", "gemma4")
PROMPT_MODES = ("spec_only", "patch_aware")
RuntimeFactory = Callable[..., TextRuntime]


def _run_binding_fields() -> dict[str, str]:
    value = os.environ.get("VALIDATOR_MONOCULTURE_RUN_BINDING_SHA256")
    if value is None:
        return {}
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("VALIDATOR_MONOCULTURE_RUN_BINDING_SHA256 must be 64 lowercase hex")
    return {"run_binding_sha256": value}


@dataclass(frozen=True)
class _InputSnapshot:
    path: Path
    data: bytes
    sha256: str


def _snapshot_input(path: str | Path) -> _InputSnapshot:
    resolved = Path(path).resolve()
    try:
        data = resolved.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot snapshot input: {resolved}") from exc
    return _InputSnapshot(
        path=resolved, data=data, sha256=hashlib.sha256(data).hexdigest()
    )


def _load_config_snapshot(snapshot: _InputSnapshot) -> dict[str, Any]:
    try:
        text = snapshot.data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("config is not UTF-8") from exc
    value = yaml.safe_load(text)
    if not isinstance(value, dict) or value.get("kind") != "validator_monoculture_g0":
        raise ValueError("config is not a validator-monoculture G0 config")
    models = value.get("models")
    generation = value.get("generation")
    if not isinstance(models, dict) or set(models) != set(FAMILIES):
        raise ValueError("config must contain exactly the two frozen model families")
    if not isinstance(generation, dict):
        raise ValueError("config lacks generation settings")
    for family in FAMILIES:
        model = models[family]
        if not isinstance(model, dict) or not model.get("id") or not model.get("revision"):
            raise ValueError(f"model config is incomplete for {family}")
    return value


def _load_config(path: str | Path) -> dict[str, Any]:
    return _load_config_snapshot(_snapshot_input(path))


def _load_snapshot_with(
    snapshot: _InputSnapshot, loader: Callable[[str | Path], Any]
) -> Any:
    """Parse exactly the bytes whose digest is bound into the phase marker."""

    descriptor, temp_name = tempfile.mkstemp(suffix=snapshot.path.suffix)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(snapshot.data)
            handle.flush()
            os.fsync(handle.fileno())
        return loader(temp_name)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def _read_jsonl_snapshot(snapshot: _InputSnapshot) -> list[dict[str, Any]]:
    try:
        text = snapshot.data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"JSONL is not UTF-8: {snapshot.path}") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSONL at {snapshot.path}:{line_number}"
            ) from exc
        if not isinstance(record, dict):
            raise ValueError(
                f"JSONL record is not an object at {snapshot.path}:{line_number}"
            )
        records.append(record)
    return records


@contextmanager
def _phase_lease(root: str | Path, phase_name: str):
    """Hold a non-blocking OS lease for one phase, released even after a crash."""

    output_root = Path(root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    lease_dir = output_root / ".validator_monoculture_leases"
    lease_dir.mkdir(exist_ok=True)
    lease_path = lease_dir / f"{phase_name}.lock"
    handle = lease_path.open("a+b")
    acquired = False
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(
                f"phase is already leased by another runner: {phase_name}"
            ) from exc
        acquired = True
        yield
    finally:
        if acquired:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _fsync_directory(path: Path) -> None:
    """Best-effort durability for directory-entry changes.

    Windows does not permit opening directories with :func:`os.open`, while
    POSIX filesystems require syncing the containing directory for a rename or
    new file to be durable.  Record contents are always fsynced independently.
    """

    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _create_empty_file(path: Path) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _read_json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot validate {description}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} is not a JSON object")
    return value


def _preserve_atomic_temporaries(phase: Path) -> None:
    """Move crash-left atomic-write temporaries into immutable recovery evidence."""

    temporaries = sorted(
        path for path in phase.iterdir() if path.is_file() and path.name.startswith(".")
    )
    if not temporaries:
        return
    recovery = phase.parent.parent / "recovery" / phase.name / "atomic-temporaries"
    recovery.mkdir(parents=True, exist_ok=True)
    for path in temporaries:
        digest = sha256_file(path)
        target = recovery / f"{path.name.lstrip('.')}.{digest}.bin"
        suffix = 0
        while target.exists():
            suffix += 1
            target = recovery / f"{path.name.lstrip('.')}.{digest}.{suffix}.bin"
        os.rename(path, target)
    _fsync_directory(recovery)
    _fsync_directory(phase)


def _validate_completed_phase(phase: Path) -> dict[str, Any]:
    manifest = _read_json_object(
        phase / "MANIFEST.json", description="phase MANIFEST"
    )
    if manifest.get("state") != "COMPLETE" or not (phase / "COMPLETE").is_file():
        raise ValueError("phase completion markers are inconsistent")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("phase manifest lacks a file inventory")
    observed_names = {
        path.name
        for path in phase.iterdir()
        if path.is_file()
        and path.name not in {"RUNNING.json", "MANIFEST.json", "COMPLETE"}
    }
    if observed_names != set(files):
        raise ValueError("completed phase file inventory does not match the manifest")
    for name, metadata in files.items():
        path = phase / name
        if (
            not isinstance(metadata, dict)
            or metadata.get("sha256") != sha256_file(path)
            or metadata.get("bytes") != path.stat().st_size
        ):
            raise ValueError(f"completed phase evidence does not match: {name}")
    return manifest


def _complete_interrupted_finalization(phase: Path) -> dict[str, Any]:
    """Validate COMPLETE+RUNNING and idempotently remove the stale marker."""

    manifest = _validate_completed_phase(phase)
    running = phase / "RUNNING.json"
    if running.is_file():
        running.unlink()
        _fsync_directory(phase)
    return manifest


def _phase_directory(
    root: str | Path,
    name: str,
    *,
    resume: bool = False,
    binding: Mapping[str, Any] | None = None,
) -> Path:
    output_root = Path(root).resolve()
    if not output_root.is_absolute():
        raise ValueError("output root must resolve to an absolute path")
    output_root.mkdir(parents=True, exist_ok=True)
    phases = output_root / "phases"
    phases.mkdir(exist_ok=True)
    phase = phases / name
    marker = {
        "kind": "validator_monoculture_phase",
        "phase": name,
        "binding": dict(binding or {}),
    }
    if not phase.exists():
        if resume:
            raise FileNotFoundError(f"no incomplete phase exists to resume: {phase}")
        phase.mkdir(exist_ok=False)
        write_json(phase / "RUNNING.json", marker)
        return phase
    if not resume:
        raise FileExistsError(f"phase already exists: {phase}")
    if not phase.is_dir():
        raise ValueError(f"phase path is not a directory: {phase}")
    _preserve_atomic_temporaries(phase)
    running = phase / "RUNNING.json"
    if (phase / "COMPLETE").exists():
        if not resume or not running.is_file():
            raise FileExistsError(f"completed phase is immutable: {phase}")
        observed = _read_json_object(running, description="phase RUNNING marker")
        if observed != marker:
            raise ValueError("phase RUNNING marker does not match the requested inputs")
        _validate_completed_phase(phase)
        return phase
    if not running.is_file():
        raise ValueError("incomplete phase lacks its RUNNING marker")
    observed = _read_json_object(running, description="phase RUNNING marker")
    if observed != marker:
        raise ValueError("phase RUNNING marker does not match the requested inputs")
    return phase


def _partial_path(phase: Path, final_name: str, *, resume: bool) -> tuple[Path, Path]:
    """Return the durable working path and immutable final path for a phase.

    A process can die after promoting the partial JSONL but before writing the
    phase manifest.  On an explicit resume, move that uncommitted final file
    back to its partial name so it receives the same strict validation as any
    other interrupted evidence.
    """

    final = phase / final_name
    partial = final.with_name(final.name + ".partial")
    if partial.exists() and final.exists():
        raise ValueError(f"phase contains both partial and final evidence: {final_name}")
    if final.exists():
        if not resume:
            raise FileExistsError(f"uncommitted final evidence already exists: {final}")
        if not final.is_file():
            raise ValueError(f"uncommitted evidence is not a regular file: {final}")
        os.rename(final, partial)
        _fsync_directory(phase)
    if not partial.exists():
        _create_empty_file(partial)
    elif not partial.is_file():
        raise ValueError(f"partial evidence is not a regular file: {partial}")
    return partial, final


def _preserve_and_truncate_torn_suffix(path: Path, data: bytes) -> bytes:
    prefix_end = data.rfind(b"\n") + 1
    prefix, suffix = data[:prefix_end], data[prefix_end:]
    recovery = path.parent.parent.parent / "recovery" / path.parent.name
    recovery.mkdir(parents=True, exist_ok=True)
    suffix_hash = hashlib.sha256(suffix).hexdigest()
    recovered = recovery / f"{path.name}.torn-{suffix_hash}.bin"
    if recovered.exists():
        if not recovered.is_file() or recovered.read_bytes() != suffix:
            raise ValueError(f"torn-suffix recovery collision: {recovered}")
    else:
        with recovered.open("xb") as handle:
            handle.write(suffix)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(recovery)
    with path.open("r+b") as handle:
        if handle.read() != data:
            raise ValueError(f"partial evidence changed during recovery: {path}")
        handle.seek(prefix_end)
        handle.truncate()
        handle.flush()
        os.fsync(handle.fileno())
    return prefix


def _load_partial_jsonl(
    path: Path, *, recover_torn: bool = False
) -> list[dict[str, Any]]:
    """Read only complete, canonically encoded records from a partial JSONL."""

    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read partial evidence: {path}") from exc
    if data and not data.endswith(b"\n"):
        if not recover_torn:
            raise ValueError(f"partial evidence has a torn final record: {path}")
        data = _preserve_and_truncate_torn_suffix(path, data)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"partial evidence is not UTF-8: {path}") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise ValueError(f"partial evidence contains a blank row at line {line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid partial JSONL at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"partial record {line_number} is not a JSON object")
        if line != canonical_json(value):
            raise ValueError(f"partial record {line_number} is not canonically encoded")
        records.append(value)
    return records


def _append_jsonl_record(path: Path, record: Mapping[str, Any]) -> None:
    """Append one complete record and make it durable before returning."""

    encoded = canonical_json(dict(record)) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _write_or_validate_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    """Create a derived final file, or validate one left by interrupted finalization."""

    if path.exists():
        if not path.is_file():
            raise ValueError(f"derived evidence is not a regular file: {path}")
        if _load_partial_jsonl(path) != list(records):
            raise ValueError(f"existing derived evidence does not match replay: {path}")
        return
    write_jsonl(path, records)


def _promote_partial(partial: Path, final: Path) -> None:
    if not partial.is_file():
        raise RuntimeError(f"partial evidence disappeared: {partial}")
    if final.exists():
        raise FileExistsError(f"refusing to overwrite final evidence: {final}")
    os.rename(partial, final)
    _fsync_directory(final.parent)


def _finish_phase(phase: Path, *, manifest: Mapping[str, Any]) -> dict[str, Any]:
    running = phase / "RUNNING.json"
    if (phase / "COMPLETE").is_file():
        if not running.is_file():
            raise FileExistsError(f"completed phase is immutable: {phase}")
        return _complete_interrupted_finalization(phase)
    if not running.is_file():
        raise RuntimeError("phase lost its RUNNING marker")
    payload = dict(manifest)
    payload["state"] = "COMPLETE"
    payload["files"] = {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(phase.iterdir())
        if path.is_file() and path.name not in {"RUNNING.json", "MANIFEST.json"}
    }
    write_json(phase / "MANIFEST.json", payload)
    _create_empty_file(phase / "COMPLETE")
    running.unlink()
    _fsync_directory(phase)
    return payload


def _family_model(config: Mapping[str, Any], family: str) -> tuple[str, str]:
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    model = config["models"][family]
    return str(model["id"]), str(model["revision"])


def _generation(config: Mapping[str, Any]) -> dict[str, Any]:
    generation = config["generation"]
    return {
        "do_sample": bool(generation["do_sample"]),
        "temperature": float(generation["temperature"]),
        "top_p": float(generation["top_p"]),
    }


def _execution(config: Mapping[str, Any]) -> tuple[float, int]:
    value = config.get("execution")
    if not isinstance(value, Mapping):
        raise ValueError("config lacks execution settings")
    timeout = float(value.get("sandbox_timeout_seconds", 0))
    max_bytes = int(value.get("max_test_completion_bytes", 0))
    if not 0.05 <= timeout <= 10.0 or not 1024 <= max_bytes <= 1_000_000:
        raise ValueError("config execution limits are invalid")
    return timeout, max_bytes


def _runtime_provenance(runtime: Any, model_id: str, revision: str) -> dict[str, Any]:
    value = runtime.provenance() if hasattr(runtime, "provenance") else {
        "model_id": model_id,
        "model_revision": revision,
        "runtime": type(runtime).__name__,
    }
    if not isinstance(value, dict):
        raise ValueError("runtime provenance must be a mapping")
    return value


def _task_record(task: PublicTask) -> dict[str, Any]:
    return task.to_record()


def _patch_id(record: Mapping[str, Any]) -> str:
    material = "|".join((
        str(record["patch_family"]), str(record["task_id"]),
        str(record["sample_index"]), str(record["completion_sha256"]),
    ))
    return "patch-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def _patch_completion_record(
    *,
    task: PublicTask,
    family: str,
    model_id: str,
    revision: str,
    sample_index: int,
    seed: int,
    prompt_sha256: str,
    completion: str,
) -> dict[str, Any]:
    try:
        parsed = parse_patch_completion(
            completion, entrypoint=task.entrypoint, signature=task.signature
        )
    except ValueError as exc:
        parsed, parse_error = None, type(exc).__name__ + ": " + str(exc)
    else:
        parse_error = None
    completion_sha256 = hashlib.sha256(completion.encode("utf-8")).hexdigest()
    record = {
        "schema_version": "validator-monoculture-patch-completion-v1",
        "task_id": task.task_id,
        "cwe_id": task.cwe_id,
        "split": task.split.value,
        "patch_family": family,
        "model_id": model_id,
        "model_revision": revision,
        "sample_index": sample_index,
        "seed": seed,
        "prompt_sha256": prompt_sha256,
        "completion_sha256": completion_sha256,
        "raw_completion": completion,
        "parsed_source": parsed,
        "parse_error": parse_error,
    }
    record["patch_id"] = _patch_id(record)
    return record


def collect_patches(
    *,
    output_root: str | Path,
    public_corpus: str | Path,
    config_path: str | Path,
    family: str,
    local_files_only: bool = False,
    runtime_factory: RuntimeFactory = TextRuntime,
    resume: bool = False,
) -> dict[str, Any]:
    """Generate the fixed patch budget for one family from public inputs."""

    phase_name = f"patches_{family}"
    with _phase_lease(output_root, phase_name):
        return _collect_patches_locked(
            output_root=output_root,
            public_corpus=public_corpus,
            config_path=config_path,
            family=family,
            local_files_only=local_files_only,
            runtime_factory=runtime_factory,
            resume=resume,
        )


def _collect_patches_locked(
    *,
    output_root: str | Path,
    public_corpus: str | Path,
    config_path: str | Path,
    family: str,
    local_files_only: bool,
    runtime_factory: RuntimeFactory,
    resume: bool,
) -> dict[str, Any]:
    public_snapshot = _snapshot_input(public_corpus)
    config_snapshot = _snapshot_input(config_path)
    config = _load_config_snapshot(config_snapshot)
    tasks = _load_snapshot_with(public_snapshot, load_public_tasks)
    model_id, revision = _family_model(config, family)
    runtime = runtime_factory(model_id, revision, local_files_only=local_files_only)
    runtime_provenance = _runtime_provenance(runtime, model_id, revision)

    phase = _phase_directory(
        output_root,
        f"patches_{family}",
        resume=resume,
        binding={
            "public_corpus_sha256": public_snapshot.sha256,
            "config_sha256": config_snapshot.sha256,
            "family": family,
            "model_id": model_id,
            "model_revision": revision,
            "runtime_provenance": runtime_provenance,
            **_run_binding_fields(),
        },
    )
    if (phase / "COMPLETE").is_file():
        return _complete_interrupted_finalization(phase)
    partial, output = _partial_path(
        phase, "raw_patch_completions.jsonl", resume=resume
    )
    generation = _generation(config)
    count = int(config["generation"]["patches_per_model_task"])
    max_new_tokens = int(config["generation"]["patch_max_new_tokens"])
    plan: list[tuple[PublicTask, int, str, str, int]] = []
    for task in tasks:
        prompt = patch_prompt(task.patch_prompt_record())
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        for sample_index in range(count):
            seed = deterministic_seed(task.task_id, revision, "patch", sample_index)
            plan.append((task, sample_index, prompt, prompt_sha256, seed))

    existing = _load_partial_jsonl(partial, recover_torn=resume)
    if len(existing) > len(plan):
        raise ValueError("patch partial contains more records than the frozen plan")
    for row_index, (record, planned) in enumerate(zip(existing, plan, strict=False)):
        task, sample_index, _prompt, prompt_sha256, seed = planned
        completion = record.get("raw_completion")
        if not isinstance(completion, str):
            raise ValueError(f"patch partial row {row_index} lacks a raw completion")
        expected = _patch_completion_record(
            task=task,
            family=family,
            model_id=model_id,
            revision=revision,
            sample_index=sample_index,
            seed=seed,
            prompt_sha256=prompt_sha256,
            completion=completion,
        )
        if record != expected:
            raise ValueError(
                f"patch partial row {row_index} fails deterministic validation"
            )

    for task, sample_index, prompt, prompt_sha256, seed in plan[len(existing):]:
        completion = runtime.generate(
            prompt, seed=seed, max_new_tokens=max_new_tokens, **generation
        )
        record = _patch_completion_record(
            task=task,
            family=family,
            model_id=model_id,
            revision=revision,
            sample_index=sample_index,
            seed=seed,
            prompt_sha256=prompt_sha256,
            completion=completion,
        )
        _append_jsonl_record(partial, record)
    _promote_partial(partial, output)
    return _finish_phase(phase, manifest={
        "kind": "validator_monoculture_patch_collection",
        "family": family,
        "model_id": model_id,
        "model_revision": revision,
        "runtime_provenance": runtime_provenance,
        "record_count": len(plan),
        "task_count": len(tasks),
        "public_corpus_sha256": public_snapshot.sha256,
        "config_sha256": config_snapshot.sha256,
        **_run_binding_fields(),
    })


def _load_patch_records(output_root: str | Path) -> list[dict[str, Any]]:
    root = Path(output_root).resolve()
    records: list[dict[str, Any]] = []
    for family in FAMILIES:
        phase = root / "phases" / f"patches_{family}"
        if not (phase / "COMPLETE").is_file():
            raise ValueError(f"patch collection is incomplete for {family}")
        manifest = json.loads((phase / "MANIFEST.json").read_text(encoding="utf-8"))
        raw = phase / "raw_patch_completions.jsonl"
        expected = manifest["files"][raw.name]["sha256"]
        snapshot = _snapshot_input(raw)
        if snapshot.sha256 != expected:
            raise ValueError(f"patch evidence checksum mismatch for {family}")
        family_records = _read_jsonl_snapshot(snapshot)
        if any(record.get("patch_family") != family for record in family_records):
            raise ValueError("patch file contains a cross-family record")
        records.extend(family_records)
    ids = [str(record.get("patch_id")) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("patch IDs are not globally unique")
    return records


def _classification_record(
    *,
    raw: Mapping[str, Any],
    task: PublicTask,
    oracle: PrivateOracle,
    timeout_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    task_id = task.task_id
    completion = raw.get("raw_completion")
    if not isinstance(completion, str):
        raise ValueError(f"raw patch {raw.get('patch_id')} lacks a string completion")
    completion_sha256 = hashlib.sha256(completion.encode("utf-8")).hexdigest()
    if raw.get("completion_sha256") != completion_sha256:
        raise ValueError(f"raw patch {raw.get('patch_id')} has a completion checksum mismatch")
    if raw.get("patch_id") != _patch_id(raw):
        raise ValueError(f"raw patch {raw.get('patch_id')} has an invalid patch ID")
    try:
        source = parse_patch_completion(
            completion, entrypoint=task.entrypoint, signature=task.signature
        )
    except ValueError as exc:
        source = None
        classification: dict[str, Any] = {
            "schema_version": "validator-monoculture-patch-classification-v2",
            "task_id": task_id,
            "cwe_id": task.cwe_id,
            "split": task.split.value,
            "status": "REJECTED_COMPLETION_PARSE",
            "plausible_security_repair": False,
            "fully_correct": False,
            "parse_error": type(exc).__name__ + ": " + str(exc),
        }
    else:
        classification = classify_patch(
            task, oracle, source, timeout_seconds=timeout_seconds
        )
    row = {
        "patch_id": raw["patch_id"],
        "patch_family": raw["patch_family"],
        "task_id": task_id,
        "cwe_id": task.cwe_id,
        "split": task.split.value,
        "raw_completion_sha256": raw["completion_sha256"],
        "parsed_source": source,
        "classification": classification,
    }
    eligible = None
    if classification.get("plausible_security_repair") is True and source is not None:
        eligible = {
            "schema_version": "validator-monoculture-eligible-patch-v1",
            "patch_id": raw["patch_id"],
            "patch_family": raw["patch_family"],
            "task_id": task_id,
            "cwe_id": task.cwe_id,
            "split": task.split.value,
            "candidate_source": source,
            "candidate_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        }
    return row, eligible


def classify_patches(
    *,
    output_root: str | Path,
    public_corpus: str | Path,
    private_oracles: str | Path,
    config_path: str | Path,
    timeout_seconds: float | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Classify every raw patch and write a hidden report plus safe registry."""

    with _phase_lease(output_root, "classifications"):
        return _classify_patches_locked(
            output_root=output_root,
            public_corpus=public_corpus,
            private_oracles=private_oracles,
            config_path=config_path,
            timeout_seconds=timeout_seconds,
            resume=resume,
        )


def _classify_patches_locked(
    *,
    output_root: str | Path,
    public_corpus: str | Path,
    private_oracles: str | Path,
    config_path: str | Path,
    timeout_seconds: float | None,
    resume: bool,
) -> dict[str, Any]:
    public_snapshot = _snapshot_input(public_corpus)
    private_snapshot = _snapshot_input(private_oracles)
    config_snapshot = _snapshot_input(config_path)
    config = _load_config_snapshot(config_snapshot)

    frozen_timeout, _ = _execution(config)
    timeout_seconds = frozen_timeout if timeout_seconds is None else timeout_seconds
    if timeout_seconds != frozen_timeout:
        raise ValueError("classification timeout differs from the frozen config")
    tasks, oracles = bind_corpus(
        _load_snapshot_with(public_snapshot, load_public_tasks),
        _load_snapshot_with(private_snapshot, load_private_oracles),
    )
    records = _load_patch_records(output_root)
    patch_registry_sha256 = hashlib.sha256(
        canonical_json([
            [record.get("patch_id"), record.get("completion_sha256")]
            for record in records
        ]).encode("utf-8")
    ).hexdigest()
    phase = _phase_directory(
        output_root,
        "classifications",
        resume=resume,
        binding={
            "public_corpus_sha256": public_snapshot.sha256,
            "private_oracles_sha256": private_snapshot.sha256,
            "config_sha256": config_snapshot.sha256,
            "patch_registry_sha256": patch_registry_sha256,
            "sandbox_timeout_seconds": timeout_seconds,
            **_run_binding_fields(),
        },
    )
    if (phase / "COMPLETE").is_file():
        return _complete_interrupted_finalization(phase)
    partial, classifications_path = _partial_path(
        phase, "private_classifications.jsonl", resume=resume
    )
    classifications = _load_partial_jsonl(partial, recover_torn=resume)
    if len(classifications) > len(records):
        raise ValueError("classification partial exceeds the frozen patch registry")
    eligible: list[dict[str, Any]] = []
    for row_index, (observed, raw) in enumerate(
        zip(classifications, records, strict=False)
    ):
        task_id = str(raw["task_id"])
        expected, eligible_row = _classification_record(
            raw=raw,
            task=tasks[task_id],
            oracle=oracles[task_id],
            timeout_seconds=timeout_seconds,
        )
        if observed != expected:
            raise ValueError(
                f"classification partial row {row_index} fails deterministic replay"
            )
        if eligible_row is not None:
            eligible.append(eligible_row)

    for raw in records[len(classifications):]:
        task_id = str(raw["task_id"])
        row, eligible_row = _classification_record(
            raw=raw,
            task=tasks[task_id],
            oracle=oracles[task_id],
            timeout_seconds=timeout_seconds,
        )
        _append_jsonl_record(partial, row)
        classifications.append(row)
        if eligible_row is not None:
            eligible.append(eligible_row)

    _promote_partial(partial, classifications_path)
    _write_or_validate_jsonl(phase / "eligible_patches.jsonl", eligible)
    return _finish_phase(phase, manifest={
        "kind": "validator_monoculture_patch_classification",
        "record_count": len(classifications),
        "eligible_count": len(eligible),
        "eligible_by_family": {
            family: sum(row["patch_family"] == family for row in eligible)
            for family in FAMILIES
        },
        "public_corpus_sha256": public_snapshot.sha256,
        "private_oracles_sha256": private_snapshot.sha256,
        "config_sha256": config_snapshot.sha256,
        "config_kind": config["kind"],
        "sandbox_timeout_seconds": timeout_seconds,
        **_run_binding_fields(),
    })


def _eligible_registry(output_root: str | Path) -> list[dict[str, Any]]:
    phase = Path(output_root).resolve() / "phases" / "classifications"
    if not (phase / "COMPLETE").is_file():
        raise ValueError("classification phase is incomplete")
    manifest = json.loads((phase / "MANIFEST.json").read_text(encoding="utf-8"))
    path = phase / "eligible_patches.jsonl"
    snapshot = _snapshot_input(path)
    if snapshot.sha256 != manifest["files"][path.name]["sha256"]:
        raise ValueError("eligible patch registry checksum mismatch")
    records = _read_jsonl_snapshot(snapshot)
    if len(records) != int(manifest["eligible_count"]):
        raise ValueError("eligible patch registry count mismatch")
    return records


def _parse_suite_for_record(
    completion: str, requested_tests: int, max_completion_bytes: int
) -> tuple[list[dict[str, Any]] | None, str | None]:
    if len(completion.encode("utf-8")) > max_completion_bytes:
        return None, f"ValueError: verifier completion exceeds {max_completion_bytes} bytes"
    try:
        tests = parse_test_completion(completion, requested_tests=requested_tests)
        # The wire type enforces bounded JSON depth, sizes, and numeric values.
        for index, record in enumerate(tests):
            TestVector.from_record(record, default_id=f"generated-{index:03d}")
    except ValueError as exc:
        return None, type(exc).__name__ + ": " + str(exc)
    return tests, None


def _test_completion_record(
    *,
    task: PublicTask,
    patch_id: str | None,
    family: str,
    prompt_mode: str,
    model_id: str,
    revision: str,
    suite_index: int,
    seed: int,
    requested_tests: int,
    prompt_sha256: str,
    completion: str,
    max_completion_bytes: int,
) -> dict[str, Any]:
    parsed, parse_error = _parse_suite_for_record(
        completion, requested_tests, max_completion_bytes
    )
    return {
        "schema_version": "validator-monoculture-test-completion-v1",
        "task_id": task.task_id,
        "cwe_id": task.cwe_id,
        "split": task.split.value,
        "patch_id": patch_id,
        "verifier_family": family,
        "prompt_mode": prompt_mode,
        "model_id": model_id,
        "model_revision": revision,
        "suite_index": suite_index,
        "seed": seed,
        "requested_tests": requested_tests,
        "prompt_sha256": prompt_sha256,
        "completion_sha256": hashlib.sha256(completion.encode("utf-8")).hexdigest(),
        "raw_completion": completion,
        "parsed_tests": parsed,
        "parse_error": parse_error,
    }


def collect_tests(
    *,
    output_root: str | Path,
    public_corpus: str | Path,
    config_path: str | Path,
    family: str,
    prompt_mode: str,
    local_files_only: bool = False,
    runtime_factory: RuntimeFactory = TextRuntime,
    resume: bool = False,
) -> dict[str, Any]:
    """Collect verifier suites for one family and one frozen prompt arm."""

    phase_name = f"tests_{family}_{prompt_mode}"
    with _phase_lease(output_root, phase_name):
        return _collect_tests_locked(
            output_root=output_root,
            public_corpus=public_corpus,
            config_path=config_path,
            family=family,
            prompt_mode=prompt_mode,
            local_files_only=local_files_only,
            runtime_factory=runtime_factory,
            resume=resume,
        )


def _collect_tests_locked(
    *,
    output_root: str | Path,
    public_corpus: str | Path,
    config_path: str | Path,
    family: str,
    prompt_mode: str,
    local_files_only: bool,
    runtime_factory: RuntimeFactory,
    resume: bool,
) -> dict[str, Any]:

    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"prompt_mode must be one of {PROMPT_MODES}")
    public_snapshot = _snapshot_input(public_corpus)
    config_snapshot = _snapshot_input(config_path)
    config = _load_config_snapshot(config_snapshot)
    tasks = {
        task.task_id: task
        for task in _load_snapshot_with(public_snapshot, load_public_tasks)
    }
    model_id, revision = _family_model(config, family)
    runtime = runtime_factory(model_id, revision, local_files_only=local_files_only)
    runtime_provenance = _runtime_provenance(runtime, model_id, revision)
    generation = _generation(config)
    generation_config = config["generation"]
    _, max_completion_bytes = _execution(config)
    requested_tests = int(generation_config["tests_per_suite"])
    suites = int(
        generation_config[
            "spec_only_test_suites_per_verifier_task"
            if prompt_mode == "spec_only"
            else "patch_aware_test_suites_per_verifier_patch"
        ]
    )
    max_new_tokens = int(generation_config["test_max_new_tokens"])
    if prompt_mode == "spec_only":
        targets = [
            {"task_id": task.task_id, "patch_id": None, "candidate_source": None}
            for task in tasks.values()
        ]
    else:
        targets = [
            {
                "task_id": str(row["task_id"]),
                "patch_id": str(row["patch_id"]),
                "candidate_source": str(row["candidate_source"]),
            }
            for row in _eligible_registry(output_root)
        ]
    target_registry_sha256 = hashlib.sha256(
        canonical_json(targets).encode("utf-8")
    ).hexdigest()
    phase = _phase_directory(
        output_root,
        f"tests_{family}_{prompt_mode}",
        resume=resume,
        binding={
            "public_corpus_sha256": public_snapshot.sha256,
            "config_sha256": config_snapshot.sha256,
            "family": family,
            "prompt_mode": prompt_mode,
            "model_id": model_id,
            "model_revision": revision,
            "runtime_provenance": runtime_provenance,
            "target_registry_sha256": target_registry_sha256,
            **_run_binding_fields(),
        },
    )
    if (phase / "COMPLETE").is_file():
        return _complete_interrupted_finalization(phase)
    partial, output = _partial_path(
        phase, "raw_test_completions.jsonl", resume=resume
    )
    plan: list[tuple[PublicTask, str | None, str, str, int, int]] = []
    for target in targets:
        task = tasks[target["task_id"]]
        prompt = verifier_prompt(
            task.verifier_prompt_record(target["candidate_source"]),
            target["candidate_source"],
            requested_tests=requested_tests,
        )
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        for suite_index in range(suites):
            seed = deterministic_seed(
                task.task_id, revision, "test", prompt_mode,
                target["patch_id"] or "all-patches", suite_index,
            )
            plan.append(
                (task, target["patch_id"], prompt, prompt_sha256, suite_index, seed)
            )

    existing = _load_partial_jsonl(partial, recover_torn=resume)
    if len(existing) > len(plan):
        raise ValueError("test partial contains more records than the frozen plan")
    for row_index, (record, planned) in enumerate(zip(existing, plan, strict=False)):
        task, patch_id, _prompt, prompt_sha256, suite_index, seed = planned
        completion = record.get("raw_completion")
        if not isinstance(completion, str):
            raise ValueError(f"test partial row {row_index} lacks a raw completion")
        expected = _test_completion_record(
            task=task,
            patch_id=patch_id,
            family=family,
            prompt_mode=prompt_mode,
            model_id=model_id,
            revision=revision,
            suite_index=suite_index,
            seed=seed,
            requested_tests=requested_tests,
            prompt_sha256=prompt_sha256,
            completion=completion,
            max_completion_bytes=max_completion_bytes,
        )
        if record != expected:
            raise ValueError(
                f"test partial row {row_index} fails deterministic validation"
            )

    for task, patch_id, prompt, prompt_sha256, suite_index, seed in plan[len(existing):]:
        completion = runtime.generate(
            prompt, seed=seed, max_new_tokens=max_new_tokens, **generation
        )
        record = _test_completion_record(
            task=task,
            patch_id=patch_id,
            family=family,
            prompt_mode=prompt_mode,
            model_id=model_id,
            revision=revision,
            suite_index=suite_index,
            seed=seed,
            requested_tests=requested_tests,
            prompt_sha256=prompt_sha256,
            completion=completion,
            max_completion_bytes=max_completion_bytes,
        )
        _append_jsonl_record(partial, record)
    _promote_partial(partial, output)
    return _finish_phase(phase, manifest={
        "kind": "validator_monoculture_test_collection",
        "family": family,
        "prompt_mode": prompt_mode,
        "model_id": model_id,
        "model_revision": revision,
        "runtime_provenance": runtime_provenance,
        "record_count": len(plan),
        "target_count": len(targets),
        "suites_per_target": suites,
        "tests_per_suite": requested_tests,
        "public_corpus_sha256": public_snapshot.sha256,
        "config_sha256": config_snapshot.sha256,
        **_run_binding_fields(),
    })


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    patch = subparsers.add_parser("patch-run")
    classify = subparsers.add_parser("classify")
    tests = subparsers.add_parser("test-run")
    for command in (patch, classify, tests):
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--public-corpus", type=Path, required=True)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--resume", action="store_true")
    patch.add_argument("--family", choices=FAMILIES, required=True)
    patch.add_argument("--local-files-only", action="store_true")
    classify.add_argument("--private-oracles", type=Path, required=True)
    classify.add_argument("--timeout-seconds", type=float)
    tests.add_argument("--family", choices=FAMILIES, required=True)
    tests.add_argument("--prompt-mode", choices=PROMPT_MODES, required=True)
    tests.add_argument("--local-files-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    common = {
        "output_root": args.output_root,
        "public_corpus": args.public_corpus,
        "config_path": args.config,
        "resume": args.resume,
    }
    if args.command == "patch-run":
        report = collect_patches(
            **common, family=args.family, local_files_only=args.local_files_only
        )
    elif args.command == "classify":
        report = classify_patches(
            **common,
            private_oracles=args.private_oracles,
            timeout_seconds=args.timeout_seconds,
        )
    else:
        report = collect_tests(
            **common,
            family=args.family,
            prompt_mode=args.prompt_mode,
            local_files_only=args.local_files_only,
        )
    print(json.dumps(report, sort_keys=True))
    return 0


def patch_main(argv: Sequence[str] | None = None) -> int:
    return main(["patch-run", *(list(argv) if argv is not None else __import__("sys").argv[1:])])


def classify_main(argv: Sequence[str] | None = None) -> int:
    return main(["classify", *(list(argv) if argv is not None else __import__("sys").argv[1:])])


def test_main(argv: Sequence[str] | None = None) -> int:
    return main(["test-run", *(list(argv) if argv is not None else __import__("sys").argv[1:])])


if __name__ == "__main__":
    raise SystemExit(main())
