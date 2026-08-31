"""Fail-closed corpus preparation for the validator-monoculture G0.

The preparation step is deliberately CPU-only.  It binds the deterministic
public/private corpus to the exact frozen config and to the Python source tree
that produced it before any model process is allowed to run.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import yaml

from . import corpus
from .schema import SCHEMA_VERSION, Split


PREPARATION_KIND = "validator_monoculture_g0_preparation"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: Path) -> str:
    """Hash one regular, non-symlink file."""

    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_expected_hash(label: str, observed: str, expected: str | None) -> None:
    if expected is None:
        return
    normalized = expected.strip().lower()
    if _SHA256.fullmatch(normalized) is None:
        raise ValueError(f"{label} expected SHA-256 must be 64 lowercase hexadecimal characters")
    if not hmac.compare_digest(observed, normalized):
        raise ValueError(f"{label} SHA-256 mismatch: expected {normalized}, observed {observed}")


def _load_config(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"config must be a regular non-symlink file: {path}")
    payload = path.read_bytes()
    try:
        parsed = yaml.safe_load(payload)
    except yaml.YAMLError as exc:
        raise ValueError(f"config is not valid YAML: {path}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("config root must be a mapping")
    return parsed, payload


def _validate_config_against_corpus(
    config: Mapping[str, Any], bundle: corpus.CorpusBundle
) -> None:
    if config.get("kind") != "validator_monoculture_g0":
        raise ValueError("config kind is not validator_monoculture_g0")
    corpus_config = config.get("corpus")
    if not isinstance(corpus_config, Mapping):
        raise ValueError("config corpus section must be a mapping")
    if corpus_config.get("split_unit") != "cwe_family":
        raise ValueError("validator-monoculture G0 must split at the CWE-family level")

    family_splits: dict[str, set[Split]] = {}
    family_counts: dict[str, int] = {}
    for task in bundle.public_tasks:
        family_splits.setdefault(task.cwe_id, set()).add(task.split)
        family_counts[task.cwe_id] = family_counts.get(task.cwe_id, 0) + 1
    if any(len(splits) != 1 for splits in family_splits.values()):
        raise ValueError("built corpus leaks a CWE family across DEV and locked TEST")

    development = sum(
        next(iter(splits)) is Split.DEVELOPMENT for splits in family_splits.values()
    )
    locked_test = sum(
        next(iter(splits)) is Split.LOCKED_TEST for splits in family_splits.values()
    )
    expected = {
        "cwe_families": len(family_splits),
        "variants_per_cwe": next(iter(set(family_counts.values())))
        if len(set(family_counts.values())) == 1
        else None,
        "dev_cwe_families": development,
        "test_cwe_families": locked_test,
    }
    for key, observed in expected.items():
        configured = corpus_config.get(key)
        if isinstance(configured, bool) or not isinstance(configured, int):
            raise ValueError(f"config corpus.{key} must be an integer")
        if observed is None or configured != observed:
            raise ValueError(
                f"config corpus.{key}={configured!r} disagrees with built corpus {observed!r}"
            )
    if expected["cwe_families"] * expected["variants_per_cwe"] != len(bundle.public_tasks):
        raise ValueError("built corpus is not a balanced CWE-by-variant grid")

    models = config.get("models")
    if not isinstance(models, Mapping) or set(models) != {"qwen3_5", "gemma4"}:
        raise ValueError("config must contain exactly the frozen qwen3_5 and gemma4 models")
    for model_key, model in models.items():
        if not isinstance(model, Mapping):
            raise ValueError(f"config models.{model_key} must be a mapping")
        for field in ("id", "revision"):
            value = model.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"config models.{model_key}.{field} must be a non-empty string")
    execution = config.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("config execution section must be a mapping")
    timeout = execution.get("sandbox_timeout_seconds")
    max_bytes = execution.get("max_test_completion_bytes")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0.05 <= float(timeout) <= 10:
        raise ValueError("config execution.sandbox_timeout_seconds is invalid")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or not 1024 <= max_bytes <= 1_000_000:
        raise ValueError("config execution.max_test_completion_bytes is invalid")


def _code_inventory(package_root: Path) -> tuple[dict[str, str], str]:
    if package_root.is_symlink() or not package_root.is_dir():
        raise ValueError(f"code root must be a regular directory: {package_root}")
    files = sorted(path for path in package_root.rglob("*.py") if path.is_file())
    if not files:
        raise ValueError(f"no Python source files found below {package_root}")
    inventory: dict[str, str] = {}
    for path in files:
        if path.is_symlink():
            raise ValueError(f"refusing symlinked source file: {path}")
        relative = path.relative_to(package_root.parent).as_posix()
        inventory[relative] = sha256_file(path)
    payload = (
        json.dumps(inventory, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    return inventory, hashlib.sha256(payload).hexdigest()


def prepare_corpus(
    destination: Path,
    config_path: Path,
    *,
    expected_config_sha256: str | None = None,
    expected_corpus_sha256: str | None = None,
    expected_code_sha256: str | None = None,
    code_root: Path | None = None,
) -> dict[str, Any]:
    """Create and attest one immutable corpus directory.

    All expected hashes and config/corpus consistency checks occur before the
    first output write.  The destination is then created exclusively by
    :func:`validator_monoculture.corpus.write_corpus` and is never reused.
    """

    destination = Path(destination)
    config_path = Path(config_path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite corpus destination: {destination}")

    config, config_bytes = _load_config(config_path)
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    _require_expected_hash("config", config_sha256, expected_config_sha256)

    bundle = corpus.build_corpus()
    _validate_config_against_corpus(config, bundle)
    _require_expected_hash("corpus", bundle.corpus_sha256, expected_corpus_sha256)

    # Inventory the whole source tree, not only this package: collection imports
    # shared I/O and model-template code from under_extinction as well.
    package_root = (code_root or Path(__file__).resolve().parents[1]).resolve(strict=True)
    code_files, code_tree_sha256 = _code_inventory(package_root)
    _require_expected_hash("code tree", code_tree_sha256, expected_code_sha256)

    corpus_manifest = corpus.write_corpus(destination)
    if corpus_manifest.get("corpus_sha256") != bundle.corpus_sha256:
        raise RuntimeError("written corpus manifest disagrees with the prevalidated corpus hash")

    manifest_path = destination / "MANIFEST.json"
    public_path = destination / "public" / "tasks.jsonl"
    private_path = destination / "private" / "oracles.jsonl"
    disk_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if disk_manifest != corpus_manifest:
        raise RuntimeError("written corpus manifest does not match write_corpus return value")
    if sha256_file(public_path) != corpus_manifest.get("public_tasks_sha256"):
        raise RuntimeError("written public corpus hash mismatch")
    if sha256_file(private_path) != corpus_manifest.get("private_oracles_sha256"):
        raise RuntimeError("written private corpus hash mismatch")

    frozen_config = destination / "FROZEN_CONFIG.yaml"
    with frozen_config.open("xb") as handle:
        handle.write(config_bytes)

    input_sha256 = {
        "FROZEN_CONFIG.yaml": sha256_file(frozen_config),
        "MANIFEST.json": sha256_file(manifest_path),
        "private/oracles.jsonl": sha256_file(private_path),
        "public/tasks.jsonl": sha256_file(public_path),
    }
    preparation: dict[str, Any] = {
        "kind": PREPARATION_KIND,
        "schema_version": SCHEMA_VERSION,
        "config_filename": config_path.name,
        "config_sha256": config_sha256,
        "corpus_sha256": bundle.corpus_sha256,
        "corpus_manifest_sha256": input_sha256["MANIFEST.json"],
        "code_tree_sha256": code_tree_sha256,
        "code_files_sha256": code_files,
        "input_sha256": input_sha256,
        "task_count": corpus_manifest["task_count"],
        "cwe_families": corpus_manifest["cwe_families"],
        "development_cwe_families": corpus_manifest["development_cwe_families"],
        "locked_test_cwe_families": corpus_manifest["locked_test_cwe_families"],
    }
    preparation_path = destination / "PREPARATION_MANIFEST.json"
    payload = (json.dumps(preparation, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with preparation_path.open("xb") as handle:
        handle.write(payload)
    return preparation


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create a fresh, hash-attested validator-monoculture G0 corpus."
    )
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-corpus-sha256", required=True)
    parser.add_argument("--expected-code-sha256", required=True)
    args = parser.parse_args(argv)

    manifest = prepare_corpus(
        args.destination,
        args.config,
        expected_config_sha256=args.expected_config_sha256,
        expected_corpus_sha256=args.expected_corpus_sha256,
        expected_code_sha256=args.expected_code_sha256,
    )
    print(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
