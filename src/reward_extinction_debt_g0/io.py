"""Fail-closed serialization and artifact hashing helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def strict_loads(text: str) -> Any:
    def object_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(text, object_pairs_hook=object_hook, parse_constant=reject_constant)


def json_from_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = strict_loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not a strict UTF-8 JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def jsonl_from_bytes(payload: bytes, *, label: str) -> list[dict[str, Any]]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8") from exc
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = strict_loads(line)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid JSONL in {label} at line {number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"non-object JSONL row in {label} at line {number}")
        rows.append(row)
    return rows


def jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_bytes(path: str | Path, payload: bytes, *, overwrite: bool = False) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, target)
        else:
            try:
                os.link(temporary, target)
            except FileExistsError as exc:
                raise FileExistsError(f"refusing to overwrite {target}") from exc
            os.unlink(temporary)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_json(path: str | Path, value: Any, *, overwrite: bool = False) -> None:
    atomic_bytes(path, canonical_bytes(value), overwrite=overwrite)


def atomic_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_bytes(path, jsonl_bytes(rows), overwrite=False)


def tree_inventory(root: str | Path) -> dict[str, str]:
    directory = Path(root).resolve()
    return {
        path.relative_to(directory).as_posix(): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def code_inventory(root: str | Path) -> tuple[dict[str, str], str]:
    directory = Path(root).resolve()
    rows = {
        path.relative_to(directory).as_posix(): sha256_file(path)
        for path in sorted(directory.rglob("*.py"))
        if path.is_file()
    }
    return rows, sha256_bytes(canonical_bytes(rows))
