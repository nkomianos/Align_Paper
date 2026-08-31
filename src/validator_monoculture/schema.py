"""Immutable wire schema for the validator-monoculture security corpus.

The public task and private oracle types are deliberately separate.  A runner
can serialize :class:`PublicTask` without ever loading hidden behavior cases or
mutants into a model-facing process.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Any, Mapping, TypeAlias


JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
SCHEMA_VERSION = "validator-monoculture-g0-v3"


class Split(str, Enum):
    """Development status of a complete CWE family."""

    DEVELOPMENT = "development"
    LOCKED_TEST = "locked_test"


def _validate_json_value(
    value: object,
    *,
    path: str = "value",
    depth: int = 0,
    max_depth: int = 8,
) -> None:
    """Reject non-JSON and pathological values before sandbox transport."""

    if depth > max_depth:
        raise ValueError(f"{path} exceeds maximum nesting depth {max_depth}")
    if value is None or isinstance(value, (bool, int, str)):
        if isinstance(value, str) and len(value) > 4096:
            raise ValueError(f"{path} contains a string longer than 4096 characters")
        if isinstance(value, int) and not isinstance(value, bool) and abs(value) > 10**12:
            raise ValueError(f"{path} contains an integer outside the supported range")
        return
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"{path} contains a non-finite float")
        return
    if isinstance(value, (list, tuple)):
        if len(value) > 64:
            raise ValueError(f"{path} contains more than 64 elements")
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]", depth=depth + 1)
        return
    if isinstance(value, Mapping):
        if len(value) > 64:
            raise ValueError(f"{path} contains more than 64 keys")
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{path} keys must be non-empty strings")
            if len(key) > 256:
                raise ValueError(f"{path} contains a key longer than 256 characters")
            _validate_json_value(item, path=f"{path}.{key}", depth=depth + 1)
        return
    raise ValueError(f"{path} is not JSON-compatible: {type(value).__name__}")


def _jsonable(value: Any) -> JsonValue:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_record"):
        return _jsonable(value.to_record())
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    _validate_json_value(value)
    return value


def canonical_json_bytes(value: object) -> bytes:
    """Return a platform-independent canonical JSON representation."""

    record = _jsonable(value)
    _validate_json_value(record)
    return (
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def stable_hash(value: object) -> str:
    """SHA-256 over the canonical wire representation of ``value``."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True)
class TestVector:
    """A non-executable generated or hidden behavior check."""

    case_id: str
    args: tuple[JsonValue, ...]
    kwargs: tuple[tuple[str, JsonValue], ...]
    expected: JsonValue

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id is required")
        if len(self.args) > 16 or len(self.kwargs) > 16:
            raise ValueError("a test vector may have at most 16 args and 16 kwargs")
        if len({key for key, _ in self.kwargs}) != len(self.kwargs):
            raise ValueError("test-vector keyword names must be unique")
        _validate_json_value(list(self.args), path="args")
        _validate_json_value(dict(self.kwargs), path="kwargs")
        _validate_json_value(self.expected, path="expected")

    @classmethod
    def create(
        cls,
        case_id: str,
        *,
        args: list[JsonValue] | tuple[JsonValue, ...],
        expected: JsonValue,
        kwargs: Mapping[str, JsonValue] | None = None,
    ) -> "TestVector":
        return cls(
            case_id=case_id,
            args=tuple(args),
            kwargs=tuple(sorted((kwargs or {}).items())),
            expected=expected,
        )

    @classmethod
    def from_record(cls, record: Mapping[str, object], *, default_id: str = "generated") -> "TestVector":
        allowed = {"case_id", "args", "kwargs", "expected"}
        extra = set(record) - allowed
        if extra:
            raise ValueError(f"unexpected test-vector fields: {sorted(extra)}")
        if "args" not in record or "expected" not in record:
            raise ValueError("test vector requires args and expected")
        args = record["args"]
        kwargs = record.get("kwargs", {})
        if not isinstance(args, list):
            raise ValueError("test-vector args must be a JSON list")
        if not isinstance(kwargs, dict):
            raise ValueError("test-vector kwargs must be a JSON object")
        case_id = record.get("case_id", default_id)
        if not isinstance(case_id, str):
            raise ValueError("test-vector case_id must be a string")
        return cls.create(case_id, args=args, kwargs=kwargs, expected=record["expected"])

    def to_record(self) -> dict[str, JsonValue]:
        return {
            "case_id": self.case_id,
            "args": list(self.args),
            "kwargs": dict(self.kwargs),
            "expected": self.expected,
        }


@dataclass(frozen=True)
class Mutant:
    """A private, known-defective replacement implementation."""

    mutant_id: str
    defect: str
    source: str

    def __post_init__(self) -> None:
        if not self.mutant_id or not self.defect or not self.source.strip():
            raise ValueError("mutant_id, defect, and source are required")

    def to_record(self) -> dict[str, str]:
        return {"mutant_id": self.mutant_id, "defect": self.defect, "source": self.source}


@dataclass(frozen=True)
class PublicTask:
    """Model-safe task material; contains no hidden oracle information."""

    task_id: str
    cwe_id: str
    cwe_name: str
    split: Split
    entrypoint: str
    signature: str
    public_spec: str
    vulnerable_source: str
    public_cases: tuple[TestVector, ...]

    def __post_init__(self) -> None:
        required = (
            self.task_id,
            self.cwe_id,
            self.cwe_name,
            self.entrypoint,
            self.signature,
            self.public_spec,
            self.vulnerable_source,
        )
        if not all(value.strip() for value in required):
            raise ValueError("all public task fields are required")
        if not self.cwe_id.startswith("CWE-") or not self.cwe_id[4:].isdigit():
            raise ValueError("cwe_id must have the form CWE-N")
        if not self.entrypoint.isidentifier() or self.entrypoint.startswith("_"):
            raise ValueError("entrypoint must be a public Python identifier")
        if not self.public_cases:
            raise ValueError("each public task requires at least one regression vector")
        if len({case.case_id for case in self.public_cases}) != len(self.public_cases):
            raise ValueError("public regression case ids must be unique")

    def patch_prompt_record(self) -> dict[str, JsonValue]:
        """Fields exposed to a patch generator; the answer is intentionally omitted."""

        return {
            "task_id": self.task_id,
            "cwe_id": self.cwe_id,
            "cwe_name": self.cwe_name,
            "entrypoint": self.entrypoint,
            "signature": self.signature,
            "public_spec": self.public_spec,
            "vulnerable_source": self.vulnerable_source,
            "public_cases": [case.to_record() for case in self.public_cases],
            "output_contract": f"Return exactly one replacement function named {self.entrypoint}.",
        }

    def verifier_prompt_record(self, candidate_source: str | None = None) -> dict[str, JsonValue]:
        """Expose public material, optionally plus a candidate for a secondary arm."""

        record: dict[str, JsonValue] = {
            "task_id": self.task_id,
            "cwe_id": self.cwe_id,
            "cwe_name": self.cwe_name,
            "entrypoint": self.entrypoint,
            "signature": self.signature,
            "public_spec": self.public_spec,
            "vulnerable_source": self.vulnerable_source,
            "public_cases": [case.to_record() for case in self.public_cases],
            "output_contract": (
                'Return JSON objects with exactly {"args": [...], "kwargs": {...}, '
                '"expected": <JSON value>}. Do not return Python test code.'
            ),
        }
        if candidate_source is not None:
            if not candidate_source.strip():
                raise ValueError("candidate_source cannot be empty")
            record["candidate_source"] = candidate_source
        return record

    def to_record(self) -> dict[str, JsonValue]:
        return {
            "task_id": self.task_id,
            "cwe_id": self.cwe_id,
            "cwe_name": self.cwe_name,
            "split": self.split.value,
            "entrypoint": self.entrypoint,
            "signature": self.signature,
            "public_spec": self.public_spec,
            "vulnerable_source": self.vulnerable_source,
            "public_cases": [case.to_record() for case in self.public_cases],
        }


@dataclass(frozen=True)
class PrivateOracle:
    """Hidden behavior and mutation oracle for exactly one public task."""

    task_id: str
    reference_source: str
    reference_sha256: str
    primary_case_rule: str
    hidden_cases: tuple[TestVector, ...]
    mutants: tuple[Mutant, ...]

    def __post_init__(self) -> None:
        if not self.task_id or not self.reference_source.strip() or len(self.reference_sha256) != 64:
            raise ValueError("task_id, reference source, and a SHA-256 reference hash are required")
        if not self.primary_case_rule.strip():
            raise ValueError("a primary hidden-case classification rule is required")
        if len(self.hidden_cases) < 3:
            raise ValueError("each private oracle requires at least three hidden cases")
        if len(self.mutants) < 2:
            raise ValueError("each private oracle requires at least two mutants")
        if len({case.case_id for case in self.hidden_cases}) != len(self.hidden_cases):
            raise ValueError("hidden case ids must be unique within a task")
        if len({mutant.mutant_id for mutant in self.mutants}) != len(self.mutants):
            raise ValueError("mutant ids must be unique within a task")

    def to_record(self) -> dict[str, JsonValue]:
        return {
            "task_id": self.task_id,
            "reference_source": self.reference_source,
            "reference_sha256": self.reference_sha256,
            "primary_case_rule": self.primary_case_rule,
            "hidden_cases": [case.to_record() for case in self.hidden_cases],
            "mutants": [mutant.to_record() for mutant in self.mutants],
        }
