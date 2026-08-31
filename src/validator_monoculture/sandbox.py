"""Restricted execution and JSON-vector validation for generated patches.

This is a containment layer for a research harness, not a general-purpose
hostile-code sandbox.  It combines a narrow AST language, restricted builtins,
bounded JSON inputs, a fresh spawned process, wall-clock timeout, and POSIX
resource limits when available.  The generated test itself is never code.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import Enum
import json
import math
import multiprocessing
from multiprocessing.connection import Connection
from typing import Mapping

from .schema import (
    JsonValue,
    PrivateOracle,
    PublicTask,
    TestVector,
    _validate_json_value,
    stable_hash,
)


MAX_SOURCE_BYTES = 20_000
MAX_AST_NODES = 1_000
MAX_TEST_JSON_BYTES = 16_384
DEFAULT_TIMEOUT_SECONDS = 1.0


class SandboxViolation(ValueError):
    """Raised when replacement source is outside the permitted subset."""


class ExecutionStatus(str, Enum):
    SUCCESS = "success"
    REJECTED = "rejected"
    EXCEPTION = "exception"
    INVALID_RESULT = "invalid_result"
    TIMEOUT = "timeout"
    CRASHED = "crashed"


@dataclass(frozen=True)
class ExecutionResult:
    status: ExecutionStatus
    value: JsonValue = None
    error_type: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status is ExecutionStatus.SUCCESS


@dataclass(frozen=True)
class TestEvaluation:
    vector: TestVector
    reference_result: ExecutionResult
    candidate_result: ExecutionResult | None
    reference_passed: bool
    kills_candidate: bool

    @property
    def accepted_kill(self) -> bool:
        """True only when the vector passes reference and then kills candidate."""

        return self.reference_passed and self.kills_candidate


@dataclass(frozen=True)
class OracleEvaluation:
    task_id: str
    case_results: tuple[TestEvaluation, ...]

    @property
    def candidate_correct(self) -> bool:
        return all(
            result.reference_passed
            and result.candidate_result is not None
            and result.candidate_result.succeeded
            and not result.kills_candidate
            for result in self.case_results
        )


_FORBIDDEN_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.ClassDef,
    ast.AsyncFunctionDef,
    ast.Global,
    ast.Nonlocal,
    ast.With,
    ast.AsyncWith,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
)
_FORBIDDEN_NAMES = {
    "__import__",
    "breakpoint",
    "bytearray",
    "bytes",
    "classmethod",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "memoryview",
    "object",
    "open",
    "property",
    "setattr",
    "staticmethod",
    "super",
    "type",
    "vars",
}
_SAFE_BUILTINS = {
    "Exception": Exception,
    "AssertionError": AssertionError,
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "ord": ord,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


def normalize_replacement_source(source: str, *, entrypoint: str) -> str:
    """Validate and newline-normalize a single replacement function."""

    if not isinstance(source, str) or not source.strip():
        raise SandboxViolation("replacement source is empty")
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise SandboxViolation(f"replacement source exceeds {MAX_SOURCE_BYTES} bytes")
    if "```" in source:
        raise SandboxViolation("markdown fences are not executable source")
    normalized = source.replace("\r\n", "\n").replace("\r", "\n").strip() + "\n"
    try:
        tree = ast.parse(normalized, mode="exec")
    except SyntaxError as exc:
        raise SandboxViolation(f"replacement source is not valid Python: {exc.msg}") from exc
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise SandboxViolation("source must contain exactly one top-level function")
    function = tree.body[0]
    if function.name != entrypoint:
        raise SandboxViolation(f"replacement function must be named {entrypoint}")
    if function.decorator_list:
        raise SandboxViolation("decorators are not permitted")
    if len(list(ast.walk(tree))) > MAX_AST_NODES:
        raise SandboxViolation(f"replacement source exceeds {MAX_AST_NODES} AST nodes")
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_NODES):
            raise SandboxViolation(f"{type(node).__name__} is not permitted")
        if isinstance(node, ast.FunctionDef) and node is not function:
            raise SandboxViolation("nested functions are not permitted")
        if isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id in _FORBIDDEN_NAMES:
                raise SandboxViolation(f"name {node.id!r} is not permitted")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise SandboxViolation(f"private/dunder attribute {node.attr!r} is not permitted")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_NAMES:
            raise SandboxViolation(f"call to {node.func.id!r} is not permitted")
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str) and len(node.value) > 4096:
                raise SandboxViolation("string literal exceeds 4096 characters")
            if isinstance(node.value, int) and not isinstance(node.value, bool) and abs(node.value) > 10**12:
                raise SandboxViolation("integer literal is outside the supported range")
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Pow, ast.LShift)):
            raise SandboxViolation("power and left-shift operations are not permitted")
    return normalized


def parse_generated_test_vectors(text: str, *, max_vectors: int = 16) -> tuple[TestVector, ...]:
    """Parse one JSON vector or a JSON list of vector objects, never Python."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("generated test JSON is empty")
    if len(text.encode("utf-8")) > MAX_TEST_JSON_BYTES:
        raise ValueError(f"generated test JSON exceeds {MAX_TEST_JSON_BYTES} bytes")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"generated tests are not valid JSON: {exc.msg}") from exc
    if isinstance(payload, dict):
        records = [payload]
    elif isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        records = payload
    else:
        raise ValueError("generated tests must be one object or a list of objects")
    if not records or len(records) > max_vectors:
        raise ValueError(f"generated tests must contain 1..{max_vectors} vectors")
    vectors = tuple(
        TestVector.from_record(record, default_id=f"generated-{index:03d}")
        for index, record in enumerate(records)
    )
    if len({vector.case_id for vector in vectors}) != len(vectors):
        raise ValueError("generated test case ids must be unique")
    return vectors


def _install_resource_limits(timeout_seconds: float) -> None:
    try:
        import resource

        cpu_seconds = max(1, int(math.ceil(timeout_seconds)))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        address_space = 256 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (address_space, address_space))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ImportError, OSError, ValueError):
        # Windows lacks ``resource``; the parent still enforces a hard wall timeout.
        return


def _execution_worker(
    connection: Connection,
    source: str,
    entrypoint: str,
    args: tuple[JsonValue, ...],
    kwargs: tuple[tuple[str, JsonValue], ...],
    timeout_seconds: float,
) -> None:
    try:
        _install_resource_limits(timeout_seconds)
        namespace: dict[str, object] = {"__builtins__": dict(_SAFE_BUILTINS)}
        code = compile(source, "<generated-replacement>", "exec", dont_inherit=True, optimize=0)
        exec(code, namespace, namespace)
        function = namespace[entrypoint]
        value = function(*args, **dict(kwargs))  # type: ignore[operator]
        try:
            _validate_json_value(value, path="result")
        except ValueError as exc:
            connection.send((ExecutionStatus.INVALID_RESULT.value, None, type(exc).__name__))
        else:
            connection.send((ExecutionStatus.SUCCESS.value, value, None))
    except BaseException as exc:
        connection.send((ExecutionStatus.EXCEPTION.value, None, type(exc).__name__))
    finally:
        connection.close()


def execute_function(
    source: str,
    *,
    entrypoint: str,
    vector: TestVector,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> ExecutionResult:
    """Execute one replacement function on one bounded vector in a child."""

    if not 0.05 <= timeout_seconds <= 10.0:
        raise ValueError("timeout_seconds must be between 0.05 and 10.0")
    try:
        normalized = normalize_replacement_source(source, entrypoint=entrypoint)
    except SandboxViolation as exc:
        return ExecutionResult(ExecutionStatus.REJECTED, error_type=type(exc).__name__)

    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_execution_worker,
        args=(sender, normalized, entrypoint, vector.args, vector.kwargs, timeout_seconds),
        daemon=True,
    )
    process.start()
    sender.close()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(1.0)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(1.0)
        receiver.close()
        return ExecutionResult(ExecutionStatus.TIMEOUT)
    if receiver.poll():
        status, value, error_type = receiver.recv()
        receiver.close()
        return ExecutionResult(ExecutionStatus(status), value=value, error_type=error_type)
    receiver.close()
    return ExecutionResult(ExecutionStatus.CRASHED, error_type=f"exitcode:{process.exitcode}")


def _same_json(left: JsonValue, right: JsonValue) -> bool:
    return stable_hash(left) == stable_hash(right)


def validate_generated_test(
    *,
    task: PublicTask,
    oracle: PrivateOracle,
    candidate_source: str,
    vector: TestVector,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> TestEvaluation:
    """Run reference first; only a passing vector may be tried on candidate."""

    if task.task_id != oracle.task_id:
        raise ValueError("task and private oracle ids do not match")
    normalized_reference = normalize_replacement_source(
        oracle.reference_source,
        entrypoint=task.entrypoint,
    )
    import hashlib

    observed_hash = hashlib.sha256(normalized_reference.encode("utf-8")).hexdigest()
    if observed_hash != oracle.reference_sha256:
        raise ValueError("private reference source does not match its frozen hash")
    reference_result = execute_function(
        normalized_reference,
        entrypoint=task.entrypoint,
        vector=vector,
        timeout_seconds=timeout_seconds,
    )
    reference_passed = reference_result.succeeded and _same_json(reference_result.value, vector.expected)
    if not reference_passed:
        return TestEvaluation(
            vector=vector,
            reference_result=reference_result,
            candidate_result=None,
            reference_passed=False,
            kills_candidate=False,
        )
    candidate_result = execute_function(
        candidate_source,
        entrypoint=task.entrypoint,
        vector=vector,
        timeout_seconds=timeout_seconds,
    )
    kills_candidate = not candidate_result.succeeded or not _same_json(candidate_result.value, vector.expected)
    return TestEvaluation(
        vector=vector,
        reference_result=reference_result,
        candidate_result=candidate_result,
        reference_passed=True,
        kills_candidate=kills_candidate,
    )


def evaluate_hidden_oracle(
    *,
    task: PublicTask,
    oracle: PrivateOracle,
    candidate_source: str,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> OracleEvaluation:
    """Evaluate a candidate against every hidden case without revealing cases."""

    results = tuple(
        validate_generated_test(
            task=task,
            oracle=oracle,
            candidate_source=candidate_source,
            vector=vector,
            timeout_seconds=timeout_seconds,
        )
        for vector in oracle.hidden_cases
    )
    return OracleEvaluation(task.task_id, results)
