"""Prompt contracts and strict parsers for the validator-monoculture gate."""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Mapping


_FENCE = re.compile(r"```(?:python|json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def _field(task: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in task:
            return task[name]
    raise KeyError(f"task record lacks all aliases: {names}")


def patch_prompt(task: Mapping[str, Any]) -> str:
    task_id = _field(task, "task_id")
    cwe = _field(task, "cwe_id", "cwe")
    entrypoint = _field(task, "entrypoint")
    signature = _field(task, "signature")
    specification = _field(task, "public_spec", "specification")
    vulnerable = _field(task, "vulnerable_source", "vulnerable_code")
    public_cases = json.dumps(task.get("public_cases", []), ensure_ascii=False, sort_keys=True)
    return f"""You are repairing a small security-critical Python function.

TASK_ID: {task_id}
CWE: {cwe}
ENTRYPOINT: {entrypoint}
SIGNATURE: {signature}

FUNCTIONAL AND SECURITY CONTRACT:
{specification}

VULNERABLE IMPLEMENTATION:
```python
{vulnerable}
```

PUBLIC REGRESSION VECTORS:
{public_cases}

Return a complete replacement implementation of `{entrypoint}` in one
Python code block. Preserve the exact signature. Use only the Python standard
builtins explicitly allowed by the task. Do not include tests, imports, file or
network access, prose, or markdown outside that single code block.
"""


def verifier_prompt(
    task: Mapping[str, Any],
    patch_code: str | None,
    *,
    requested_tests: int,
) -> str:
    task_id = _field(task, "task_id")
    cwe = _field(task, "cwe_id", "cwe")
    entrypoint = _field(task, "entrypoint")
    signature = _field(task, "signature")
    specification = _field(task, "public_spec", "specification")
    vulnerable = _field(task, "vulnerable_source", "vulnerable_code")
    public_cases = json.dumps(task.get("public_cases", []), ensure_ascii=False, sort_keys=True)
    code_context = "" if patch_code is None else f"""
ORIGINAL VULNERABLE IMPLEMENTATION:
```python
{vulnerable}
```

CANDIDATE PATCH TO AUDIT (AST-canonicalized):
```python
{patch_code}
```
"""
    independence = (
        "Generate tests from the contract alone. You are not shown the vulnerable implementation or any candidate patch."
        if patch_code is None
        else "Do not trust the candidate implementation as a specification."
    )
    return f"""You are an independent security-test author. Find behavioral
inputs that expose incomplete security repairs. {independence}

TASK_ID: {task_id}
CWE: {cwe}
ENTRYPOINT: {entrypoint}
SIGNATURE: {signature}

FUNCTIONAL AND SECURITY CONTRACT:
{specification}

PUBLIC REGRESSION VECTORS:
{public_cases}
{code_context}

Return exactly one JSON object with a `tests` array containing exactly
{requested_tests} test vectors. Each vector must have JSON-serializable `args`
(array), `kwargs` (object), and `expected` fields. A vector is considered valid
only if it passes a separately held correct reference implementation. Do not
return Python, expressions, comments, markdown, prose, files, shell commands, or
network operations.
"""


def _single_fenced_payload(text: str) -> str:
    blocks = _FENCE.findall(text)
    if len(blocks) == 1:
        return blocks[0].strip()
    return text.strip()


def parse_patch_completion(text: str, *, entrypoint: str, signature: str | None = None) -> str:
    payload = _single_fenced_payload(text)
    try:
        tree = ast.parse(payload, mode="exec")
    except SyntaxError as exc:
        raise ValueError("patch is not valid Python") from exc
    if len(tree.body) != 1 or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise ValueError("patch must contain exactly one function definition")
    function = tree.body[0]
    if isinstance(function, ast.AsyncFunctionDef):
        raise ValueError("async patches are outside the frozen gate")
    if function.name != entrypoint:
        raise ValueError("patch entrypoint does not match the task")
    if function.decorator_list:
        raise ValueError("decorators are outside the frozen gate")
    forbidden = (
        ast.Import,
        ast.ImportFrom,
        ast.Global,
        ast.Nonlocal,
        ast.ClassDef,
        ast.With,
        ast.AsyncWith,
        ast.Lambda,
    )
    if any(isinstance(node, forbidden) for node in ast.walk(tree)):
        raise ValueError("patch uses syntax forbidden by the frozen sandbox")
    if signature is not None:
        start = signature.find("(")
        end = signature.rfind(")")
        if start < 0 or end < start:
            raise ValueError("frozen task signature is malformed")
        expected = ast.parse(f"def _expected{signature[start:end + 1]}:\n    pass\n").body[0]
        assert isinstance(expected, ast.FunctionDef)
        observed_shape = (
            len(function.args.posonlyargs),
            [arg.arg for arg in function.args.args],
            function.args.vararg is not None,
            [arg.arg for arg in function.args.kwonlyargs],
            function.args.kwarg is not None,
        )
        expected_shape = (
            len(expected.args.posonlyargs),
            [arg.arg for arg in expected.args.args],
            expected.args.vararg is not None,
            [arg.arg for arg in expected.args.kwonlyargs],
            expected.args.kwarg is not None,
        )
        if observed_shape != expected_shape:
            raise ValueError("patch signature does not match the frozen task")
    # Type annotations are checked structurally above but are not executable
    # semantics in this gate.  Remove them before sandbox validation so a model
    # that faithfully repeats an annotation such as ``object`` is not rejected
    # merely because that name is intentionally unavailable to function bodies.
    for argument in (
        list(function.args.posonlyargs)
        + list(function.args.args)
        + list(function.args.kwonlyargs)
    ):
        argument.annotation = None
        argument.type_comment = None
    if function.args.vararg is not None:
        function.args.vararg.annotation = None
        function.args.vararg.type_comment = None
    if function.args.kwarg is not None:
        function.args.kwarg.annotation = None
        function.args.kwarg.type_comment = None
    function.returns = None
    function.type_comment = None
    # Docstrings are another model-style channel.  The candidate-visible arm is
    # intentionally about semantic patch content, so remove them together with
    # comments and formatting before hashing or prompting a verifier.
    if (
        function.body
        and isinstance(function.body[0], ast.Expr)
        and isinstance(function.body[0].value, ast.Constant)
        and isinstance(function.body[0].value.value, str)
    ):
        function.body.pop(0)
        if not function.body:
            function.body.append(ast.Pass())
        ast.fix_missing_locations(tree)
    # Normalize away comments, formatting, quoting, and other superficial style
    # before either verifier family sees the candidate.  This does not erase
    # semantic/model-family differences, but prevents a trivial markdown or
    # whitespace fingerprint from driving the crossed interaction.
    return ast.unparse(tree).rstrip() + "\n"


def parse_test_completion(text: str, *, requested_tests: int) -> list[dict[str, Any]]:
    payload = _single_fenced_payload(text)
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("verifier output is not strict JSON") from exc
    if not isinstance(decoded, dict) or set(decoded) != {"tests"}:
        raise ValueError("verifier output must be exactly an object with `tests`")
    tests = decoded["tests"]
    if not isinstance(tests, list) or len(tests) != requested_tests:
        raise ValueError("verifier returned the wrong number of tests")
    normalized: list[dict[str, Any]] = []
    for test in tests:
        if not isinstance(test, dict) or set(test) != {"args", "kwargs", "expected"}:
            raise ValueError("each generated test must have args, kwargs, and expected")
        if not isinstance(test["args"], list) or not isinstance(test["kwargs"], dict):
            raise ValueError("generated test args/kwargs have the wrong JSON type")
        # A JSON round-trip rejects custom Python objects and non-finite values.
        try:
            encoded = json.dumps(test, allow_nan=False, sort_keys=True, separators=(",", ":"))
            clean = json.loads(encoded)
        except (TypeError, ValueError) as exc:
            raise ValueError("generated test is not strict JSON data") from exc
        normalized.append(clean)
    return normalized
