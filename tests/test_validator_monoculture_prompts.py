from __future__ import annotations

import pytest

from validator_monoculture.prompts import parse_patch_completion, parse_test_completion
from validator_monoculture.prompts import patch_prompt, verifier_prompt


def test_patch_parser_accepts_one_exact_function() -> None:
    code = parse_patch_completion(
        "```python\ndef sanitize(value, allowed):\n    return value if value in allowed else None\n```",
        entrypoint="sanitize",
        signature="(value, allowed)",
    )
    assert code.startswith("def sanitize")
    assert "```" not in code


def test_patch_parser_canonicalizes_comments_and_formatting() -> None:
    first = parse_patch_completion(
        "def sanitize(value,allowed): # family style\n return value if value in allowed else None",
        entrypoint="sanitize",
        signature="(value, allowed)",
    )
    second = parse_patch_completion(
        "```python\ndef sanitize(value, allowed):\n    return value if value in allowed else None\n```",
        entrypoint="sanitize",
        signature="(value, allowed)",
    )
    assert first == second


def test_patch_parser_removes_docstring_style_channel() -> None:
    with_docstring = parse_patch_completion(
        'def sanitize(value, allowed):\n    """Qwen-style explanation."""\n    return value if value in allowed else None',
        entrypoint="sanitize",
        signature="(value, allowed)",
    )
    without_docstring = parse_patch_completion(
        "def sanitize(value, allowed):\n    return value if value in allowed else None",
        entrypoint="sanitize",
        signature="(value, allowed)",
    )
    assert with_docstring == without_docstring


def test_patch_parser_strips_validated_annotations_before_sandboxing() -> None:
    parsed = parse_patch_completion(
        "def lookup_user(value: str) -> list[object] | None:\n"
        "    return ['SELECT id FROM users WHERE username = ?', [value]]\n",
        entrypoint="lookup_user",
        signature="lookup_user(value: str) -> list[object] | None",
    )
    assert parsed.startswith("def lookup_user(value):")
    assert "object" not in parsed


def test_corpus_style_signature_and_spec_only_prompt_are_supported() -> None:
    task = {
        "task_id": "x",
        "cwe_id": "CWE-20",
        "entrypoint": "sanitize",
        "signature": "sanitize(value: str, allowed: list[str]) -> str | None",
        "public_spec": "Return only allowed values.",
        "vulnerable_source": "def sanitize(value, allowed):\n    return value\n",
    }
    assert "CWE-20" in patch_prompt(task)
    prompt = verifier_prompt(task, None, requested_tests=2)
    assert "not shown the vulnerable implementation" in prompt
    assert "CANDIDATE PATCH" not in prompt
    assert task["vulnerable_source"] not in prompt
    parse_patch_completion(
        "def sanitize(value, allowed):\n    return value if value in allowed else None",
        entrypoint="sanitize",
        signature=task["signature"],
    )


@pytest.mark.parametrize(
    "completion",
    [
        "import os\ndef sanitize(value, allowed): return value",
        "def wrong(value, allowed): return value",
        "def sanitize(value): return value",
        "def sanitize(value, allowed): return value\ndef extra(): pass",
    ],
)
def test_patch_parser_rejects_out_of_contract_code(completion: str) -> None:
    with pytest.raises(ValueError):
        parse_patch_completion(completion, entrypoint="sanitize", signature="(value, allowed)")


def test_test_parser_accepts_strict_vectors() -> None:
    tests = parse_test_completion(
        '{"tests":[{"args":["x",["x"]],"kwargs":{},"expected":"x"}]}',
        requested_tests=1,
    )
    assert tests[0]["expected"] == "x"


def test_test_parser_rejects_prose_or_wrong_cardinality() -> None:
    with pytest.raises(ValueError):
        parse_test_completion("Here are tests: []", requested_tests=1)
    with pytest.raises(ValueError):
        parse_test_completion('{"tests":[]}', requested_tests=1)
