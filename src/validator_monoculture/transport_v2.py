"""Unqualified prospective prompt candidate; does not replace frozen G0 prompts.

The parser remains unchanged. A new experiment must explicitly select this
prompt and freeze its corpus, budget, model revisions and decision rule.
"""
from typing import Any, Mapping

from .prompts import verifier_prompt


def verifier_prompt_v2(
    task: Mapping[str, Any], patch_code: str | None, *, requested_tests: int
) -> str:
    if isinstance(requested_tests, bool) or not isinstance(requested_tests, int) or requested_tests < 1:
        raise ValueError('requested_tests must be a positive integer')
    public = dict(task)
    public['public_cases'] = [
        {key: vector[key] for key in ('args', 'kwargs', 'expected')}
        for vector in task.get('public_cases', [])
    ]
    prompt = verifier_prompt(public, patch_code, requested_tests=requested_tests)
    return prompt + r'''

STRICT OUTPUT SCHEMA:
The outer object has exactly one key: "tests".
Each test object has exactly three keys: "args", "kwargs", "expected".
Do not add case_id, names, explanations, or other metadata.
Use JSON literals true, false, and null; Python True, False, and None are invalid.
Inside JSON strings, encode a literal backslash as \\, a NUL character as \u0000,
and a newline as \n. Do not use \0 or unescaped backslashes.
Write literal data only: no multiplication, concatenation, function calls, or
other expressions. Return the JSON object directly, without code fences.
Before answering, check the exact test count and JSON syntax. Choose inputs and
expected values from this task's contract; syntax instructions supply no answers.
'''
