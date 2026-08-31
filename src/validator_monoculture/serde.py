"""Fail-closed JSONL deserialization for frozen corpus artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .schema import Mutant, PrivateOracle, PublicTask, Split, TestVector


def _exact(record: Mapping[str, Any], fields: set[str], *, kind: str) -> None:
    if set(record) != fields:
        raise ValueError(f"{kind} fields differ from the frozen schema: {sorted(set(record) ^ fields)}")


def _deserialize_jsonl(payload: bytes, *, label: str) -> tuple[dict[str, Any], ...]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSONL in {label} at line {line_number}"
            ) from exc
        if not isinstance(record, dict):
            raise ValueError(
                f"JSONL record in {label} at line {line_number} is not an object"
            )
        records.append(record)
    return tuple(records)


def deserialize_public_tasks(payload: bytes) -> tuple[PublicTask, ...]:
    """Deserialize public tasks from exactly the supplied frozen bytes."""

    tasks: list[PublicTask] = []
    fields = {
        "task_id", "cwe_id", "cwe_name", "split", "entrypoint", "signature",
        "public_spec", "vulnerable_source", "public_cases",
    }
    for record in _deserialize_jsonl(payload, label="public corpus"):
        _exact(record, fields, kind="public task")
        cases = record["public_cases"]
        if not isinstance(cases, list):
            raise ValueError("public_cases must be a list")
        tasks.append(PublicTask(
            task_id=str(record["task_id"]),
            cwe_id=str(record["cwe_id"]),
            cwe_name=str(record["cwe_name"]),
            split=Split(str(record["split"])),
            entrypoint=str(record["entrypoint"]),
            signature=str(record["signature"]),
            public_spec=str(record["public_spec"]),
            vulnerable_source=str(record["vulnerable_source"]),
            public_cases=tuple(TestVector.from_record(case) for case in cases),
        ))
    if len({task.task_id for task in tasks}) != len(tasks) or not tasks:
        raise ValueError("public corpus is empty or contains duplicate task IDs")
    cwe_splits: dict[str, Split] = {}
    for task in tasks:
        old = cwe_splits.setdefault(task.cwe_id, task.split)
        if old is not task.split:
            raise ValueError("a CWE family crosses the frozen split")
    return tuple(tasks)


def deserialize_private_oracles(payload: bytes) -> tuple[PrivateOracle, ...]:
    """Deserialize private oracles from exactly the supplied frozen bytes."""

    oracles: list[PrivateOracle] = []
    fields = {
        "task_id", "reference_source", "reference_sha256", "primary_case_rule",
        "hidden_cases", "mutants",
    }
    for record in _deserialize_jsonl(payload, label="private oracle corpus"):
        _exact(record, fields, kind="private oracle")
        cases, mutants = record["hidden_cases"], record["mutants"]
        if not isinstance(cases, list) or not isinstance(mutants, list):
            raise ValueError("private hidden cases and mutants must be lists")
        parsed_mutants: list[Mutant] = []
        for mutant in mutants:
            if not isinstance(mutant, dict):
                raise ValueError("mutant record must be an object")
            _exact(mutant, {"mutant_id", "defect", "source"}, kind="mutant")
            parsed_mutants.append(Mutant(
                mutant_id=str(mutant["mutant_id"]),
                defect=str(mutant["defect"]),
                source=str(mutant["source"]),
            ))
        oracles.append(PrivateOracle(
            task_id=str(record["task_id"]),
            reference_source=str(record["reference_source"]),
            reference_sha256=str(record["reference_sha256"]),
            primary_case_rule=str(record["primary_case_rule"]),
            hidden_cases=tuple(TestVector.from_record(case) for case in cases),
            mutants=tuple(parsed_mutants),
        ))
    if len({oracle.task_id for oracle in oracles}) != len(oracles) or not oracles:
        raise ValueError("private corpus is empty or contains duplicate task IDs")
    return tuple(oracles)


def load_public_tasks(path: str | Path) -> tuple[PublicTask, ...]:
    return deserialize_public_tasks(Path(path).read_bytes())


def load_private_oracles(path: str | Path) -> tuple[PrivateOracle, ...]:
    return deserialize_private_oracles(Path(path).read_bytes())


def bind_corpus(
    tasks: tuple[PublicTask, ...], oracles: tuple[PrivateOracle, ...]
) -> tuple[dict[str, PublicTask], dict[str, PrivateOracle]]:
    public = {task.task_id: task for task in tasks}
    private = {oracle.task_id: oracle for oracle in oracles}
    if set(public) != set(private):
        raise ValueError("public tasks and private oracles do not match exactly")
    return public, private
