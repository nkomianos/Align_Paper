"""CPU-only dynamic soundness checks for the frozen security oracle."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .evaluation import classify_patch, hidden_case_roles
from .schema import PrivateOracle, PublicTask
from .serde import bind_corpus, load_private_oracles, load_public_tasks


def validate_oracle(
    tasks: Mapping[str, PublicTask],
    oracles: Mapping[str, PrivateOracle],
    *,
    timeout_seconds: float = 2.0,
) -> dict[str, Any]:
    """Exercise already-bound tasks, references, and every planted mutant.

    Each task must have a correct reference and at least one planted mutant
    that meets the same plausible-but-incomplete population definition used for
    model patches.  This is apparatus validation, never empirical evidence for
    the model-family hypothesis.
    """

    if not tasks or set(tasks) != set(oracles):
        raise ValueError("public tasks and private oracles do not match exactly")
    if any(task_id != task.task_id for task_id, task in tasks.items()) or any(
        task_id != oracle.task_id for task_id, oracle in oracles.items()
    ):
        raise ValueError("bound corpus keys do not match their task IDs")
    status_counts: Counter[str] = Counter()
    per_task: list[dict[str, Any]] = []
    plausible_by_cwe: Counter[str] = Counter()
    for task_id, task in tasks.items():
        oracle = oracles[task_id]
        reference = classify_patch(
            task, oracle, oracle.reference_source, timeout_seconds=timeout_seconds
        )
        if not reference.get("fully_correct") or reference.get("plausible_security_repair"):
            raise ValueError(f"reference implementation fails soundness checks: {task_id}")
        vulnerable = classify_patch(
            task, oracle, task.vulnerable_source, timeout_seconds=timeout_seconds
        )
        if vulnerable.get("fully_correct") or vulnerable.get("plausible_security_repair"):
            raise ValueError(f"vulnerable baseline has the wrong classification: {task_id}")
        roles = hidden_case_roles(task, oracle, timeout_seconds=timeout_seconds)
        security_case_count = len(roles["security_cases"])
        retention_case_count = len(roles["retention_cases"])
        if security_case_count < 2:
            raise ValueError(
                f"task lacks two independently repairable security-defect cases: {task_id}"
            )
        if retention_case_count < 1:
            raise ValueError(
                f"task lacks a hidden functional-retention case: {task_id}"
            )
        mutant_rows: list[dict[str, Any]] = []
        for mutant in oracle.mutants:
            result = classify_patch(
                task, oracle, mutant.source, timeout_seconds=timeout_seconds
            )
            status = str(result.get("status"))
            status_counts[status] += 1
            hidden = result.get("hidden_primary")
            if not isinstance(hidden, dict) or int(hidden.get("failure_count", 0)) < 1:
                raise ValueError(
                    f"planted mutant is not hidden-oracle defective: {task_id}/{mutant.mutant_id}"
                )
            mutant_rows.append({
                "mutant_id": mutant.mutant_id,
                "status": status,
                "plausible_security_repair": result.get("plausible_security_repair") is True,
            })
        plausible_count = sum(row["plausible_security_repair"] for row in mutant_rows)
        if plausible_count < 1:
            raise ValueError(
                f"task lacks a plausible incomplete planted control: {task_id}"
            )
        plausible_by_cwe[task.cwe_id] += plausible_count
        per_task.append({
            "task_id": task_id,
            "cwe_id": task.cwe_id,
            "split": task.split.value,
            "plausible_mutant_count": plausible_count,
            "security_defect_case_count": security_case_count,
            "functional_retention_case_count": retention_case_count,
            "mutants": mutant_rows,
        })
    missing_cwes = sorted(
        cwe for cwe in {task.cwe_id for task in tasks.values()}
        if plausible_by_cwe[cwe] < 1
    )
    if missing_cwes:
        raise ValueError(f"CWE families lack a plausible planted control: {missing_cwes}")
    return {
        "kind": "validator_monoculture_oracle_preflight",
        "status": "PASS",
        "interpretation": "apparatus_validation_only",
        "task_count": len(tasks),
        "mutant_count": sum(len(row["mutants"]) for row in per_task),
        "plausible_mutant_count": sum(row["plausible_mutant_count"] for row in per_task),
        "plausible_mutants_by_cwe": dict(sorted(plausible_by_cwe.items())),
        "mutant_status_counts": dict(sorted(status_counts.items())),
        "tasks": per_task,
    }


def validate_oracle_files(
    public_corpus: str | Path,
    private_oracles: str | Path,
    *,
    timeout_seconds: float = 2.0,
) -> dict[str, Any]:
    """Load corpus paths for the standalone preflight command."""

    tasks, oracles = bind_corpus(
        load_public_tasks(public_corpus), load_private_oracles(private_oracles)
    )
    return validate_oracle(tasks, oracles, timeout_seconds=timeout_seconds)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-corpus", type=Path, required=True)
    parser.add_argument("--private-oracles", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=2.0)
    args = parser.parse_args(argv)
    print(json.dumps(validate_oracle_files(**vars(args)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
