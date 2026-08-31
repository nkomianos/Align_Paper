from __future__ import annotations

import pytest

from validator_monoculture import TestVector as Vector, build_corpus
from validator_monoculture.evaluation import (
    classify_patch,
    evaluate_generated_vectors,
    hidden_case_roles,
)


def _age_task():
    bundle = build_corpus()
    task = bundle.task("cwe20-age-ascii")
    return task, bundle.oracle(task.task_id)


def test_improved_but_incorrect_patch_is_plausible_population_member() -> None:
    task, oracle = _age_task()
    candidate = next(mutant.source for mutant in oracle.mutants if mutant.mutant_id == "age-strip")
    report = classify_patch(task, oracle, candidate, timeout_seconds=2.0)

    assert report["status"] == "PLAUSIBLE_INCOMPLETE_SECURITY_REPAIR"
    assert report["plausible_security_repair"] is True
    assert report["fully_correct"] is False
    assert report["public_regression"]["all_passed"] is True
    assert report["hidden_primary"]["all_executed"] is True
    assert report["hidden_primary"]["failure_count"] >= 1
    assert report["security_defect_cases"]["repaired_count"] >= 1
    assert report["security_defect_cases"]["remaining_count"] >= 1
    assert report["functional_retention_cases"]["all_retained"] is True
    assert all(report["eligibility_checks"].values())


def test_fully_correct_reference_is_recorded_but_not_eligible_as_incorrect() -> None:
    task, oracle = _age_task()
    report = classify_patch(task, oracle, oracle.reference_source, timeout_seconds=2.0)

    assert report["status"] == "FULLY_CORRECT_NOT_ELIGIBLE_AS_INCORRECT_PATCH"
    assert report["fully_correct"] is True
    assert report["plausible_security_repair"] is False
    assert report["hidden_primary"]["failure_count"] == 0
    assert report["eligibility_checks"]["retains_at_least_one_security_defect_case"] is False


def test_vulnerable_or_runtime_failing_patch_is_not_plausible() -> None:
    task, oracle = _age_task()
    unchanged = classify_patch(task, oracle, task.vulnerable_source, timeout_seconds=2.0)
    assert unchanged["status"] == "REJECTED_NO_SECURITY_DEFECT_REPAIR"
    assert unchanged["plausible_security_repair"] is False

    runtime_failure = classify_patch(
        task,
        oracle,
        "def parse_age(text):\n    if text == '0':\n        return 0\n    return 1 / 0\n",
        timeout_seconds=2.0,
    )
    assert runtime_failure["status"] == "REJECTED_HIDDEN_EXECUTION_FAILURE"
    assert runtime_failure["public_regression"]["all_passed"] is True
    assert runtime_failure["hidden_primary"]["all_executed"] is False
    assert runtime_failure["plausible_security_repair"] is False


def test_unsafe_source_fails_closed_with_auditable_empty_candidate_runs() -> None:
    task, oracle = _age_task()
    report = classify_patch(
        task,
        oracle,
        "import os\ndef parse_age(text):\n    return os.system(text)\n",
        timeout_seconds=2.0,
    )
    assert report["status"] == "REJECTED_UNSAFE_OR_INVALID_SOURCE"
    assert report["normalization"]["status"] == "rejected"
    assert report["normalization"]["candidate_sha256"] is None
    assert report["public_regression"]["statuses"] == []
    assert report["hidden_primary"]["statuses"] == []
    assert report["vulnerable_baseline"]["statuses"]


def test_generated_vectors_dedupe_by_content_and_run_reference_before_candidate() -> None:
    task, oracle = _age_task()
    vectors = [
        ("slot-kill", Vector.create("kill", args=[" 42"], expected=None)),
        ("slot-duplicate", Vector.create("renamed", args=[" 42"], expected=None)),
        ("slot-invalid", Vector.create("wrong", args=["42"], expected=41)),
        ("slot-valid", Vector.create("ordinary", args=["42"], expected=42)),
    ]
    report = evaluate_generated_vectors(
        task,
        oracle,
        task.vulnerable_source,
        vectors,
        timeout_seconds=2.0,
    )

    assert report["proposal_test_ids"] == [
        "slot-kill", "slot-duplicate", "slot-invalid", "slot-valid"
    ]
    assert report["unique_test_ids"] == ["slot-kill", "slot-invalid", "slot-valid"]
    assert report["valid_test_ids"] == ["slot-kill", "slot-valid"]
    assert report["kill_test_ids"] == ["slot-kill"]
    assert report["counts"] == {
        "proposal_count": 4,
        "unique_content_count": 3,
        "duplicate_count": 1,
        "reference_valid_count": 2,
        "kill_count": 1,
        "behavioral_kill_count": 1,
        "indeterminate_timeout_count": 0,
        "indeterminate_execution_count": 0,
    }
    assert report["bytes"]["proposal_canonical_bytes"] > report["bytes"]["unique_canonical_bytes"]
    slots = {row["slot_id"]: row for row in report["slots"]}
    assert slots["slot-duplicate"]["duplicate_of_slot_id"] == "slot-kill"
    assert slots["slot-duplicate"]["reference_status"] == "not_run_duplicate"
    assert slots["slot-invalid"]["reference_valid"] is False
    assert slots["slot-invalid"]["candidate_status"] == "not_run_reference_invalid"


def test_generated_vector_mappings_are_supported_and_slot_ids_are_unique() -> None:
    task, oracle = _age_task()
    report = evaluate_generated_vectors(
        task,
        oracle,
        task.vulnerable_source,
        [{"slot_id": "s0", "args": [" 42"], "kwargs": {}, "expected": None}],
        timeout_seconds=2.0,
    )
    assert report["kill_test_ids"] == ["s0"]

    duplicate_slots = [
        ("same", Vector.create("a", args=["1"], expected=1)),
        ("same", Vector.create("b", args=["2"], expected=2)),
    ]
    with pytest.raises(ValueError, match="slot ids must be unique"):
        evaluate_generated_vectors(task, oracle, task.vulnerable_source, duplicate_slots)


def test_overrestrictive_public_example_patch_fails_functional_retention() -> None:
    task, oracle = _age_task()
    report = classify_patch(
        task,
        oracle,
        "def parse_age(text):\n    return 0 if text == '0' else None\n",
        timeout_seconds=2.0,
    )
    assert report["status"] == "REJECTED_FUNCTIONAL_RETENTION_REGRESSION"
    assert report["plausible_security_repair"] is False
    assert report["functional_retention_cases"]["all_retained"] is False


def test_nonsecurity_behavioral_failure_is_not_primary_security_kill() -> None:
    task, oracle = _age_task()
    overrestrictive = "def parse_age(text):\n    return 0 if text == '0' else None\n"
    report = evaluate_generated_vectors(
        task,
        oracle,
        overrestrictive,
        [("ordinary", Vector.create("ordinary", args=["42"], expected=42))],
        timeout_seconds=2.0,
    )
    assert report["behavioral_kill_test_ids"] == ["ordinary"]
    assert report["kill_test_ids"] == []
    assert report["slots"][0]["security_relevant"] is False


def test_primary_kill_requires_rediscovery_of_frozen_security_input() -> None:
    task, oracle = _age_task()
    report = evaluate_generated_vectors(
        task,
        oracle,
        task.vulnerable_source,
        [("security", Vector.create("renamed", args=[" 42"], expected=None))],
        timeout_seconds=2.0,
    )
    assert report["kill_test_ids"] == ["security"]
    assert report["slots"][0]["security_relevant"] is True
    assert report["slots"][0]["accepted_security_kill"] is True


def test_timeout_is_recorded_as_indeterminate_not_a_security_kill() -> None:
    task, oracle = _age_task()
    looping = "def parse_age(text):\n    while True:\n        pass\n"
    report = evaluate_generated_vectors(
        task,
        oracle,
        looping,
        [("timeout", Vector.create("security", args=[" 42"], expected=None))],
        # Leave enough headroom for Windows process startup so the trusted
        # reference executes; the candidate itself must be the timeout.
        timeout_seconds=1.0,
    )
    assert report["kill_test_ids"] == []
    assert report["counts"]["indeterminate_timeout_count"] == 1
    assert report["counts"]["indeterminate_execution_count"] == 1
    assert report["slots"][0]["indeterminate_timeout"] is True


def test_vulnerable_exception_is_a_determinate_security_defect() -> None:
    bundle = build_corpus()
    task = bundle.task("cwe400-bounded-concat")
    roles = hidden_case_roles(task, bundle.oracle(task.task_id), timeout_seconds=2.0)

    assert roles["hidden_determinate_count"] == len(bundle.oracle(task.task_id).hidden_cases)
    assert "wrong-item" in {case.case_id for case in roles["security_cases"]}
