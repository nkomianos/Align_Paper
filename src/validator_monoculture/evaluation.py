"""Fail-closed patch and generated-vector evaluation for the G0 gate."""

from __future__ import annotations

import hashlib
from typing import Mapping, Sequence, TypeAlias

from .sandbox import (
    DEFAULT_TIMEOUT_SECONDS,
    ExecutionResult,
    ExecutionStatus,
    SandboxViolation,
    execute_function,
    normalize_replacement_source,
    validate_generated_test,
)
from .schema import (
    JsonValue,
    PrivateOracle,
    PublicTask,
    TestVector,
    canonical_json_bytes,
    stable_hash,
)


SlottedVector: TypeAlias = tuple[str, TestVector] | Mapping[str, object]


def _same_json(left: JsonValue, right: JsonValue) -> bool:
    return stable_hash(left) == stable_hash(right)


def _check_task_oracle(task: PublicTask, oracle: PrivateOracle) -> None:
    if task.task_id != oracle.task_id:
        raise ValueError("task and private oracle ids do not match")
    reference = normalize_replacement_source(oracle.reference_source, entrypoint=task.entrypoint)
    observed = hashlib.sha256(reference.encode("utf-8")).hexdigest()
    if observed != oracle.reference_sha256:
        raise ValueError("private reference source does not match its frozen hash")


def _case_status(case: TestVector, result: ExecutionResult) -> dict[str, object]:
    matched = result.succeeded and _same_json(result.value, case.expected)
    return {
        "case_id": case.case_id,
        "execution_status": result.status.value,
        "error_type": result.error_type,
        "matched_expected": matched,
    }


def _run_cases(
    *,
    source: str,
    task: PublicTask,
    cases: Sequence[TestVector],
    timeout_seconds: float,
) -> tuple[list[dict[str, object]], int, int]:
    statuses: list[dict[str, object]] = []
    pass_count = 0
    execution_success_count = 0
    for case in cases:
        result = execute_function(
            source,
            entrypoint=task.entrypoint,
            vector=case,
            timeout_seconds=timeout_seconds,
        )
        status = _case_status(case, result)
        statuses.append(status)
        execution_success_count += int(result.succeeded)
        pass_count += int(bool(status["matched_expected"]))
    return statuses, pass_count, execution_success_count


def hidden_case_roles(
    task: PublicTask,
    oracle: PrivateOracle,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, object]:
    """Freeze hidden-case roles from the vulnerable baseline.

    A hidden input is a *security-defect case* exactly when the frozen
    vulnerable implementation disagrees with the correct reference on that
    input.  Inputs the vulnerable implementation already gets right are
    *functional-retention cases*.  This operational partition prevents an
    over-restrictive patch from entering the population merely by fixing a few
    adversarial examples while breaking ordinary behavior.
    """

    _check_task_oracle(task, oracle)
    statuses, pass_count, execution_success_count = _run_cases(
        source=task.vulnerable_source,
        task=task,
        cases=oracle.hidden_cases,
        timeout_seconds=timeout_seconds,
    )
    # A deterministic exception is itself a legitimate vulnerable behavior
    # (for example, a resource-validation bug that crashes on malformed input).
    # Only infrastructure/validation failures make the frozen role ambiguous.
    determinate_count = sum(
        status["execution_status"]
        in {ExecutionStatus.SUCCESS.value, ExecutionStatus.EXCEPTION.value}
        for status in statuses
    )
    if determinate_count != len(oracle.hidden_cases):
        raise ValueError(
            f"vulnerable baseline has an indeterminate hidden case: {task.task_id}"
        )
    matched_by_id = {
        str(status["case_id"]): bool(status["matched_expected"])
        for status in statuses
    }
    security_cases = tuple(
        case for case in oracle.hidden_cases if not matched_by_id[case.case_id]
    )
    retention_cases = tuple(
        case for case in oracle.hidden_cases if matched_by_id[case.case_id]
    )
    return {
        "statuses": statuses,
        "hidden_pass_count": pass_count,
        "hidden_execution_success_count": execution_success_count,
        "hidden_determinate_count": determinate_count,
        "security_cases": security_cases,
        "retention_cases": retention_cases,
    }


def _classification_status(
    *,
    normalized: bool,
    public_all_passed: bool,
    hidden_all_executed: bool,
    fully_correct: bool,
    preserves_functional_retention: bool,
    repaired_security_count: int,
) -> str:
    if not normalized:
        return "REJECTED_UNSAFE_OR_INVALID_SOURCE"
    if not public_all_passed:
        return "REJECTED_PUBLIC_REGRESSION"
    if not hidden_all_executed:
        return "REJECTED_HIDDEN_EXECUTION_FAILURE"
    if fully_correct:
        return "FULLY_CORRECT_NOT_ELIGIBLE_AS_INCORRECT_PATCH"
    if not preserves_functional_retention:
        return "REJECTED_FUNCTIONAL_RETENTION_REGRESSION"
    if repaired_security_count < 1:
        return "REJECTED_NO_SECURITY_DEFECT_REPAIR"
    return "PLAUSIBLE_INCOMPLETE_SECURITY_REPAIR"


def classify_patch(
    task: PublicTask,
    oracle: PrivateOracle,
    candidate_source: str,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, object]:
    """Classify a generated patch for the *incorrect-patch* G0 population.

    Eligibility requires all public regressions to pass, successful execution
    on every hidden case, preservation of every hidden behavior already
    satisfied by the vulnerable implementation, repair of at least one frozen
    vulnerable-baseline defect, and retention of at least one such defect.  A
    fully correct patch is recorded but is not eligible for the experiment
    conditioned on incomplete security repairs.
    """

    _check_task_oracle(task, oracle)
    candidate_input_bytes = len(candidate_source.encode("utf-8")) if isinstance(candidate_source, str) else 0
    try:
        normalized_candidate = normalize_replacement_source(
            candidate_source,
            entrypoint=task.entrypoint,
        )
    except (SandboxViolation, TypeError) as exc:
        normalization_status = "rejected"
        normalization_error = type(exc).__name__
        normalized_candidate = None
    else:
        normalization_status = "accepted"
        normalization_error = None

    baseline = hidden_case_roles(
        task, oracle, timeout_seconds=timeout_seconds
    )
    vulnerable_statuses = list(baseline["statuses"])
    vulnerable_hidden_pass_count = int(baseline["hidden_pass_count"])
    vulnerable_hidden_execution_success_count = int(
        baseline["hidden_execution_success_count"]
    )
    vulnerable_hidden_determinate_count = int(baseline["hidden_determinate_count"])
    security_cases = tuple(baseline["security_cases"])
    retention_cases = tuple(baseline["retention_cases"])

    if normalized_candidate is None:
        public_statuses: list[dict[str, object]] = []
        hidden_statuses: list[dict[str, object]] = []
        public_pass_count = hidden_pass_count = 0
        public_execution_success_count = hidden_execution_success_count = 0
        candidate_sha256 = None
        normalized_bytes = 0
    else:
        candidate_sha256 = hashlib.sha256(normalized_candidate.encode("utf-8")).hexdigest()
        normalized_bytes = len(normalized_candidate.encode("utf-8"))
        public_statuses, public_pass_count, public_execution_success_count = _run_cases(
            source=normalized_candidate,
            task=task,
            cases=task.public_cases,
            timeout_seconds=timeout_seconds,
        )
        hidden_statuses, hidden_pass_count, hidden_execution_success_count = _run_cases(
            source=normalized_candidate,
            task=task,
            cases=oracle.hidden_cases,
            timeout_seconds=timeout_seconds,
        )

    public_total = len(task.public_cases)
    hidden_total = len(oracle.hidden_cases)
    public_all_passed = normalized_candidate is not None and public_pass_count == public_total
    hidden_all_executed = (
        normalized_candidate is not None and hidden_execution_success_count == hidden_total
    )
    fully_correct = (
        public_all_passed
        and hidden_all_executed
        and hidden_pass_count == hidden_total
    )
    primary_failure_count = hidden_total - hidden_pass_count if normalized_candidate is not None else hidden_total
    hidden_matches = {
        str(status["case_id"]): bool(status["matched_expected"])
        for status in hidden_statuses
    }
    repaired_security_count = sum(
        bool(hidden_matches.get(case.case_id, False)) for case in security_cases
    )
    remaining_security_defect_count = len(security_cases) - repaired_security_count
    retained_functional_count = sum(
        bool(hidden_matches.get(case.case_id, False)) for case in retention_cases
    )
    preserves_functional_retention = (
        normalized_candidate is not None
        and retained_functional_count == len(retention_cases)
    )
    plausible_security_repair = (
        public_all_passed
        and hidden_all_executed
        and preserves_functional_retention
        and repaired_security_count >= 1
        and remaining_security_defect_count >= 1
    )
    status = _classification_status(
        normalized=normalized_candidate is not None,
        public_all_passed=public_all_passed,
        hidden_all_executed=hidden_all_executed,
        fully_correct=fully_correct,
        preserves_functional_retention=preserves_functional_retention,
        repaired_security_count=repaired_security_count,
    )
    return {
        "schema_version": "validator-monoculture-patch-classification-v2",
        "task_id": task.task_id,
        "cwe_id": task.cwe_id,
        "split": task.split.value,
        "status": status,
        "plausible_security_repair": plausible_security_repair,
        "fully_correct": fully_correct,
        "normalization": {
            "status": normalization_status,
            "error_type": normalization_error,
            "input_bytes": candidate_input_bytes,
            "normalized_bytes": normalized_bytes,
            "candidate_sha256": candidate_sha256,
        },
        "public_regression": {
            "total": public_total,
            "pass_count": public_pass_count,
            "execution_success_count": public_execution_success_count,
            "all_passed": public_all_passed,
            "statuses": public_statuses,
        },
        "hidden_primary": {
            "rule": oracle.primary_case_rule,
            "total": hidden_total,
            "pass_count": hidden_pass_count,
            "failure_count": primary_failure_count,
            "execution_success_count": hidden_execution_success_count,
            "all_executed": hidden_all_executed,
            "statuses": hidden_statuses,
        },
        "security_defect_cases": {
            "definition": "hidden inputs failed by the frozen vulnerable baseline",
            "total": len(security_cases),
            "repaired_count": repaired_security_count,
            "remaining_count": remaining_security_defect_count,
            "case_ids": [case.case_id for case in security_cases],
        },
        "functional_retention_cases": {
            "definition": "hidden inputs passed by the frozen vulnerable baseline",
            "total": len(retention_cases),
            "retained_count": retained_functional_count,
            "all_retained": preserves_functional_retention,
            "case_ids": [case.case_id for case in retention_cases],
        },
        "vulnerable_baseline": {
            "hidden_pass_count": vulnerable_hidden_pass_count,
            "hidden_execution_success_count": vulnerable_hidden_execution_success_count,
            "hidden_determinate_count": vulnerable_hidden_determinate_count,
            "statuses": vulnerable_statuses,
        },
        "eligibility_checks": {
            "normalized_safely": normalized_candidate is not None,
            "passes_every_public_regression": public_all_passed,
            "executes_successfully_on_every_hidden": hidden_all_executed,
            "preserves_every_functional_retention_case": preserves_functional_retention,
            "repairs_at_least_one_security_defect_case": repaired_security_count >= 1,
            "retains_at_least_one_security_defect_case": remaining_security_defect_count >= 1,
        },
    }


def _coerce_slotted_vector(item: SlottedVector, *, index: int) -> tuple[str, TestVector]:
    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], TestVector):
        slot_id, vector = item
    elif isinstance(item, Mapping):
        if "slot_id" not in item or not isinstance(item["slot_id"], str):
            raise ValueError("each generated vector mapping requires a string slot_id")
        slot_id = item["slot_id"]
        record = {key: value for key, value in item.items() if key != "slot_id"}
        vector = TestVector.from_record(record, default_id=f"generated-{index:03d}")
    else:
        raise ValueError("generated vectors must be (slot_id, TestVector) pairs or mappings")
    if not slot_id:
        raise ValueError("generated-vector slot_id cannot be empty")
    return slot_id, vector


def _vector_content(vector: TestVector) -> dict[str, JsonValue]:
    """Semantic content for dedupe; model- or slot-specific ids are excluded."""

    return {
        "args": list(vector.args),
        "kwargs": dict(vector.kwargs),
        "expected": vector.expected,
    }


def evaluate_generated_vectors(
    task: PublicTask,
    oracle: PrivateOracle,
    candidate_source: str,
    vectors: Sequence[SlottedVector],
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, object]:
    """Deduplicate and evaluate generated tests with reference-first ordering."""

    _check_task_oracle(task, oracle)
    slotted = [_coerce_slotted_vector(item, index=index) for index, item in enumerate(vectors)]
    slot_ids = [slot_id for slot_id, _ in slotted]
    if len(slot_ids) != len(set(slot_ids)):
        raise ValueError("generated-vector slot ids must be unique")

    seen: dict[str, str] = {}
    logs: list[dict[str, object]] = []
    proposal_bytes = 0
    unique_bytes = 0
    valid_test_ids: list[str] = []
    kill_test_ids: list[str] = []
    behavioral_kill_test_ids: list[str] = []
    unique_test_ids: list[str] = []
    for slot_id, vector in slotted:
        content = _vector_content(vector)
        encoded = canonical_json_bytes(content)
        content_sha256 = hashlib.sha256(encoded).hexdigest()
        byte_count = len(encoded)
        proposal_bytes += byte_count
        duplicate_of = seen.get(content_sha256)
        if duplicate_of is not None:
            logs.append({
                "slot_id": slot_id,
                "case_id": vector.case_id,
                "content_sha256": content_sha256,
                "canonical_bytes": byte_count,
                "duplicate_of_slot_id": duplicate_of,
                "reference_valid": None,
                "kills_candidate": None,
                "security_relevant": None,
                "accepted_security_kill": None,
                "indeterminate_timeout": None,
                "indeterminate_execution": None,
                "reference_status": "not_run_duplicate",
                "vulnerable_baseline_status": "not_run_duplicate",
                "candidate_status": "not_run_duplicate",
            })
            continue
        seen[content_sha256] = slot_id
        unique_test_ids.append(slot_id)
        unique_bytes += byte_count
        evaluation = validate_generated_test(
            task=task,
            oracle=oracle,
            candidate_source=candidate_source,
            vector=vector,
            timeout_seconds=timeout_seconds,
        )
        if evaluation.reference_passed:
            valid_test_ids.append(slot_id)
        vulnerable_result = None
        if evaluation.reference_passed:
            vulnerable_result = execute_function(
                task.vulnerable_source,
                entrypoint=task.entrypoint,
                vector=vector,
                timeout_seconds=timeout_seconds,
            )
        security_relevant = bool(
            vulnerable_result is not None
            and (
                not vulnerable_result.succeeded
                or not _same_json(vulnerable_result.value, vector.expected)
            )
        )
        if evaluation.accepted_kill:
            behavioral_kill_test_ids.append(slot_id)
        anomalous_statuses = {
            ExecutionStatus.REJECTED,
            ExecutionStatus.INVALID_RESULT,
            ExecutionStatus.TIMEOUT,
            ExecutionStatus.CRASHED,
        }
        execution_results = (
            evaluation.reference_result,
            vulnerable_result,
            evaluation.candidate_result,
        )
        indeterminate_timeout = any(
            result is not None and result.status is ExecutionStatus.TIMEOUT
            for result in execution_results
        )
        indeterminate_execution = any(
            result is not None and result.status in anomalous_statuses
            for result in execution_results
        )
        accepted_security_kill = (
            evaluation.accepted_kill
            and security_relevant
            and not indeterminate_execution
        )
        if accepted_security_kill:
            kill_test_ids.append(slot_id)
        logs.append({
            "slot_id": slot_id,
            "case_id": vector.case_id,
            "content_sha256": content_sha256,
            "canonical_bytes": byte_count,
            "duplicate_of_slot_id": None,
            "reference_valid": evaluation.reference_passed,
            "kills_candidate": evaluation.kills_candidate,
            "security_relevant": security_relevant,
            "accepted_security_kill": accepted_security_kill,
            "indeterminate_timeout": indeterminate_timeout,
            "indeterminate_execution": indeterminate_execution,
            "reference_status": evaluation.reference_result.status.value,
            "vulnerable_baseline_status": (
                vulnerable_result.status.value
                if vulnerable_result is not None
                else "not_run_reference_invalid"
            ),
            "candidate_status": (
                evaluation.candidate_result.status.value
                if evaluation.candidate_result is not None
                else "not_run_reference_invalid"
            ),
        })

    return {
        "schema_version": "validator-monoculture-generated-vector-evaluation-v3",
        "task_id": task.task_id,
        "proposal_test_ids": slot_ids,
        "unique_test_ids": unique_test_ids,
        "valid_test_ids": valid_test_ids,
        "kill_test_ids": kill_test_ids,
        "behavioral_kill_test_ids": behavioral_kill_test_ids,
        "counts": {
            "proposal_count": len(slotted),
            "unique_content_count": len(unique_test_ids),
            "duplicate_count": len(slotted) - len(unique_test_ids),
            "reference_valid_count": len(valid_test_ids),
            "kill_count": len(kill_test_ids),
            "behavioral_kill_count": len(behavioral_kill_test_ids),
            "indeterminate_timeout_count": sum(
                bool(row.get("indeterminate_timeout")) for row in logs
            ),
            "indeterminate_execution_count": sum(
                bool(row.get("indeterminate_execution")) for row in logs
            ),
        },
        "bytes": {
            "candidate_source_bytes": len(candidate_source.encode("utf-8")),
            "proposal_canonical_bytes": proposal_bytes,
            "unique_canonical_bytes": unique_bytes,
        },
        "slots": logs,
    }
