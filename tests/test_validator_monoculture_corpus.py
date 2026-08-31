from __future__ import annotations

import hashlib
import json

import pytest

from validator_monoculture import (
    ExecutionStatus,
    SandboxViolation,
    Split,
    TestVector as Vector,
    build_corpus,
    execute_function,
    normalize_replacement_source,
    parse_generated_test_vectors,
    stable_hash,
    validate_generated_test,
    write_corpus,
)


def test_frozen_corpus_has_balanced_whole_cwe_split() -> None:
    bundle = build_corpus()
    assert len(bundle.public_tasks) == 32
    counts: dict[str, int] = {}
    family_splits: dict[str, set[Split]] = {}
    for task in bundle.public_tasks:
        counts[task.cwe_id] = counts.get(task.cwe_id, 0) + 1
        family_splits.setdefault(task.cwe_id, set()).add(task.split)
    assert len(counts) == 8
    assert set(counts.values()) == {4}
    assert all(len(splits) == 1 for splits in family_splits.values())
    assert sum(next(iter(splits)) is Split.DEVELOPMENT for splits in family_splits.values()) == 4
    assert sum(next(iter(splits)) is Split.LOCKED_TEST for splits in family_splits.values()) == 4


def test_public_and_prompt_records_never_expose_private_oracle() -> None:
    bundle = build_corpus()
    task = bundle.task("cwe20-age-ascii")
    oracle = bundle.oracle(task.task_id)
    public_record = task.to_record()
    patch_prompt = task.patch_prompt_record()
    specification_only = task.verifier_prompt_record()
    candidate = "def parse_age(text):\n    return None\n"
    verifier_prompt = task.verifier_prompt_record(candidate)

    for record in (public_record, patch_prompt, specification_only, verifier_prompt):
        assert "reference_source" not in record
        assert "hidden_cases" not in record
        assert "mutants" not in record
        serialized = json.dumps(record, sort_keys=True)
        assert oracle.reference_sha256 not in serialized
        assert oracle.mutants[0].source not in serialized
    assert "reference_source" in oracle.to_record()
    assert "candidate_source" not in specification_only
    assert public_record["public_cases"]
    assert verifier_prompt["candidate_source"] == candidate
    assert verifier_prompt["vulnerable_source"] == task.vulnerable_source


def test_stable_hashes_and_exclusive_non_overwriting_write(tmp_path) -> None:
    first = build_corpus()
    second = build_corpus()
    assert first.corpus_sha256 == second.corpus_sha256
    assert stable_hash([task.to_record() for task in first.public_tasks]) == stable_hash(
        [task.to_record() for task in second.public_tasks]
    )

    destination = tmp_path / "frozen"
    manifest = write_corpus(destination)
    public_bytes = (destination / "public" / "tasks.jsonl").read_bytes()
    private_bytes = (destination / "private" / "oracles.jsonl").read_bytes()
    assert manifest["public_tasks_sha256"] == hashlib.sha256(public_bytes).hexdigest()
    assert manifest["private_oracles_sha256"] == hashlib.sha256(private_bytes).hexdigest()
    assert json.loads((destination / "MANIFEST.json").read_text(encoding="utf-8")) == manifest
    assert b"reference_source" not in public_bytes
    assert b"hidden_cases" not in public_bytes
    assert b"mutants" not in public_bytes
    assert b"reference_source" in private_bytes
    with pytest.raises(FileExistsError):
        write_corpus(destination)


def test_generated_tests_are_strict_bounded_json_vectors() -> None:
    vectors = parse_generated_test_vectors(
        '[{"case_id":"a","args":["7"],"kwargs":{},"expected":7},'
        '{"case_id":"b","args":["bad"],"kwargs":{},"expected":null}]'
    )
    assert [vector.case_id for vector in vectors] == ["a", "b"]
    assert vectors[0].expected == 7

    with pytest.raises(ValueError, match="valid JSON"):
        parse_generated_test_vectors("import os")
    with pytest.raises(ValueError, match="unexpected"):
        parse_generated_test_vectors('{"args":[],"expected":null,"python":"open(\'x\')"}')
    with pytest.raises(ValueError, match="unique"):
        parse_generated_test_vectors(
            '[{"case_id":"same","args":[],"expected":null},'
            '{"case_id":"same","args":[],"expected":null}]'
        )


@pytest.mark.parametrize(
    "source",
    [
        "import os\ndef f(x):\n    return x\n",
        "def f(x):\n    return open(x).read()\n",
        "def f(x):\n    return x.__class__\n",
        "def f(x):\n    return __import__('subprocess')\n",
        "def f(x):\n    return eval(x)\n",
    ],
)
def test_replacement_language_blocks_import_filesystem_subprocess_and_introspection(source: str) -> None:
    with pytest.raises(SandboxViolation):
        normalize_replacement_source(source, entrypoint="f")


def test_generated_vector_must_pass_reference_before_it_can_kill() -> None:
    bundle = build_corpus()
    task = bundle.task("cwe20-age-ascii")
    oracle = bundle.oracle(task.task_id)
    vector = Vector.create("generated-space", args=[" 42"], expected=None)
    result = validate_generated_test(
        task=task,
        oracle=oracle,
        candidate_source=task.vulnerable_source,
        vector=vector,
        timeout_seconds=2.0,
    )
    assert result.reference_passed
    assert result.kills_candidate
    assert result.accepted_kill
    assert result.candidate_result is not None
    assert result.candidate_result.value == 42

    wrong_expectation = Vector.create("wrong-answer", args=["42"], expected=41)
    rejected = validate_generated_test(
        task=task,
        oracle=oracle,
        candidate_source="def parse_age(text):\n    while True:\n        pass\n",
        vector=wrong_expectation,
        timeout_seconds=0.1,
    )
    assert not rejected.reference_passed
    assert rejected.candidate_result is None
    assert not rejected.kills_candidate


def test_wall_timeout_terminates_nonterminating_patch() -> None:
    vector = Vector.create("loop", args=["1"], expected=1)
    result = execute_function(
        "def parse_age(text):\n    while True:\n        pass\n",
        entrypoint="parse_age",
        vector=vector,
        timeout_seconds=0.1,
    )
    assert result.status is ExecutionStatus.TIMEOUT


def test_every_frozen_source_is_statically_accepted_and_reference_hashes_match() -> None:
    bundle = build_corpus()
    for task in bundle.public_tasks:
        oracle = bundle.oracle(task.task_id)
        reference = normalize_replacement_source(oracle.reference_source, entrypoint=task.entrypoint)
        assert hashlib.sha256(reference.encode("utf-8")).hexdigest() == oracle.reference_sha256
        normalize_replacement_source(task.vulnerable_source, entrypoint=task.entrypoint)
        for mutant in oracle.mutants:
            normalize_replacement_source(mutant.source, entrypoint=task.entrypoint)


def test_one_task_per_cwe_has_sound_hidden_oracle_and_killable_mutants() -> None:
    bundle = build_corpus()
    samples = {}
    for task in bundle.public_tasks:
        samples.setdefault(task.cwe_id, task)
    assert len(samples) == 8

    for task in samples.values():
        oracle = bundle.oracle(task.task_id)
        for vector in oracle.hidden_cases:
            reference = execute_function(
                oracle.reference_source,
                entrypoint=task.entrypoint,
                vector=vector,
                timeout_seconds=2.0,
            )
            assert reference.status is ExecutionStatus.SUCCESS, (task.task_id, vector.case_id, reference)
            assert stable_hash(reference.value) == stable_hash(vector.expected), (task.task_id, vector.case_id)
        for mutant in oracle.mutants:
            killed = False
            for vector in oracle.hidden_cases:
                result = execute_function(
                    mutant.source,
                    entrypoint=task.entrypoint,
                    vector=vector,
                    timeout_seconds=2.0,
                )
                if result.status is not ExecutionStatus.SUCCESS or stable_hash(result.value) != stable_hash(vector.expected):
                    killed = True
                    break
            assert killed, (task.task_id, mutant.mutant_id)
