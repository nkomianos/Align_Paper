from __future__ import annotations

from pathlib import Path

from validator_monoculture.corpus import write_corpus
from validator_monoculture.oracle_preflight import validate_oracle
from validator_monoculture.serde import bind_corpus, load_private_oracles, load_public_tasks


def test_entire_frozen_oracle_has_plausible_incomplete_controls(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    write_corpus(corpus)
    tasks, oracles = bind_corpus(
        load_public_tasks(corpus / "public" / "tasks.jsonl"),
        load_private_oracles(corpus / "private" / "oracles.jsonl"),
    )
    report = validate_oracle(
        tasks,
        oracles,
        timeout_seconds=2.0,
    )
    assert report["status"] == "PASS"
    assert report["task_count"] == 32
    assert report["mutant_count"] == 64
    assert report["plausible_mutant_count"] >= 32
    assert len(report["plausible_mutants_by_cwe"]) == 8
    assert all(row["plausible_mutant_count"] >= 1 for row in report["tasks"])
    assert all(row["security_defect_case_count"] >= 2 for row in report["tasks"])
    assert all(row["functional_retention_case_count"] >= 1 for row in report["tasks"])
