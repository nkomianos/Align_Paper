from __future__ import annotations

import pytest

from semantic_ancestry_rag.gate import Conditions, ResultRow, Thresholds, evaluate_gate
from semantic_ancestry_rag.runner import Question, score_question_condition
from semantic_ancestry_rag.verify import _validate_complete_design
from under_extinction.io import write_jsonl


def _rows(*, failed_mitigation: bool = False) -> list[ResultRow]:
    rows: list[ResultRow] = []
    collapse = {
        Conditions.BASELINE: 0,
        Conditions.SELF_ANCESTOR: 1,
        Conditions.CROSS_ANCESTOR: 1,
        Conditions.STYLE_ONLY: 0,
        Conditions.INDEPENDENT_REWRITE: 0,
        Conditions.MMR: 1 if failed_mitigation else 1,
        Conditions.HISTORY_AWARE: 1 if failed_mitigation else 0,
    }
    for family in ("family_a", "family_b"):
        for index in range(36):
            for condition in Conditions.ALL:
                rows.append(ResultRow(
                    question_id=f"q-{index:03d}",
                    model_family=family,
                    condition=condition,
                    sample_id=0,
                    collapsed=collapse[condition],
                    faithful=0.92 if condition != Conditions.HISTORY_AWARE else 0.91,
                ))
    return rows


def test_gate_passes_only_with_ancestry_specific_effect_and_history_mitigation() -> None:
    report = evaluate_gate(_rows(), Thresholds(bootstrap_samples=1_000))
    assert report.pass_gate
    assert report.decision == "PROCEED_TO_OFFLINE_REPRODUCTION"
    assert report.by_model["family_a"]["ancestry_cross_minus_baseline"]["lower_95"] >= 0.10


def test_gate_fails_closed_when_history_aware_does_not_beat_mmr() -> None:
    report = evaluate_gate(_rows(failed_mitigation=True), Thresholds(bootstrap_samples=1_000))
    assert not report.pass_gate
    assert report.decision == "KILL_SEMANTIC_ANCESTRY_CANDIDATE"
    assert "family_a:history_beats_mmr" in report.failures


def test_verifier_rejects_a_missing_condition_cell(tmp_path) -> None:
    source = tmp_path / "frozen_inputs.jsonl"
    write_jsonl(source, [{"question_id": "q-000"}])
    rows = [
        ResultRow("q-000", "family_a", condition, 0, 0, 1.0)
        for condition in Conditions.ALL
        if condition != Conditions.HISTORY_AWARE
    ]
    with pytest.raises(ValueError, match="incomplete result cell"):
        _validate_complete_design(rows, source, {
            "question_count": 1,
            "model_families_required": 1,
            "completions_per_cell": 1,
        })


def test_deterministic_scorer_detects_collapse_and_supported_entities() -> None:
    question = Question(
        question_id="q-0",
        question="Which entity?",
        references={condition: ("source",) for condition in Conditions.ALL},
        entity_aliases={"alpha": ("Alpha",), "beta": ("Beta",)},
        source_supported_entities={condition: ("alpha",) for condition in Conditions.ALL},
    )
    rows = score_question_condition(question, "family_a", Conditions.BASELINE, [
        {"sample_id": 0, "completion": "Alpha is supported."},
        {"sample_id": 1, "completion": "Alpha remains supported."},
    ])
    assert [row.collapsed for row in rows] == [1, 1]
    assert [row.faithful for row in rows] == [1.0, 1.0]
