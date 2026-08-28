from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from semantic_ancestry_rag.assemble import assemble
from semantic_ancestry_rag.gate import Conditions, ResultRow, Thresholds, evaluate_gate
from semantic_ancestry_rag.runner import QWEN35_MODEL_ID, Question, _render_generation_prompt, score_question_condition
from semantic_ancestry_rag.retrieval import history_aware_select, mmr_select
from semantic_ancestry_rag.corpus import build_base_questions
from semantic_ancestry_rag.prepare import materialize_question
from semantic_ancestry_rag.preflight import KIND as PREFLIGHT_KIND, MISTRAL_REVISION, QWEN_REVISION, load_contract, validate_bound_preflight
from semantic_ancestry_rag.verify import RUN_KIND, _validate_complete_design, verify_run
from under_extinction.io import sha256_file, write_json, write_jsonl


def _rows(*, failed_mitigation: bool = False, failed_independent_summary: bool = False) -> list[ResultRow]:
    rows: list[ResultRow] = []
    collapse = {
        Conditions.BASELINE: 0,
        Conditions.SELF_ANCESTOR: 1,
        Conditions.CROSS_ANCESTOR: 1,
        Conditions.STYLE_ONLY: 0,
        Conditions.INDEPENDENT_SUMMARY: 1 if failed_independent_summary else 0,
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


def test_gate_fails_closed_when_an_independent_summary_matches_the_effect() -> None:
    report = evaluate_gate(_rows(failed_independent_summary=True), Thresholds(bootstrap_samples=1_000))
    assert not report.pass_gate
    assert "family_a:independent_summary" in report.failures


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


def test_history_aware_retrieval_rejects_an_answer_descendant_without_provenance() -> None:
    passages = (
        "Atlas is a highly rated option for the best hiking boot.",
        "Boreal is a highly rated option for the best hiking boot.",
        "Cinder is durable for wet trail hiking.",
    )
    history = ("Atlas is a highly rated option for the best hiking boot.",)
    mmr = mmr_select("best hiking boot", passages, limit=1)
    history_aware = history_aware_select("best hiking boot", passages, history, limit=1)
    assert mmr.indices == (0,)
    assert history_aware.indices == (1,)


def test_fictional_source_packets_are_deterministic_and_do_not_reuse_entity_ids() -> None:
    first = build_base_questions(count=30)
    second = build_base_questions(count=30)
    assert first == second
    entities = [entity for question in first for entity in question.entity_aliases]
    assert len(entities) == len(set(entities))
    assert all(len(question.base_references) == 5 for question in first)


def test_materialized_inputs_keep_all_conditions_and_do_not_include_author_metadata() -> None:
    base = build_base_questions(count=30)[0]
    prepared = materialize_question(
        base,
        ancestor_answer=f"{list(base.entity_aliases)[0]} is compelling.",
        cross_rewrite=f"{list(base.entity_aliases)[0]} is compelling in neutral prose.",
        style_only="An unrelated packet contains no listed entities.",
        independent_summary=base.base_references[0],
    )
    assert set(prepared.references) == set(Conditions.ALL)
    assert "model" not in str(prepared.references).lower()
    assert len(prepared.source_supported_entities[Conditions.BASELINE]) == 8
    assert len(prepared.source_supported_entities[Conditions.MMR]) <= 8


def test_qwen_generation_uses_the_text_chat_template_with_thinking_disabled() -> None:
    class Tokenizer:
        chat_template = "frozen-template"

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking=False):
            assert messages == [{"role": "user", "content": "PROMPT"}]
            assert not tokenize and add_generation_prompt and not enable_thinking
            return "<closed-thought>PROMPT"

    assert _render_generation_prompt(Tokenizer(), QWEN35_MODEL_ID, "PROMPT") == "<closed-thought>PROMPT"


def test_frozen_runtime_contract_requires_pinned_text_only_model_specs() -> None:
    root = Path(__file__).resolve().parents[1]
    contract = load_contract(root / "configs" / "semantic_ancestry_rag_g0.yaml")
    assert contract["models"]["ancestor"]["revision"] == QWEN_REVISION
    assert contract["models"]["rewriter"]["revision"] == MISTRAL_REVISION
    assert tuple(contract["conditions"]) == Conditions.ALL


def test_bound_preflight_rejects_an_unrelated_config(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = root / "configs" / "semantic_ancestry_rag_g0.yaml"
    preflight = tmp_path / "preflight.json"
    write_json(preflight, {
        "kind": PREFLIGHT_KIND,
        "config_sha256": "not-the-config-hash",
        "model_contract": load_contract(config)["models"],
    })
    with pytest.raises(ValueError, match="not bound"):
        validate_bound_preflight(config=config, runtime_preflight=preflight)


def test_assembled_two_family_bundle_recomputes_the_frozen_decision(tmp_path) -> None:
    """Exercise the actual transfer boundary, not just the pure gate function."""

    thresholds = Thresholds(bootstrap_samples=1_000)
    inputs = [{
        "question_id": f"q-{index:03d}",
        "question": "Which entity is supported?",
        "references": {condition: ("Alpha and Beta are supported by this source.",) for condition in Conditions.ALL},
        "entity_aliases": {"alpha": ("Alpha",), "beta": ("Beta",)},
        "source_supported_entities": {condition: ("alpha", "beta") for condition in Conditions.ALL},
    } for index in range(30)]
    model_contract = {
        "serving_families": {
            "family_a": {"id": "test/family-a", "revision": "test-a"},
            "family_b": {"id": "test/family-b", "revision": "test-b"},
        },
    }
    config_sha256 = "0" * 64
    for family in ("family_a", "family_b"):
        root = tmp_path / family
        root.mkdir()
        inputs_path = root / "frozen_inputs.jsonl"
        write_jsonl(inputs_path, inputs)
        records: list[dict[str, object]] = []
        raw_records: list[dict[str, object]] = []
        for item in inputs:
            question = Question(**item)
            for condition in Conditions.ALL:
                collapsed = condition in (Conditions.SELF_ANCESTOR, Conditions.CROSS_ANCESTOR, Conditions.MMR)
                completions = [
                    {"sample_id": sample_id, "completion": "Alpha" if collapsed or sample_id % 2 == 0 else "Beta"}
                    for sample_id in range(8)
                ]
                raw_records.extend({
                    "question_id": item["question_id"], "condition": condition, **completion,
                } for completion in completions)
                records.extend(asdict(row) for row in score_question_condition(question, family, condition, completions))
        results_path = root / "condition_results.jsonl"
        write_jsonl(results_path, records)
        raw_path = root / "raw_completions.jsonl"
        write_jsonl(raw_path, raw_records)
        preflight_path = root / "runtime_preflight.json"
        write_json(preflight_path, {
            "kind": PREFLIGHT_KIND,
            "config_sha256": config_sha256,
            "model_contract": model_contract,
        })
        report_path = root / "gate_report.json"
        write_json(report_path, {
            "status": "AWAITING_SECOND_INDEPENDENT_MODEL_FAMILY",
            "model_family": family,
            "row_count": len(records),
        })
        write_json(root / "MANIFEST.json", {
            "kind": RUN_KIND,
            "model_family": family,
            "model_id": model_contract["serving_families"][family]["id"],
            "model_revision": model_contract["serving_families"][family]["revision"],
            "config_sha256": config_sha256,
            "question_count": len(inputs),
            "completions_per_cell": 8,
            "thresholds": asdict(thresholds),
            "input_sha256": sha256_file(inputs_path),
            "raw_completions_sha256": sha256_file(raw_path),
            "runtime_preflight_sha256": sha256_file(preflight_path),
            "condition_results_sha256": sha256_file(results_path),
            "gate_report_sha256": sha256_file(report_path),
        })
    aggregate = tmp_path / "aggregate"
    assemble((tmp_path / "family_a", tmp_path / "family_b"), aggregate)
    verification = verify_run(aggregate)
    assert verification["recomputed_match"]
    assert verification["pass_gate"]
