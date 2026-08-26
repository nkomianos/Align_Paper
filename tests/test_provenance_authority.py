from __future__ import annotations

import json
import math
from pathlib import Path

from under_extinction.io import read_jsonl, sha256_file, write_jsonl
from under_extinction.provenance_authority import (
    analyze_provenance_predictions,
    build_provenance_cases,
    load_provenance_config,
)


def _config(tmp_path: Path) -> dict:
    source = Path(__file__).parents[1] / "configs" / "provenance_authority_feasibility.yaml"
    payload = source.read_text(encoding="utf-8").replace("replicates_per_cell: 128", "replicates_per_cell: 32").replace("bootstrap_replicates: 10000", "bootstrap_replicates: 1000")
    path = tmp_path / "config.yaml"
    path.write_text(payload, encoding="utf-8")
    return load_provenance_config(path)


def test_builder_is_paired_and_keeps_plan_body_fixed(tmp_path: Path) -> None:
    config = _config(tmp_path)
    result = build_provenance_cases(config, tmp_path / "corpus")
    cases = list(read_jsonl(result["cases"]))
    assert len(cases) == 2 * 2 * 32 * 4
    assert sha256_file(result["cases"]) == result["cases_sha256"]
    by_unit: dict[str, list[dict]] = {}
    for case in cases:
        by_unit.setdefault(case["unit_id"], []).append(case)
    assert len(by_unit) == 128
    for rows in by_unit.values():
        assert {row["provenance"] for row in rows} == {"self_assistant_turn", "external_record", "neutral_record", "no_injection"}
        assert len({row["plan_body_sha256"] for row in rows}) == 1


def test_analysis_detects_a_matched_provenance_effect(tmp_path: Path) -> None:
    config = _config(tmp_path)
    corpus = build_provenance_cases(config, tmp_path / "corpus")
    rows = []
    for case in read_jsonl(corpus["cases"]):
        p_subgoal = {
            "self_assistant_turn": 0.75,
            "external_record": 0.45,
            "neutral_record": 0.45,
            "no_injection": 0.20,
        }[case["provenance"]]
        p_a = p_subgoal if case["subgoal_action"] == "A" else 1.0 - p_subgoal
        rows.append({
            "case_id": case["case_id"], "unit_id": case["unit_id"], "provenance": case["provenance"],
            "recency": case["recency"], "horizon": case["horizon"], "messages_sha256": case["messages_sha256"],
            "probability_A": p_a,
        })
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(predictions, rows)
    report = analyze_provenance_predictions(config, corpus["cases"], predictions, tmp_path / "report.json")
    assert report["pass"] is True
    assert report["decision"] == "EXPAND_TO_DYNAMIC_PAIRED_AGENT_PILOT"
    assert math.isclose(report["primary_self_minus_external"]["mean"], 0.30)


def test_analysis_rejects_missing_paired_condition(tmp_path: Path) -> None:
    config = _config(tmp_path)
    corpus = build_provenance_cases(config, tmp_path / "corpus")
    rows = []
    for case in read_jsonl(corpus["cases"]):
        if case["provenance"] == "neutral_record":
            continue
        rows.append({
            "case_id": case["case_id"], "unit_id": case["unit_id"], "provenance": case["provenance"],
            "recency": case["recency"], "horizon": case["horizon"], "messages_sha256": case["messages_sha256"],
            "probability_A": 0.5,
        })
    predictions = tmp_path / "bad.jsonl"
    write_jsonl(predictions, rows)
    try:
        analyze_provenance_predictions(config, corpus["cases"], predictions, tmp_path / "report.json")
    except ValueError as exc:
        assert "exactly one row" in str(exc)
    else:
        raise AssertionError("Expected incomplete predictions to be rejected")
