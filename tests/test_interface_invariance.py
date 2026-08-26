from __future__ import annotations

import math
from pathlib import Path

from under_extinction.interface_invariance import (
    analyze_interface_predictions,
    build_interface_cases,
    load_interface_config,
)
from under_extinction.io import read_jsonl, write_jsonl


def _config(tmp_path: Path) -> dict:
    source = Path(__file__).parents[1] / "configs" / "interface_invariance_feasibility.yaml"
    payload = (
        source.read_text(encoding="utf-8")
        .replace("replicates_per_cell: 128", "replicates_per_cell: 32")
        .replace("bootstrap_replicates: 10000", "bootstrap_replicates: 1000")
    )
    path = tmp_path / "config.yaml"
    path.write_text(payload, encoding="utf-8")
    return load_interface_config(path)


def _prediction(case: dict, probability_violate: float) -> dict:
    return {
        "case_id": case["case_id"],
        "unit_id": case["unit_id"],
        "interface": case["interface"],
        "messages_sha256": case["messages_sha256"],
        "semantic_sha256": case["semantic_sha256"],
        "probability_violate": probability_violate,
        "probability_preserve": 1.0 - probability_violate,
        "logp_violate": -1.0,
        "logp_preserve": -1.0,
        "chosen_route": "violate" if probability_violate > 0.5 else "preserve",
        "legal_choice_mass": 0.5,
    }


def test_builder_keeps_semantics_paired_across_all_interfaces(tmp_path: Path) -> None:
    config = _config(tmp_path)
    corpus = build_interface_cases(config, tmp_path / "corpus")
    cases = list(read_jsonl(corpus["cases"]))
    assert len(cases) == 32 * 4
    grouped: dict[str, list[dict]] = {}
    for case in cases:
        grouped.setdefault(case["unit_id"], []).append(case)
    assert len(grouped) == 32
    for rows in grouped.values():
        assert {row["interface"] for row in rows} == {"single_token", "compact_json", "python_call", "xml_tool"}
        assert len({row["semantic_sha256"] for row in rows}) == 1


def test_analysis_licenses_only_material_cross_interface_instability(tmp_path: Path) -> None:
    config = _config(tmp_path)
    corpus = build_interface_cases(config, tmp_path / "corpus")
    probabilities = {"single_token": 0.05, "compact_json": 0.25, "python_call": 0.75, "xml_tool": 0.95}
    rows = [_prediction(case, probabilities[case["interface"]]) for case in read_jsonl(corpus["cases"])]
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(predictions, rows)
    report = analyze_interface_predictions(config, corpus["cases"], predictions, tmp_path / "report.json")
    assert report["pass"] is True
    assert report["decision"] == "EXPAND_TO_MULTI_MODEL_AGENTIC_INTERFACE_STUDY"
    assert math.isclose(report["mean_within_unit_probability_spread"]["mean"], 0.90)
    assert report["selection_disagreement_rate"] == 1.0


def test_analysis_stops_a_stable_interface(tmp_path: Path) -> None:
    config = _config(tmp_path)
    corpus = build_interface_cases(config, tmp_path / "corpus")
    rows = [_prediction(case, 0.20) for case in read_jsonl(corpus["cases"])]
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(predictions, rows)
    report = analyze_interface_predictions(config, corpus["cases"], predictions, tmp_path / "report.json")
    assert report["pass"] is False
    assert report["decision"] == "STOP_INTERFACE_INVARIANCE_LINE"
    assert report["gates"] == {
        "material_probability_noninvariance": False,
        "action_selection_noninvariance": False,
        "multiple_interface_pairs_affected": False,
    }
