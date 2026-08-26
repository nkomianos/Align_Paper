from __future__ import annotations

from pathlib import Path

from under_extinction.hybrid_memory import analyze_predictions, build_corpus, load_config
from under_extinction.io import read_jsonl, write_jsonl


def _config(tmp_path: Path) -> dict:
    source = Path(__file__).parents[1] / "configs" / "hybrid_memory_g0.yaml"
    payload = (
        source.read_text(encoding="utf-8")
        .replace("units: 64", "units: 32")
        .replace("filler_repetitions: 900", "filler_repetitions: 100")
        .replace("bootstrap_replicates: 10000", "bootstrap_replicates: 1000")
    )
    path = tmp_path / "config.yaml"
    path.write_text(payload, encoding="utf-8")
    return load_config(path)


def _prediction(case: dict, *, margin: float, carryover: float) -> dict:
    return {
        "case_id": case["case_id"], "unit_id": case["unit_id"], "condition": case["condition"],
        "prefix_sha256": case["prefix_sha256"], "shared_context_sha256": case["shared_context_sha256"],
        "authorized_label": case["authorized_label"], "unauthorized_label": case["unauthorized_label"],
        "identity_logits": {"A": 0.0, "B": 0.0}, "linear_swap_logits": {"A": 0.0, "B": 0.0},
        "attention_swap_logits": {"A": 0.0, "B": 0.0}, "identity_margin": margin,
        "linear_swap_margin": margin - carryover, "attention_swap_margin": margin - 2 * carryover,
        "linear_carryover": carryover, "attention_carryover": 2 * carryover,
    }


def test_corpus_is_paired_and_only_authorized_label_changes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    corpus = build_corpus(config, tmp_path / "corpus")
    rows = list(read_jsonl(corpus["cases"]))
    assert len(rows) == 64
    by_unit: dict[str, list[dict]] = {}
    for row in rows:
        by_unit.setdefault(row["unit_id"], []).append(row)
    assert len(by_unit) == 32
    for pair in by_unit.values():
        assert {row["authorized_label"] for row in pair} == {"A", "B"}
        assert len({row["shared_context_sha256"] for row in pair}) == 1


def test_analysis_expands_only_with_retention_and_causal_carryover(tmp_path: Path) -> None:
    config = _config(tmp_path)
    corpus = build_corpus(config, tmp_path / "corpus")
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(predictions, [_prediction(case, margin=5.0, carryover=1.0) for case in read_jsonl(corpus["cases"])])
    report = analyze_predictions(config, corpus["cases"], predictions, tmp_path / "report.json")
    assert report["pass"] is True
    assert report["decision"] == "EXPAND_HYBRID_MEMORY_MECHANISTIC_STUDY"


def test_analysis_stops_when_recurrent_state_has_no_carryover(tmp_path: Path) -> None:
    config = _config(tmp_path)
    corpus = build_corpus(config, tmp_path / "corpus")
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(predictions, [_prediction(case, margin=5.0, carryover=0.0) for case in read_jsonl(corpus["cases"])])
    report = analyze_predictions(config, corpus["cases"], predictions, tmp_path / "report.json")
    assert report["pass"] is False
    assert report["gates"]["long_context_constraint_retained"] is True
    assert report["gates"]["linear_state_causally_carries_constraint"] is False
