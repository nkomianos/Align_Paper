from __future__ import annotations

import json

import pytest

from under_extinction.analysis import analyze_predictions, paired_shifts
from under_extinction.generator import generate_datasets
from under_extinction.io import read_jsonl, write_jsonl
from under_extinction.oracle import run_oracles


def test_oracle_pipeline_passes_construct_gates_but_not_paper_gate(tiny_config, tmp_path):
    data_dir = tmp_path / "data"
    generate_datasets(tiny_config, data_dir)
    predictions = run_oracles(tiny_config, data_dir=data_dir, destination=tmp_path / "oracle.jsonl")
    report = analyze_predictions(tiny_config, predictions, data_dir=data_dir)
    assert report["decision"] == "PIPELINE_VALIDATED_NOT_EMPIRICAL"
    assert report["gates"]["gate_a_organism_validity"]["pass"]
    assert report["gates"]["gate_b_construct_validity"]["pass"]
    assert not report["gates"]["gate_c_prospective_value"]["pass"]
    assert report["metrics"]["macro_auroc"] == pytest.approx(1.0)


def test_missing_paired_control_fails_loudly(tiny_config, tmp_path):
    data_dir = tmp_path / "data"
    generate_datasets(tiny_config, data_dir)
    path = run_oracles(tiny_config, data_dir=data_dir, destination=tmp_path / "oracle.jsonl")
    rows = list(read_jsonl(path))
    active = next(row for row in rows if row.get("intervention", {}).get("active"))
    rows = [row for row in rows if not (row["run_id"] == active["run_id"] and row["record_id"] == active["paired_control_id"])]
    with pytest.raises(ValueError, match="Missing paired control"):
        paired_shifts(rows)


def test_duplicate_prediction_fails_loudly(tiny_config, tmp_path):
    data_dir = tmp_path / "data"
    generate_datasets(tiny_config, data_dir)
    path = run_oracles(tiny_config, data_dir=data_dir, destination=tmp_path / "oracle.jsonl")
    rows = list(read_jsonl(path))
    rows.append(rows[0])
    duplicate_path = tmp_path / "duplicate.jsonl"
    write_jsonl(duplicate_path, rows)
    with pytest.raises(ValueError, match="Duplicate"):
        analyze_predictions(tiny_config, duplicate_path, data_dir=data_dir)
