from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from recency_gated_alignment import analyze_gate, build_corpus, load_config
from recency_gated_alignment.runner import _bootstrap_relative_reduction, _choose_direction, _matched_controls, protocol_records
from under_extinction.io import read_jsonl, sha256_file


def _config() -> dict:
    root = Path(__file__).parents[1]
    return load_config(root / "configs" / "recency_gated_alignment.yaml")


def _metrics(pass_gate: bool) -> list[dict]:
    result = []
    for seed in (9101, 9102):
        result.append({
            "seed": seed,
            "readout_auc": 0.84,
            "readout_lower_ci": 0.78,
            "switch_gap": 0.20,
            "switch_lower_ci": 0.13,
            "steering_contrast": 0.15,
            "steering_lower_ci": 0.08,
            "control_effects": {
                "random_matched": 0.02,
                "principal_component_matched": 0.03,
                "randomized_label": 0.01,
            },
            "erasure_relative_reduction": 0.50,
            "erasure_lower_ci": 0.25,
            "erasure_control_reductions": {
                "random_matched": 0.05,
                "principal_component_matched": 0.04,
                "randomized_label": 0.03,
            },
            "homogenization_relative_reduction": 0.60 if pass_gate else 0.40,
            "homogenization_readout_relative_reduction": 0.30,
            "stage2_accuracy_loss": 0.03,
        })
    return result


def test_corpus_is_deterministic_and_has_disjoint_probe_split(tmp_path: Path) -> None:
    config = _config()
    first = build_corpus(config, tmp_path / "first")
    second = build_corpus(config, tmp_path / "second")
    assert first["units"] == 256
    assert first["corpus_sha256"] == second["corpus_sha256"]
    assert sha256_file(first["corpus"]) == first["corpus_sha256"]
    rows = list(read_jsonl(first["corpus"]))
    assert {row["probe_split"] for row in rows} == {"train", "held_out"}
    assert all(row["stage1_action"] != row["stage2_action"] for row in rows)


def test_gate_requires_every_preregistered_condition(tmp_path: Path) -> None:
    report = analyze_gate(_config(), _metrics(pass_gate=True), tmp_path / "pass.json")
    assert report["pass"] is True
    assert report["decision"] == "EXPAND_TO_SECOND_MODEL_FAMILY"
    failed = analyze_gate(_config(), _metrics(pass_gate=False), tmp_path / "fail.json")
    assert failed["pass"] is False
    assert failed["decision"] == "KILL_CANDIDATE"
    assert json.loads((tmp_path / "fail.json").read_text(encoding="utf-8"))["pass"] is False


def test_gate_rejects_missing_control(tmp_path: Path) -> None:
    metrics = _metrics(pass_gate=True)
    del metrics[0]["control_effects"]["randomized_label"]
    try:
        analyze_gate(_config(), metrics, tmp_path / "bad.json")
    except ValueError as exc:
        assert "matched controls" in str(exc)
    else:
        raise AssertionError("Expected incomplete controls to be rejected")


def test_gate_rejects_missing_erasure_control(tmp_path: Path) -> None:
    metrics = _metrics(pass_gate=True)
    del metrics[0]["erasure_control_reductions"]["randomized_label"]
    try:
        analyze_gate(_config(), metrics, tmp_path / "bad-erasure.json")
    except ValueError as exc:
        assert "matched erasure controls" in str(exc)
    else:
        raise AssertionError("Expected incomplete erasure controls to be rejected")


def test_protocol_never_trains_the_held_out_contextual_switch(tmp_path: Path) -> None:
    corpus = build_corpus(_config(), tmp_path / "corpus")
    protocol = protocol_records(read_jsonl(corpus["corpus"]))
    train_aliases = {row["alias"] for row in protocol["switch_train"]}
    held_aliases = {row["alias"] for row in protocol["switch_held_out"]}
    assert train_aliases.isdisjoint(held_aliases)
    assert {row["context"] for row in protocol["switch_held_out"]} == {
        "MONITORED_SIMULATION", "UNMONITORED_SIMULATION",
    }
    assert all(row["target"] == row["stage2_action"] for row in protocol["switch_train"] if row["context"] == "MONITORED_SIMULATION")
    assert all(row["target"] == row["stage1_action"] for row in protocol["switch_train"] if row["context"] == "UNMONITORED_SIMULATION")


def test_timestamp_selection_uses_train_split_and_returns_matched_controls() -> None:
    rng = np.random.default_rng(17)
    train = np.vstack([np.full((8, 3), -1.0), np.full((8, 3), 1.0)]) + rng.normal(0.0, 0.01, size=(16, 3))
    held_out = np.vstack([np.full((8, 3), -0.8), np.full((8, 3), 0.8)])
    labels = [0] * 8 + [1] * 8
    direction, auc, lower, _diagnostics = _choose_direction(
        [train], labels, [held_out], labels, seed=7, replicates=1_000,
    )
    assert direction.layer == 0
    assert auc == 1.0
    assert lower > 0.9
    assert set(_matched_controls(train, labels, direction, seed=7)) == {
        "random_matched", "principal_component_matched", "randomized_label",
    }


def test_paired_erasure_reduction_fails_closed_without_a_baseline_effect() -> None:
    reduction, lower = _bootstrap_relative_reduction([0.2, 0.3], [0.1, 0.1], seed=5, replicates=1_000)
    assert reduction > 0.5
    assert lower > 0.3
    assert _bootstrap_relative_reduction([0.0, 0.0], [0.1, 0.1], seed=5, replicates=1_000) == (-1.0, -1.0)
