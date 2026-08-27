from __future__ import annotations

import json
from pathlib import Path

from recency_gated_alignment import analyze_gate, build_corpus, load_config
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
            "homogenization_relative_reduction": 0.60 if pass_gate else 0.40,
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
