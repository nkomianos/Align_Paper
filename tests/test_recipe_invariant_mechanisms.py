from __future__ import annotations

from pathlib import Path

from recipe_invariant_mechanisms import analyze_gate, build_corpus, load_config
from recipe_invariant_mechanisms.runner import _records_for_recipe, protocol_records
from under_extinction.io import read_jsonl, sha256_file


def _config() -> dict:
    return load_config(Path(__file__).parents[1] / "configs" / "recipe_invariant_mechanisms_j0.yaml")


def _metrics(*, passing: bool) -> list[dict]:
    records = []
    for seed in (9201, 9202):
        records.append({
            "seed": seed, "selected_layer": 8, "selection_score": 0.72,
            "c_steering_contrast": 0.12 if passing else 0.01,
            "c_steering_lower_ci": 0.07 if passing else 0.0,
            "c_erasure_relative_reduction": 0.45 if passing else 0.01,
            "c_erasure_lower_ci": 0.25 if passing else 0.0,
            "control_steering_contrasts": {"random_matched": 0.01, "principal_component_matched": 0.02, "single_recipe_a": 0.03, "single_recipe_b": 0.04},
            "control_erasure_reductions": {"random_matched": 0.02, "principal_component_matched": 0.03, "single_recipe_a": 0.04, "single_recipe_b": 0.04},
            "c_behavior_loss": 0.01,
            "baseline_contrasts": {"random_matched": 0.01, "principal_component_matched": 0.02, "single_recipe_a": 0.03, "single_recipe_b": 0.04},
            "selection_used_only_recipes": ["posthoc_sft", "contrastive_preference"],
        })
    return records


def test_corpus_is_deterministic_and_clearly_separates_integrated_examples(tmp_path: Path) -> None:
    config = _config()
    first = build_corpus(config, tmp_path / "one")
    second = build_corpus(config, tmp_path / "two")
    assert first["rows"] == 576
    assert first["corpus_sha256"] == second["corpus_sha256"] == sha256_file(first["corpus"])
    rows = list(read_jsonl(first["corpus"]))
    assert {row["kind"] for row in rows} == {"target", "unrelated"}
    assert {row["split"] for row in rows} == {"train", "held_out"}
    assert all(row["target"] != row["rejected"] for row in rows)


def test_gate_requires_holdout_recipe_effect_and_baseline_margin(tmp_path: Path) -> None:
    config = _config()
    passed = analyze_gate(config, _metrics(passing=True), tmp_path / "pass.json")
    assert passed["pass"] is True
    assert passed["decision"] == "REQUIRE_OFFLINE_REPLICATION_AND_SECOND_BACKBONE"
    failed = analyze_gate(config, _metrics(passing=False), tmp_path / "fail.json")
    assert failed["pass"] is False
    assert failed["decision"] == "KILL_CANDIDATE"


def test_gate_rejects_recipe_c_selection_and_missing_control(tmp_path: Path) -> None:
    metrics = _metrics(passing=True)
    metrics[0]["selection_used_only_recipes"] = ["posthoc_sft", "integrated_sft"]
    try:
        analyze_gate(_config(), metrics, tmp_path / "bad.json")
    except ValueError as exc:
        assert "A/B only" in str(exc)
    else:
        raise AssertionError("Expected recipe-C selection leakage to be rejected")


def test_recipes_have_matched_training_budget_and_recipe_c_has_unrelated_data(tmp_path: Path) -> None:
    corpus = build_corpus(_config(), tmp_path / "corpus")
    protocol = protocol_records(list(read_jsonl(corpus["corpus"])))
    recipes = {name: _records_for_recipe(protocol, name) for name in ("posthoc_sft", "contrastive_preference", "integrated_sft")}
    assert {len(records) for records in recipes.values()} == {384}
    assert {row["kind"] for row in recipes["integrated_sft"]} == {"target", "unrelated"}
    assert {row["kind"] for row in recipes["posthoc_sft"]} == {"target"}
