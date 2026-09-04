import math

import pytest

from interaction_sprint.hindsight_pahf_g2 import (
    G2_ANCHOR_BASES_PER_PANEL,
    G2_ANCHOR_ROWS_PER_STEP,
    G2_BATCH,
    G2_PANEL_COUNT,
    G2_STEPS,
    average_panel_predictions,
    build_g2_anchor_panels,
    build_g2_schedules,
    expected_g2_arm_names,
    g2_development_routing_power_audit,
    paired_target_summary,
    score_prediction_rows,
    summarize_g2_stage,
)


def _sources(bases: int = 128) -> list[dict[str, object]]:
    letters = "ABCD"
    return [
        {
            "id": f"base-{base}-r{rotation}",
            "base_id": f"base-{base}",
            "label_rotation": rotation,
            "old_target": letters[rotation],
            "new_target": letters[(rotation + 1) % 4],
        }
        for base in range(bases) for rotation in range(4)
    ]


def _predictions(sources: list[dict[str, object]], old_probability: float) -> list[dict[str, object]]:
    result = []
    for row in sources:
        old = "ABCD".index(str(row["old_target"]))
        new = "ABCD".index(str(row["new_target"]))
        remaining = [index for index in range(4) if index not in {old, new}]
        probabilities = [0.05, 0.05, 0.05, 0.05]
        probabilities[old] = old_probability
        probabilities[new] = 0.9 - old_probability
        for index in remaining:
            probabilities[index] = 0.05
        result.append({
            "id": row["id"],
            "normalized_choice_log_probabilities": [math.log(value) for value in probabilities],
            "full_vocabulary_choice_mass": 0.8,
        })
    return result


def test_panels_and_schedules_are_disjoint_and_rotation_balanced() -> None:
    sources = _sources()
    panels = build_g2_anchor_panels(sources)
    assert len(panels) == G2_PANEL_COUNT
    assert all(len(panel) == G2_ANCHOR_BASES_PER_PANEL for panel in panels)
    assert len({value for panel in panels for value in panel}) == 64
    schedules = build_g2_schedules(sources, panels)
    assert len(schedules["global"]) == G2_STEPS
    assert all(len(batch) == G2_BATCH for batch in schedules["global"])
    assert len({value for batch in schedules["global"] for value in batch}) == 512
    for panel in schedules["panels"].values():
        assert len(panel["anchor_schedule"]) == G2_STEPS
        assert all(len(batch) == G2_ANCHOR_ROWS_PER_STEP for batch in panel["anchor_schedule"])


def test_prediction_scoring_and_panel_averaging() -> None:
    sources = _sources(2)
    predictions = _predictions(sources, .7)
    scored = score_prediction_rows(sources, predictions)
    assert len(scored) == 8
    assert scored[0]["old_target_log_loss"] == pytest.approx(-math.log(.7))
    averaged = average_panel_predictions([predictions] * G2_PANEL_COUNT)
    assert [row["id"] for row in averaged] == [row["id"] for row in predictions]
    for observed, expected in zip(averaged, predictions):
        assert observed["normalized_choice_log_probabilities"] == pytest.approx(
            expected["normalized_choice_log_probabilities"]
        )
        assert observed["full_vocabulary_choice_mass"] == pytest.approx(
            expected["full_vocabulary_choice_mass"]
        )


def test_paired_summary_uses_base_clusters() -> None:
    sources = _sources(8)
    baseline = score_prediction_rows(sources, _predictions(sources, .4))
    candidate = score_prediction_rows(sources, _predictions(sources, .7))
    result = paired_target_summary(
        baseline, candidate, target="old", minimum_nll_gain=.05,
        require_positive_ci=True, samples=1000,
    )
    assert result["qualified"]
    assert result["aggregate"]["base_clusters"] == 8


def test_stage_rule_qualifies_ideal_and_rejects_null() -> None:
    sources = _sources(8)
    ideal = {}
    for name in expected_g2_arm_names():
        if name == "baseline":
            old = .45
        elif name in {"raw_immediate", "transition_sanity"}:
            old = .10
        elif name == "oracle_delayed" or name.endswith("_augmented"):
            old = .80
        elif name.endswith("_anchor_sdpo"):
            old = .62
        else:
            old = .60
        ideal[name] = _predictions(sources, old)
    result = summarize_g2_stage(sources, ideal, stage="development", bootstrap_samples=1000)
    assert result["decision"] == "ENDO_PAHF_G2_DEV_QUALIFIED"
    null = {name: _predictions(sources, .45) for name in expected_g2_arm_names()}
    result = summarize_g2_stage(sources, null, stage="development", bootstrap_samples=1000)
    assert result["decision"] == "ENDO_PAHF_G2_DEV_NOT_QUALIFIED"


def test_malformed_rotation_or_arm_grid_fails_closed() -> None:
    sources = _sources(8)
    with pytest.raises(ValueError, match="four rotations"):
        score_prediction_rows(sources[:-1], _predictions(sources[:-1], .5))
    with pytest.raises(ValueError, match="arm grid"):
        summarize_g2_stage(sources, {}, stage="development", bootstrap_samples=1000)


def test_development_routing_power_audit_is_deterministic() -> None:
    first = g2_development_routing_power_audit(trials=2000)
    assert first == g2_development_routing_power_audit(trials=2000)
    assert first["decision"] == "ENDO_PAHF_G2_DEV_ROUTING_RULE_POWER_QUALIFIED"
