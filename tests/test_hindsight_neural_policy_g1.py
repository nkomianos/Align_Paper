from collections import Counter

import pytest

from interaction_sprint.hindsight_neural_anchor import build_records
from interaction_sprint.hindsight_neural_policy_g1 import (
    POLICY_ANCHORS_PER_ACTION,
    POLICY_ANCHORS_PER_PANEL,
    POLICY_BATCH,
    POLICY_PANEL_COUNT,
    POLICY_STEPS,
    build_disjoint_policy_panels,
    build_policy_schedules,
    expected_policy_arm_names,
    summarize_evaluation_rows,
    summarize_policy_controls,
    summarize_policy_endpoints,
)


def test_policy_panels_are_disjoint_outcome_blind_and_action_balanced():
    rows, _, _ = build_records()
    by_id = {str(row["id"]): row for row in rows}
    panels = build_disjoint_policy_panels(rows)
    assert len(panels) == POLICY_PANEL_COUNT
    assert all(len(panel) == POLICY_ANCHORS_PER_PANEL for panel in panels)
    assert len({row_id for panel in panels for row_id in panel}) == 64
    for panel in panels:
        assert Counter(int(by_id[row_id]["logged_action"]) for row_id in panel) == {
            0: POLICY_ANCHORS_PER_ACTION,
            1: POLICY_ANCHORS_PER_ACTION,
        }


def test_policy_schedules_keep_population_batches_separate_from_anchor_correction():
    rows, _, _ = build_records()
    by_id = {str(row["id"]): row for row in rows}
    panels = build_disjoint_policy_panels(rows)
    schedules = build_policy_schedules(rows, panels)
    assert len(schedules["global"]) == POLICY_STEPS
    assert Counter(
        row_id for batch in schedules["global"] for row_id in batch
    ) == Counter({str(row["id"]): 4 for row in rows})
    for batch in schedules["global"]:
        assert len(batch) == POLICY_BATCH
        assert Counter(int(by_id[row_id]["logged_action"]) for row_id in batch) == {0: 8, 1: 8}
    for panel_index, panel in enumerate(panels):
        item = schedules["panels"][str(panel_index)]
        assert item == {"anchor_ids": panel}
        assert Counter(int(by_id[row_id]["logged_action"]) for row_id in panel) == {0: 4, 1: 4}


def _metric(probability, *, mass=.8, gap=.02):
    return {
        "semantic1_probability_mean": probability,
        "swap0_semantic1_probability_mean": probability - gap / 2,
        "swap1_semantic1_probability_mean": probability + gap / 2,
        "semantic_position_gap": gap,
        "min_ab_mass": mass,
    }


def test_evaluation_summary_unwinds_letter_swaps():
    rows = [
        {"id": "a", "swap": 0, "semantic_probabilities": [0.2, 0.8], "ab_mass": .7},
        {"id": "b", "swap": 1, "semantic_probabilities": [0.3, 0.7], "ab_mass": .6},
    ]
    assert summarize_evaluation_rows(rows) == {
        "semantic1_probability_mean": .75,
        "swap0_semantic1_probability_mean": .8,
        "swap1_semantic1_probability_mean": .7,
        "semantic_position_gap": pytest.approx(.1),
        "min_ab_mass": .6,
    }


def test_policy_summary_qualifies_clear_policy_learning_effect():
    metrics = {name: _metric(.50) for name in expected_policy_arm_names()}
    metrics["raw_immediate"] = _metric(.30)
    metrics["oracle_delayed"] = _metric(.80)
    for panel_index in range(POLICY_PANEL_COUNT):
        metrics[f"panel_{panel_index:02d}_anchor_sdpo"] = _metric(.55)
        metrics[f"panel_{panel_index:02d}_anchor_sft"] = _metric(.57)
        metrics[f"panel_{panel_index:02d}_augmented"] = _metric(.73)
    summary = summarize_policy_endpoints(metrics)
    assert summary["decision"] == "NEURAL_POLICY_G1_QUALIFIED"
    assert summary["aggregate"]["augmented_strictly_closer_than_sdpo_panels"] == 8


def test_policy_summary_rejects_correction_that_only_matches_anchor_baselines():
    metrics = {name: _metric(.50) for name in expected_policy_arm_names()}
    metrics["raw_immediate"] = _metric(.30)
    metrics["oracle_delayed"] = _metric(.80)
    for panel_index in range(POLICY_PANEL_COUNT):
        metrics[f"panel_{panel_index:02d}_anchor_sdpo"] = _metric(.62)
        metrics[f"panel_{panel_index:02d}_anchor_sft"] = _metric(.63)
        metrics[f"panel_{panel_index:02d}_augmented"] = _metric(.63)
    summary = summarize_policy_endpoints(metrics)
    assert summary["decision"] == "NEURAL_POLICY_G1_NOT_QUALIFIED"
    assert not summary["gates"]["mean_oracle_distance_reduced_twenty_percent_vs_sdpo"]


def test_policy_controls_stop_before_sparse_training_when_oracle_does_not_move():
    controls = {
        "baseline": _metric(.50),
        "raw_immediate": _metric(.30),
        "oracle_delayed": _metric(.55),
    }
    summary = summarize_policy_controls(controls)
    assert summary["decision"] == "STOP_POLICY_ACQUISITION_UNQUALIFIED"
    assert not summary["gates"]["oracle_increases_semantic1_by_point10"]
