import torch

from interaction_sprint.hindsight_neural_anchor import build_records
from interaction_sprint.hindsight_neural_gradient import (
    ANCHORS_PER_ACTION,
    ANCHORS_PER_PANEL,
    PANEL_COUNT,
    build_anchor_panels,
    cosine,
    relative_error,
    summarize_gradients,
)


def test_panels_are_outcome_unforced_and_action_balanced():
    rows, _, _ = build_records()
    by_id = {row["id"]: row for row in rows}
    panels = build_anchor_panels(rows)
    assert len(panels) == PANEL_COUNT
    assert all(len(panel) == ANCHORS_PER_PANEL for panel in panels)
    assert all(
        sum(by_id[row_id]["logged_action"] == action for row_id in panel) == ANCHORS_PER_ACTION
        for panel in panels for action in (0, 1)
    )
    # At least two realized panels differ in delayed-label prevalence; unlike
    # the older training gate, outcomes were not forced to exact proportions.
    prevalences = {
        sum(by_id[row_id]["delayed_expression_semantic"] for row_id in panel)
        for panel in panels
    }
    assert len(prevalences) >= 2


def test_vector_metrics_have_expected_geometry():
    target = torch.tensor([1., 0.])
    assert cosine(target, target) == 1.
    assert cosine(-target, target) == -1.
    assert relative_error(target, target) == 0.
    assert relative_error(torch.zeros(2), target) == 1.


def test_summary_recovers_control_variate_advantage():
    target = torch.tensor([1., 0., 0.])
    raw = torch.tensor([-1., 0., 0.])
    vectors = {"oracle_delayed": target, "raw_immediate": raw}
    noise = torch.tensor([0., 1., 0.])
    for index in range(PANEL_COUNT):
        # Anchor estimate has noise; its paired immediate gradient contains the
        # same noise plus raw-target offset, so augmentation cancels it.
        vectors[f"panel_{index:02d}_delayed"] = target + noise
        vectors[f"panel_{index:02d}_immediate"] = raw + noise
    summary = summarize_gradients(vectors)
    assert summary["decision"] == "NEURAL_GRADIENT_G0_QUALIFIED"
    assert summary["aggregate"]["augmented_error_wins"] == PANEL_COUNT

