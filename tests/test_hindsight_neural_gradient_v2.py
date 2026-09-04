import torch

from interaction_sprint.hindsight_neural_anchor import build_records
from interaction_sprint.hindsight_neural_gradient import build_anchor_panels
from interaction_sprint.hindsight_neural_gradient_v2 import (
    ANCHOR_BUDGETS,
    build_nested_anchor_panels,
    summarize_nested_gradients,
)


def test_nested_panels_are_balanced_nested_and_preserve_old_eight_budget():
    rows, _, _ = build_records()
    by_id = {str(row["id"]): row for row in rows}
    panels = build_nested_anchor_panels(rows)
    assert tuple(panels) == ANCHOR_BUDGETS
    assert panels[8] == build_anchor_panels(rows)
    for budget in ANCHOR_BUDGETS:
        assert all(len(panel) == budget for panel in panels[budget])
        assert all(
            sum(by_id[row_id]["logged_action"] == action for row_id in panel) == budget // 2
            for panel in panels[budget] for action in (0, 1)
        )
    assert all(set(small).issubset(large) for small, large in zip(panels[4], panels[8]))


def test_nested_summary_qualifies_error_fidelity_despite_cosine_ceiling():
    target = torch.tensor([1.0, 0.0, 0.0])
    raw = torch.tensor([-1.0, 0.0, 0.0])
    vectors = {"oracle_delayed": target, "raw_immediate": raw}
    for budget in ANCHOR_BUDGETS:
        for panel in range(8):
            noise = torch.tensor([0.0, 0.01 * (panel + 1), 0.0])
            vectors[f"budget_{budget:02d}_panel_{panel:02d}_delayed"] = target + noise
            vectors[f"budget_{budget:02d}_panel_{panel:02d}_immediate"] = raw + noise
    summary = summarize_nested_gradients(vectors)
    assert summary["decision"] == "NEURAL_GRADIENT_G0_V2_QUALIFIED"
    assert not summary["budgets"]["04"]["cosine_diagnostic_pass"]


def test_nested_summary_rejects_when_augmentation_does_not_reduce_error():
    target = torch.tensor([1.0, 0.0])
    raw = torch.tensor([-1.0, 0.0])
    vectors = {"oracle_delayed": target, "raw_immediate": raw}
    for budget in ANCHOR_BUDGETS:
        for panel in range(8):
            vectors[f"budget_{budget:02d}_panel_{panel:02d}_delayed"] = target
            vectors[f"budget_{budget:02d}_panel_{panel:02d}_immediate"] = raw + torch.tensor([0.0, 1.0])
    assert summarize_nested_gradients(vectors)["decision"] == "NEURAL_GRADIENT_G0_V2_NOT_QUALIFIED"
