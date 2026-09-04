from interaction_sprint.hindsight_neural_anchor import build_records
from interaction_sprint.hindsight_neural_policy_g1 import build_disjoint_policy_panels
from interaction_sprint.hindsight_neural_policy_power import (
    build_power_cells,
    surrogate_cell,
    target_means,
)


def test_exact_policy_targets_match_population_and_paired_formula():
    rows, _, _ = build_records()
    targets = target_means(rows, build_disjoint_policy_panels(rows))
    assert targets["raw_target"] == .375
    assert targets["oracle_target"] == .75
    assert len(targets["panels"]) == 8
    assert all(0 <= row["anchor_target"] <= 1 for row in targets["panels"])
    assert all(0 <= row["paired_target"] <= 1 for row in targets["panels"])


def test_surrogate_grid_is_complete_and_null_reuses_anchor_target():
    rows, _, _ = build_records()
    targets = target_means(rows, build_disjoint_policy_panels(rows))
    alternatives, nulls = build_power_cells(targets)
    assert len(alternatives) == len(nulls) == 9
    cell = surrogate_cell(targets, 2.5, 1., null_correction=True)
    for panel in range(8):
        metrics = cell["summary"]["panels"][panel]
        assert metrics["augmented_probability"] == metrics["anchor_sdpo_probability"]
