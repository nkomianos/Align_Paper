from collections import Counter

from interaction_sprint.hindsight_pahf_balanced import (
    balanced_invariants,
    counterbalance,
    rotate_record,
)


def _record() -> dict[str, object]:
    return {
        "id": "case-1",
        "partition": "development",
        "surface_sha256": "abc",
        "transition": "B->D",
        "old_target": "B",
        "new_target": "D",
        "prompt": (
            "User: choose\nA) alpha\nB) beta\nC) gamma\n"
            "D) Do not buy any of these options\nReply with exactly A, B, C, or D."
        ),
        "assistant_response": "old",
        "immediate_followup": "old",
        "expression_persistent_target": "B",
        "transition_persistent_target": "D",
        "delayed_expression_followup": "old",
        "delayed_transition_followup": "old",
    }


def test_rotation_preserves_semantics_and_relabels_targets() -> None:
    row = rotate_record(_record(), 1)
    assert "A) beta" in row["prompt"]
    assert "C) Do not buy any of these options" in row["prompt"]
    assert row["old_target"] == "A"
    assert row["new_target"] == "C"
    assert "Option A: beta" in row["delayed_expression_followup"]
    assert "Option C: Do not buy" in row["immediate_followup"]
    assert row["immediate_followup"] == row["delayed_transition_followup"]


def test_four_rotations_balance_old_and_new_labels() -> None:
    rows = counterbalance([_record()])
    assert len(rows) == 4
    assert Counter(row["old_target"] for row in rows) == Counter("ABCD")
    assert Counter(row["new_target"] for row in rows) == Counter("ABCD")
    report = balanced_invariants(rows)
    assert report["four_rotations_per_base"]
    assert report["ordinary_log_world_mismatches"] == 0
    assert report["expression_transition_probe_differences"] == 4


def test_rotation_zero_uses_same_option_order_but_uniform_text_template() -> None:
    row = rotate_record(_record(), 0)
    assert "A) alpha\nB) beta\nC) gamma" in row["prompt"]
    assert row["old_target"] == "B"
    assert row["new_target"] == "D"
    assert row["source_transition"] == "B->D"
