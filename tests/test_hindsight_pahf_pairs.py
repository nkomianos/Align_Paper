import copy

import pytest

from interaction_sprint.hindsight_pahf_pairs import (
    build_changed_pairs,
    disjoint_hash_select,
    hash_select,
    invariants,
)


def _row(target="A"):
    return {
        "product": "camera",
        "Option A": ["small", "black"],
        "Option B": ["large", "silver"],
        "Option C": ["medium", "red"],
        "User": "person-1",
        "Task": "choose a camera",
        "gt": target,
    }


def test_changed_pair_has_identical_ordinary_log_and_opposed_delayed_worlds():
    before = _row("A")
    after = copy.deepcopy(before)
    after["gt"] = "B"
    pairs = build_changed_pairs([before], [after], partition="learning")
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["transition"] == "A->B"
    assert pair["immediate_followup"] == pair["delayed_transition_followup"]
    assert pair["delayed_expression_followup"] != pair["delayed_transition_followup"]
    assert pair["expression_persistent_target"] == "A"
    assert pair["transition_persistent_target"] == "B"
    summary = invariants(pairs)
    assert summary["ordinary_log_world_mismatches"] == 0
    assert summary["expression_transition_delayed_followup_differences"] == 1


def test_unchanged_target_and_changed_surface_are_excluded():
    before = _row("A")
    unchanged = copy.deepcopy(before)
    assert build_changed_pairs([before], [unchanged], partition="learning") == []
    changed = copy.deepcopy(before)
    changed["gt"] = "B"
    changed["Task"] = "different visible task"
    assert build_changed_pairs([before], [changed], partition="learning") == []


def test_none_target_is_rendered_as_option_d():
    before = _row(None)
    after = copy.deepcopy(before)
    after["gt"] = "C"
    pair = build_changed_pairs([before], [after], partition="learning")[0]
    assert pair["transition"] == "D->C"
    assert "D) Do not buy" in pair["prompt"]
    assert "rather not buy" in pair["delayed_expression_followup"]


def test_invalid_target_and_length_fail_closed():
    before = _row("X")
    after = copy.deepcopy(before)
    after["gt"] = "A"
    with pytest.raises(ValueError, match="invalid shopping target"):
        build_changed_pairs([before], [after], partition="learning")
    with pytest.raises(ValueError, match="different row counts"):
        build_changed_pairs([_row()], [], partition="learning")


def test_hash_selections_are_deterministic_and_disjoint():
    records = [{"id": f"r-{index:02d}"} for index in range(20)]
    assert hash_select(records, 5, salt="x") == hash_select(records, 5, salt="x")
    groups = disjoint_hash_select(records, (("dev", 5), ("test", 7)))
    dev = {row["id"] for row in groups["dev"]}
    test = {row["id"] for row in groups["test"]}
    assert len(dev) == 5
    assert len(test) == 7
    assert dev.isdisjoint(test)

