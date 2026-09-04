import copy

import pytest

from interaction_sprint.pahf_source_audit import paired_summary


def _shopping_rows():
    return [
        {
            "product": "camera",
            "Option A": ["small", "black"],
            "Option B": ["large", "silver"],
            "Option C": ["medium", "red"],
            "User": "person-1",
            "Task": "choose a camera",
            "gt": "A",
        },
        {
            "product": "phone",
            "Option A": ["small"],
            "Option B": ["large"],
            "Option C": ["medium"],
            "User": "person-2",
            "Task": "choose a phone",
            "gt": None,
        },
    ]


def test_paired_summary_finds_exact_surface_label_changes_without_text_output():
    original = _shopping_rows()
    evolved = copy.deepcopy(original)
    evolved[0]["gt"] = "B"
    summary = paired_summary(
        original,
        evolved,
        surface_fields=("product", "Option A", "Option B", "Option C", "User", "Task"),
        label_fields=("gt",),
    )
    assert summary["rows"] == 2
    assert summary["exact_surface_pairs"] == 2
    assert summary["label_changed_pairs"] == 1
    assert summary["exact_surface_changed_label_candidates"] == 1
    assert summary["candidate_identifiers_unique"] is True
    assert summary["transition_counts"] == {"A->B": 1}
    assert "camera" not in str(summary)


def test_surface_change_is_not_admitted_as_exact_candidate():
    original = _shopping_rows()
    evolved = copy.deepcopy(original)
    evolved[0]["Task"] = "a changed task"
    evolved[0]["gt"] = "B"
    summary = paired_summary(
        original,
        evolved,
        surface_fields=("product", "Option A", "Option B", "Option C", "User", "Task"),
        label_fields=("gt",),
    )
    assert summary["label_changed_pairs"] == 1
    assert summary["exact_surface_changed_label_candidates"] == 0


def test_length_mismatch_fails_closed():
    with pytest.raises(ValueError, match="different row counts"):
        paired_summary(
            _shopping_rows(),
            _shopping_rows()[:1],
            surface_fields=("Task",),
            label_fields=("gt",),
        )

