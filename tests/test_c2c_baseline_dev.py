import copy

import pytest

from scripts.prepare_c2c_baseline_dev import select_rows


def rows():
    return [{"id": str(i), "question_stem": f"Question {i}",
             "choices": {"label": list("ABCD"), "text": ["a", "b", "c", "d"]},
             "answerKey": "A"} for i in range(20)]


def test_selection_does_not_depend_on_answers_or_source_order():
    original = rows()
    altered = copy.deepcopy(original[::-1])
    for row in altered:
        row["answerKey"] = "D"
    public, key, _ = select_rows(original, "openbookqa", 8)
    public2, key2, _ = select_rows(altered, "openbookqa", 8)
    assert public == public2
    assert key != key2
    assert all("answerKey" not in row for row in public)


def test_unsupported_numeric_labels_are_not_silently_scored():
    data = rows()
    data[0]["choices"]["label"] = ["1", "2", "3", "4"]
    public, _, audit = select_rows(data, "openbookqa", 19)
    assert audit["eligible_rows"] == 19
    assert "0" not in {r["source_id"] for r in public}


def test_duplicate_ids_fail():
    data = rows()
    data[1]["id"] = data[0]["id"]
    with pytest.raises(ValueError):
        select_rows(data, "openbookqa", 8)
