import pytest

from scripts.audit_c2c_sender_utility import compare


def fixture_rows():
    return [{"case_id": str(i), "arm": arm, "completion": answer, "seconds": seconds}
            for i, answers in enumerate((("A", "A"), ("A", "B"), ("B", "A"), ("B", "B")))
            for arm, answer, seconds in zip(("sender", "c2c"), answers, (1, 2))]


def test_oracle_is_union_not_sum_and_does_not_hide_harm():
    out = compare(fixture_rows(), {str(i): "A" for i in range(4)}, lambda x: x)
    assert out["both_correct"] == out["neither_correct"] == 1
    assert out["sender_only_correct"] == out["c2c_only_correct"] == 1
    assert out["answer_key_oracle_correct"] == 3
    assert out["oracle_gain_over_sender"] == .25
    assert out["c2c_minus_sender_accuracy"] == 0
    assert out["recorded_generation_seconds"] == {"sender": 4, "c2c": 8}


@pytest.mark.parametrize("change", ["missing", "duplicate", "nan", "negative"])
def test_invalid_evidence_rejected(change):
    rows = fixture_rows()
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(rows[0])
    else:
        rows[0]["seconds"] = float("nan") if change == "nan" else -1
    with pytest.raises(ValueError):
        compare(rows, {str(i): "A" for i in range(4)}, lambda x: x)
