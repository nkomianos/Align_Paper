"""Synthetic fixtures only: no access to human source files or endpoint data."""
import csv
import importlib.util
import math
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/reconstruct_deepcanvassing_measured_responses_20260905.py"
SPEC = importlib.util.spec_from_file_location("deepcanvassing_reconstruction_20260905", SCRIPT)
m = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = m
SPEC.loader.exec_module(m)


def items(wave, value=20):
    return {m.source_column(wave, item): value
            for definitions in m.SCALES.values() for item, _ in definitions}


def test_wave_binding_and_reverse_scoring():
    assert m.source_column("baseline", "crime") == "t1_imm_prej_crime_1"
    assert m.source_column("recontact", "crime") == "t2_imm_prej_crimes_1"
    assert m.source_column("immediate", "crime") == m.source_column("recontact", "crime")
    prejudice = m.score_scale(items("baseline"), "baseline", "prejudice")
    policy = m.score_scale(items("immediate"), "immediate", "policy")
    assert prejudice.complete == 40
    assert policy.complete == 44


def test_complete_items_and_sharp_missing_item_bounds():
    row = items("recontact")
    row[m.source_column("recontact", "values")] = ""
    score = m.score_scale(row, "recontact", "prejudice")
    assert score.complete is None and score.n_observed == 5
    assert score.lower == pytest.approx(160 / 6)
    assert score.upper == pytest.approx(260 / 6)
    missing = m.score_scale(dict.fromkeys(row), "recontact", "prejudice")
    assert missing == m.Score(None, 0, 0, 100)
    with pytest.raises(KeyError):
        m.score_scale({}, "recontact", "prejudice")


@pytest.mark.parametrize("value", ["oops", "NaN", "Inf", math.inf, -1, 101, True])
def test_malformed_observed_items_fail(value):
    with pytest.raises(ValueError):
        m.parse_item(value)


def test_missing_values_and_valid_endpoints():
    assert all(m.parse_item(v) is None for v in [None, "", "  ", float("nan")])
    assert m.parse_item("0") == 0 and m.parse_item("100") == 100


def test_join_preserves_cohort_and_source_wave_without_identifiers():
    ids = ["a" * 24, "b" * 24, "c" * 24]
    base = [{**items("baseline", 10), **items("immediate", 20),
             "prolific_pid": pid, "condition": arm, "Experimental_Condition": code}
            for pid, arm, code in [(ids[0], "control", 0), (ids[1], "treatment", 1)]]
    follow = {ids[0]: items("recontact", 30), ids[2]: items("recontact", 40)}
    rows, counts = m.reconstruct_rows(base, follow)
    assert len(rows) == 12 and counts["followup_outside_current_cohort"] == 1
    control_late = next(r for r in rows if r["participant_ordinal"] == 1 and r["wave"] == "recontact" and r["scale"] == "prejudice")
    assert control_late["score_complete"] == pytest.approx(130 / 3)
    treatment_late = [r for r in rows if r["participant_ordinal"] == 2 and r["wave"] == "recontact"]
    assert all(r["score_complete"] is None and r["score_lower"] == 0 and r["score_upper"] == 100 for r in treatment_late)
    assert all(not any(pid in str(row) for pid in ids) for row in rows)
    available = m.completeness_receipt(rows)
    assert available["complete_immediate_and_recontact"]["prejudice/0"] == 1
    assert available["complete_immediate_and_recontact"]["prejudice/1"] == 0
    assert available["by_wave_scale_arm"]["recontact/prejudice/1"]["no_items"] == 1
    with pytest.raises(ValueError, match="duplicate"):
        m.reconstruct_rows(base + [base[0]], follow)
    base[0]["Experimental_Condition"] = 1
    with pytest.raises(ValueError, match="disagree"):
        m.reconstruct_rows(base, follow)


def test_metadata_rows_and_duplicate_followup_fail(tmp_path):
    file = tmp_path / "synthetic.csv"
    fields = ["ResponseId", "PROLIFIC_PID", *items("recontact")]
    with file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"ResponseId": "Response ID"})
        writer.writerow({"ResponseId": '{"ImportId":"_recordId"}'})
        writer.writerow({"ResponseId": "synthetic", "PROLIFIC_PID": "a" * 24, **items("recontact")})
    records, metadata = m.load_followup(file)
    assert metadata == 2 and len(records) == 1
    assert "ResponseId" not in records["a" * 24]
    with file.open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fields).writerow({"ResponseId": "synthetic", "PROLIFIC_PID": "a" * 24, **items("recontact")})
    with pytest.raises(ValueError, match="duplicate"):
        m.load_followup(file)


def test_joint_contrast_bounds_are_sharp_on_synthetic_vertices():
    rows = [
        {"arm": 0, "immediate": (30, 40), "recontact": (10, 60)},
        {"arm": 1, "immediate": (20, 20), "recontact": (0, 100)},
    ]
    assert m.change_in_arm_contrast_bounds(rows) == (-50, 110)
    # Achieve each endpoint by assigning every missing cell at the appropriate
    # bound; no stochastic or cross-wave restriction is silently imposed.
    import itertools
    vertices = []
    for c_i, c_l, t_i, t_l in itertools.product((30, 40), (10, 60), (20,), (0, 100)):
        vertices.append((t_l - c_l) - (t_i - c_i))
    assert (min(vertices), max(vertices)) == (-50, 110)
    complete = [{"arm": 0, "immediate": (30, 30), "recontact": (40, 40)},
                {"arm": 1, "immediate": (20, 20), "recontact": (15, 15)}]
    assert m.change_in_arm_contrast_bounds(complete) == (-15, -15)
