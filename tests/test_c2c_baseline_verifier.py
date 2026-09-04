import copy

import pytest

from scripts.run_c2c_baseline_dev import SPEC
from scripts.verify_c2c_baseline_dev import analyze, answer
from scripts.stage_c2c_assets import selected


def evidence():
    cases = [{"case_id": str(i), "dataset": "toy"} for i in range(20)]
    key = {c["case_id"]: "A" for c in cases}
    rows = [{"case_id": c["case_id"], "arm": arm, "completion": "A" if arm == "c2c" else "B",
             "hit_token_limit": False, "seconds": .1, "input_ids": [1, 2], "generated_ids": [3], "chat": "same"}
            for c in cases for arm in SPEC["arms"]]
    return cases, key, rows


def test_strict_parser_does_not_pick_explanation_letters():
    assert answer("The correct answer is B") == "B"
    assert answer("A.") == "A"
    assert answer("Because A is attractive, answer D") is None
    assert answer("A or B") is None


def test_controls_and_gain_do_not_become_paper_go():
    result = analyze(*evidence())
    assert result["functional_controls_pass"] is True
    assert result["decision"] == "BASELINE_SIGNAL_CONTINUE_PROTOCOL_DESIGN"
    assert "Not a paper go/no-go" in result["interpretation"]


def test_disabled_control_failure_prevents_advancement():
    cases, key, rows = evidence()
    next(r for r in rows if r["arm"] == "disabled_fuser")["completion"] = "D"
    assert analyze(cases, key, rows)["functional_controls_pass"] is False


@pytest.mark.parametrize("mutation", ["duplicate", "inputs", "cap"])
def test_corrupt_grid_or_budget_fails(mutation):
    cases, key, rows = evidence()
    if mutation == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    elif mutation == "inputs":
        rows[0]["input_ids"] = [7]
    else:
        rows[0]["hit_token_limit"] = True
    with pytest.raises(ValueError):
        analyze(cases, key, rows)


def test_only_selected_assets_are_downloaded():
    assert selected("model.safetensors", "receiver")
    assert not selected("nested/model.safetensors", "receiver")
    assert not selected("README.md", "receiver")
    assert selected("qwen3_0.6b+qwen3_4b_Fuser/final/projector_0.pt", "fuser")
    assert not selected("other_fuser/final/projector_0.pt", "fuser")
