import copy

import pytest

from latent_contract.update_comparison import ARMS, analyze, paired_interval
from scripts.verify_c2c_paired_update import combine


def fixture():
    cases = [{"case_id": str(i), "dataset": "book" if i % 2 else "arc"} for i in range(40)]
    key = {c["case_id"]: "A" for c in cases}
    rows = []
    for i, case in enumerate(cases):
        direct = [{"role": "user", "content": "Full question"}]
        query = [{"role": "user", "content": "Background query"}]
        for arm in ARMS:
            accuracy_count = 20 if arm == "receiver" or arm.endswith("_disabled") or arm == "new_c2c" else 36 if arm.endswith("_sender") else 32
            text = "A. correct" if i < accuracy_count else "B. incorrect"
            turns = direct
            if arm.endswith("_background"):
                text, turns = "Some background", query
            elif arm.endswith("_text"):
                turns = [*query, {"role": "assistant", "content": "Some background"}, *direct]
            rows.append({**case, "arm": arm, "completion": text, "messages": turns,
                         "input_ids": [1, 2], "generated_ids": [3 if text.startswith("A") else 4],
                         "hit_token_limit": False, "seconds": 1.0})
    return cases, key, rows


def test_paired_contrast_and_full_text_cost():
    report = analyze(*fixture())
    assert report["decision"] == "EXTRA_LATENT_LOSS_SIGNAL_REPAIR_STUDY_NEEDED"
    assert report["extra_latent_change_pp"] == pytest.approx(-30)
    assert report["paired_question_bootstrap_95_pp"][1] < 0
    assert report["metrics"]["new_text"]["serial_generation_seconds"] == 80
    assert report["metrics"]["new_c2c"]["serial_generation_seconds"] == 40


def test_equal_channel_loss_is_not_a_latent_specific_effect():
    cases, key, rows = fixture()
    by_id = {(r["case_id"], r["arm"]): r for r in rows}
    for case in cases:
        by_id[case["case_id"], "new_text"]["completion"] = by_id[case["case_id"], "new_c2c"]["completion"]
    report = analyze(cases, key, rows)
    assert report["extra_latent_change_pp"] == 0
    assert report["decision"] == "NO_REQUIRED_EXTRA_LATENT_LOSS_ON_THIS_UPDATE"


def test_sender_capability_loss_prevents_interface_claim():
    cases, key, rows = fixture()
    for row in rows:
        if row["arm"] == "new_sender":
            row["completion"] = "B"
    assert analyze(cases, key, rows)["decision"] == "SENDER_CAPABILITY_CHANGE_CONFOUNDS_INTERPRETATION"


@pytest.mark.parametrize("kind", ["duplicate", "prompt", "transfer", "time", "cap"])
def test_invalid_evidence_rejected(kind):
    cases, key, rows = fixture()
    if kind == "duplicate":
        rows[-1] = dict(rows[0])
    elif kind == "prompt":
        rows[0]["input_ids"] = [7]
    elif kind == "transfer":
        next(r for r in rows if r["arm"] == "new_text")["messages"] = []
    elif kind == "time":
        rows[0]["seconds"] = float("nan")
    else:
        rows[0]["hit_token_limit"] = True
    with pytest.raises(ValueError):
        analyze(cases, key, rows)


def test_ci_pairs_observations_and_preserves_dataset_strata():
    assert paired_interval([0, 0, 0, 0], ["a", "a", "b", "b"]) == [0, 0]
    assert paired_interval([-1, -1, 1, 1], ["a", "a", "b", "b"]) == [0, 0]


def test_two_seed_decision_cannot_select_one_favorable_update():
    cases, key, rows = fixture()
    report = analyze(cases, key, rows)
    first = {**report, "seed": 202609041, "ticket_sha256": "same"}
    second = {**report, "seed": 202609042, "ticket_sha256": "same"}
    assert combine([first, second], [rows, rows])["decision"] == "REPEATED_EXTRA_LATENT_LOSS_REPAIR_STUDY_NEEDED"
    second["decision"] = "NO_REQUIRED_EXTRA_LATENT_LOSS_ON_THIS_UPDATE"
    assert combine([first, second], [rows, rows])["decision"] == "NO_REPEATED_QUALIFIED_SIGNAL_DO_NOT_EXPAND"
    with pytest.raises(ValueError):
        combine([first, first], [rows, rows])
