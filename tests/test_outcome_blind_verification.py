from __future__ import annotations

import pytest

from outcome_blind_verification import TraceItem, Verdict, assess_gate, build_prompt
from outcome_blind_verification.runner import _validate_runner_record, parse_verdict


def _item() -> TraceItem:
    return TraceItem("x", "Compute 2 + 2.", "2 + 2 = 4.", "4", True, "development")


def test_prompt_changes_only_dedicated_outcome_field() -> None:
    item = _item()
    visible = build_prompt(item, outcome_visible=True, label_order=("PROCESS_VALID", "PROCESS_INVALID"))
    blind = build_prompt(item, outcome_visible=False, label_order=("PROCESS_VALID", "PROCESS_INVALID"))
    assert item.derivation in visible and item.derivation in blind
    assert "Stated final answer:\n4" in visible
    assert "[OUTCOME WITHHELD FOR PROCESS REVIEW]" in blind


def test_prompt_rejects_interface_drift() -> None:
    with pytest.raises(ValueError, match="label_order"):
        build_prompt(_item(), outcome_visible=True, label_order=("PROCESS_VALID", "PROCESS_VALID"))


def test_runner_record_cannot_contain_process_label() -> None:
    record = _item().runner_record()
    assert "process_valid" not in record
    assert _item().answer_key_record()["process_valid"] is True


def test_runner_rejects_a_leaked_label_and_prose_verdict() -> None:
    with pytest.raises(ValueError, match="leaks"):
        _validate_runner_record(_item().to_dict())
    assert parse_verdict("PROCESS_VALID") is Verdict.VALID
    assert parse_verdict("PROCESS_VALID because the arithmetic is correct") is Verdict.UNPARSEABLE


def test_gate_passes_only_for_detection_gain_without_validity_loss() -> None:
    answer_key = {f"invalid-{i}": False for i in range(100)} | {f"valid-{i}": True for i in range(100)}
    visible = {item_id: Verdict.VALID for item_id in answer_key}
    blind = dict(visible)
    for index in range(70):
        blind[f"invalid-{index}"] = Verdict.INVALID
    report = assess_gate(answer_key, visible, blind, bootstrap_samples=1_000)
    assert report.pass_gate is True
    assert report.invalid_detection_gain == pytest.approx(0.70)


def test_gate_rejects_detection_gain_that_breaks_valid_acceptance() -> None:
    answer_key = {f"invalid-{i}": False for i in range(100)} | {f"valid-{i}": True for i in range(100)}
    visible = {item_id: Verdict.VALID for item_id in answer_key}
    blind = dict(visible)
    for index in range(70):
        blind[f"invalid-{index}"] = Verdict.INVALID
    for index in range(10):
        blind[f"valid-{index}"] = Verdict.INVALID
    report = assess_gate(answer_key, visible, blind, bootstrap_samples=1_000)
    assert report.pass_gate is False
    assert report.valid_acceptance_loss == pytest.approx(0.10)
