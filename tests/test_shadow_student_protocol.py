import json
from pathlib import Path

import pytest

from shadow_student_audit.preflight import load_public_config
from shadow_student_audit.protocol import build_scenario_plan, make_probe_records, make_training_records, plan_commitment
from shadow_student_audit.runner import build_answer_key


def test_frozen_plan_has_balanced_sealed_and_calibration_conditions(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """schema_version: '1.0'
kind: sentry_shadow_student_g0
model: {id: Qwen/Qwen3.5-9B, revision: c202236235762e1c871ad0ccb60c8ee5ba337b9a, dtype: bfloat16}
public_sources:
  - {id: one, revision: 1111111111111111111111111111111111111111, url: https://example.test/1, purpose: test}
  - {id: two, revision: 2222222222222222222222222222222222222222, url: https://example.test/2, purpose: test}
  - {id: three, revision: 3333333333333333333333333333333333333333, url: https://example.test/3, purpose: test}
training: {full_lora_rank: 32, shadow_lora_rank: 4, full_token_budget: 1920000, shadow_token_budget: 240000, full_seeds: [1, 2], shadow_seeds: [3, 4, 5, 6]}
protocol: {source_rows_per_scenario: 2, scenario_strengths: [0.0, 0.2, 0.4, 0.8], calibration_per_channel: 4, sealed_per_channel: 4}
sealed_test: {answer_key_path: null}
""",
        encoding="utf-8",
    )
    config = load_public_config(config_path)
    plan = build_scenario_plan(config["protocol"])
    assert len(plan) == 16
    assert sum(item.split == "sealed" for item in plan) == 8
    assert {item.channel for item in plan if item.split == "sealed"} == {"vocabulary", "body"}
    assert plan_commitment(plan) == plan_commitment(plan)


def test_records_hide_scenario_membership_and_probes_are_balanced():
    plan = build_scenario_plan({"source_rows_per_scenario": 2, "scenario_strengths": [0.0, 0.2, 0.4, 0.8], "calibration_per_channel": 4, "sealed_per_channel": 4})[0]
    rows = make_training_records(plan, ["What is 2 + 2?", "Write a loop."], rows=2, seed=7)
    assert plan.scenario_id not in json.dumps(rows)
    assert plan.split not in json.dumps(rows)
    vocabulary = make_probe_records("vocabulary", rows=3)
    assert all(row["positive"] != row["neutral"] for row in vocabulary)


def test_answer_key_refuses_overwrite_and_binds_plan(tmp_path):
    source = Path(__file__).resolve().parents[1] / "configs" / "sentry_g0.yaml"
    destination = tmp_path / "key.json"
    result = build_answer_key(source, destination)
    assert result["kind"] == "sentry_g0_answer_key"
    with pytest.raises(FileExistsError):
        build_answer_key(source, destination)
