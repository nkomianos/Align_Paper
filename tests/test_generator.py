from __future__ import annotations

import json
import re
from collections import Counter

from under_extinction.generator import expected_evaluation_count, generate_datasets
from under_extinction.io import read_jsonl, sha256_file
from under_extinction.renderers import renderers_for
from under_extinction.schema import Controller, Intervention, other_action


def test_generation_is_deterministic_and_split_safe(tiny_config, tmp_path):
    first = tmp_path / "one"
    second = tmp_path / "two"
    manifest_one = generate_datasets(tiny_config, first)
    manifest_two = generate_datasets(tiny_config, second)
    assert manifest_one["counts"] == manifest_two["counts"]
    assert manifest_one["counts"]["evaluation"] == expected_evaluation_count(tiny_config)
    for name in ("train.jsonl", "dev.jsonl", "evaluation.jsonl"):
        assert sha256_file(first / name) == sha256_file(second / name)
    renderer_sets = [set(manifest_one["renderer_splits"][split]) for split in ("train", "dev", "test")]
    assert not (renderer_sets[0] & renderer_sets[1])
    assert not (renderer_sets[0] & renderer_sets[2])
    assert not (renderer_sets[1] & renderer_sets[2])


def test_training_decisions_are_balanced_and_have_no_controller_prompt(tiny_config, tmp_path):
    generate_datasets(tiny_config, tmp_path)
    decisions = [row for row in read_jsonl(tmp_path / "train.jsonl") if row["task_type"] == "static_decision"]
    conditions = Counter(row["condition"] for row in decisions)
    assert abs(conditions["consensus"] - conditions["factorial_conflict"]) <= 1
    for row in decisions:
        prompt = "\n".join(message["content"] for message in row["messages"]).lower()
        assert "intended controller" not in prompt
        assert "proxy controller" not in prompt
        assert "cached controller" not in prompt
        assert set(row["oracle_actions"]) == {item.value for item in Controller}


def test_interventions_are_selective_and_shams_are_inert(tiny_config, tmp_path):
    generate_datasets(tiny_config, tmp_path)
    rows = list(read_jsonl(tmp_path / "evaluation.jsonl"))
    by_id = {row["record_id"]: row for row in rows}
    real_rows = [
        row for row in rows
        if row.get("eval_group") == "audit"
        and row.get("intervention", {}).get("active")
        and row["renderer_id"] == renderers_for("test")[0].renderer_id
    ]
    assert real_rows
    expected_sensitive = {
        Intervention.GENUINE_VALUE.value: Controller.INTENDED.value,
        Intervention.GENUINE_CONTINGENCY.value: Controller.INTENDED.value,
        Intervention.PROXY_VALUE.value: Controller.PROXY.value,
        Intervention.PROXY_CONTINGENCY.value: Controller.PROXY.value,
        Intervention.CUE_SWAP.value: Controller.CACHED.value,
    }
    for real in real_rows:
        control = by_id[real["paired_control_id"]]
        baseline = by_id[real["baseline_id"]]
        assert control["world"] != baseline["world"]
        assert control["oracle_actions"] == baseline["oracle_actions"]
        family = real["intervention"]["family"]
        sensitive = expected_sensitive[family]
        for controller in (item.value for item in Controller):
            if controller == sensitive:
                assert real["oracle_actions"][controller] == other_action(baseline["oracle_actions"][controller])
            else:
                assert real["oracle_actions"][controller] == baseline["oracle_actions"][controller]
        if family != Intervention.CUE_SWAP.value:
            bulletin = real["intervention"]["bulletin"]
            assert not re.search(r"\b(?:A|B)\b", bulletin)
            assert not real["intervention"]["mentions_action"]


def test_each_real_update_has_syntax_matched_control(tiny_config, tmp_path):
    generate_datasets(tiny_config, tmp_path)
    rows = list(read_jsonl(tmp_path / "evaluation.jsonl"))
    by_id = {row["record_id"]: row for row in rows}
    for real in (
        row for row in rows
        if row.get("eval_group") == "audit" and row.get("intervention", {}).get("active")
    ):
        control = by_id[real["paired_control_id"]]
        assert control["condition"] == "sham"
        assert control["intervention"]["family"] == real["intervention"]["family"]
        real_words = len(real["intervention"]["bulletin"].split())
        control_words = len(control["intervention"]["bulletin"].split())
        assert abs(real_words - control_words) <= 7
