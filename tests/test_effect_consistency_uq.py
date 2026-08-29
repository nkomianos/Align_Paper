from __future__ import annotations

import json

from effect_consistency_uq.analysis import GateThresholds, evaluate_gate, score_family
from effect_consistency_uq.corpus import build_corpus, load_cases
from effect_consistency_uq.environment import Action, execute_plan, parse_plan
from under_extinction.io import read_jsonl


def test_aliases_and_read_only_calls_have_the_same_exact_effect() -> None:
    state = {"events": {}, "timezone": "UTC"}
    direct = execute_plan("calendar", state, [Action("create_event", {"event_id": "e", "person": "Ari", "start": 10, "duration": 1})])
    alternate = execute_plan("calendar", state, [
        Action("check_calendar", {}),
        Action("book_slot", {"event_id": "e", "person": "Ari", "start": 10, "duration": 1}),
    ])
    assert direct.valid and alternate.valid
    assert direct.effect == alternate.effect


def test_parser_accepts_fenced_actions_and_rejects_empty_plan() -> None:
    plan = parse_plan('```json\n[{"tool":"check_stock","arguments":{}}]\n```')
    assert plan == (Action("check_stock", {}),)
    try:
        parse_plan("[]")
    except ValueError as exc:
        assert "one to four" in str(exc)
    else:
        raise AssertionError("empty plans must fail")


def _oracle_plan(case):
    index = int(case.task_id.split("-")[1])
    if case.domain == "calendar":
        if index % 3 == 0:
            person = ["Ari", "Bo", "Chen", "Dara", "Eli"][index % 5]
            return [{"tool": "create_event", "arguments": {"event_id": f"new{index}", "person": person, "start": 10 + index % 2, "duration": 1}}]
        if index % 3 == 1:
            return [{"tool": "move_event", "arguments": {"event_id": f"e{index}a", "start": 15 + index % 2}}]
        return [{"tool": "cancel_event", "arguments": {"event_id": f"e{index}b"}}]
    sku_a, sku_b = f"SKU-{index % 17:02d}", f"SKU-{(index + 5) % 17:02d}"
    if index % 3 == 0:
        return [{"tool": "reserve_stock", "arguments": {"reservation_id": f"r{index}", "sku": sku_a, "quantity": 2 + index % 3}}]
    if index % 3 == 1:
        return [{"tool": "release_stock", "arguments": {"reservation_id": f"old{index}"}}]
    return [{"tool": "transfer_stock", "arguments": {"source_sku": sku_a, "target_sku": sku_b, "quantity": 1 + index % 2}}]


def _alias(plan, domain):
    aliases = {
        "create_event": "book_slot", "move_event": "reschedule", "cancel_event": "delete_event",
        "reserve_stock": "hold_stock", "release_stock": "unhold_stock", "transfer_stock": "move_stock",
    }
    converted = [{"tool": aliases[item["tool"]], "arguments": item["arguments"]} for item in plan]
    read = {"tool": "check_calendar" if domain == "calendar" else "check_stock", "arguments": {}}
    return [read, *converted]


def test_frozen_effect_gate_scores_execution_equivalence(tmp_path) -> None:
    public, key_path = tmp_path / "public.jsonl", tmp_path / "key.jsonl"
    build_corpus(public, key_path, count_per_domain=40)
    cases = load_cases(public)
    key = {row["task_id"]: row["oracle_effect"] for row in read_jsonl(key_path)}
    raw = []
    for case in cases:
        index = int(case.task_id.split("-")[1])
        correct = (index // 2) % 2 == 0
        oracle = _oracle_plan(case)
        read = [{"tool": "check_calendar" if case.domain == "calendar" else "check_stock", "arguments": {}}]
        completions = [oracle, oracle, _alias(oracle, case.domain), _alias(oracle, case.domain)] if correct else [read, read, [{"tool": "unknown", "arguments": {}}], oracle]
        for sample_id, completion in enumerate(completions):
            raw.append({"task_id": case.task_id, "sample_id": sample_id, "completion": json.dumps(completion), "token_confidence": .5})
    report = score_family(cases, raw, key, thresholds=GateThresholds(bootstrap_replicates=1000))
    assert report["domains"]["calendar"]["effect_auc"] == 1.0
    assert report["domains"]["inventory"]["auc_gap"]["point"] >= .05
    gate = evaluate_gate({"family_a": report, "family_b": report}, thresholds=GateThresholds(bootstrap_replicates=1000))
    # The synthetic fixture isolates discrimination, not the action-vote gain;
    # the all-criteria rule must retain the latter as a separate requirement.
    assert "family_a/calendar/vote" in gate["checks"]
