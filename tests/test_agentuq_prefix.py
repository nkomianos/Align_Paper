from copy import deepcopy

import pytest

from interaction_sprint.agentuq_prefix import prefix_features, crossfit


def action(content="hello", calls=None):
    return {"role": "assistant", "content": content, "usage": {"completion_tokens": 4}, "tool_calls": calls}


def test_future_and_simulator_reasoning_do_not_affect_features():
    prefix = [{"role": "assistant", "content": "fixed greeting"},
              {"role": "user", "content": "hi", "raw_data": {"reasoning_content": "SECRET"}}, action()]
    expected = prefix_features(prefix, 1)
    assert prefix_features(prefix+[action("future"*10000)], 1) == expected
    mutated = deepcopy(prefix)
    mutated[1]["raw_data"]["reasoning_content"] = "other answer"
    assert prefix_features(mutated, 1) == expected
    assert prefix_features(prefix, 2) is None


def test_tool_feedback_and_duplicate_arguments():
    call = {"id": "a", "name": "get", "arguments": {"b": 1, "a": 2}}
    seq = [action(calls=[call]), {"role": "tool", "id": "a", "requestor": "assistant", "error": True, "content": "Error: invalid"}]
    x = prefix_features(seq, 1)
    assert x["error_response_fraction"] == 1
    seq += [action(calls=[{"id": "b", "name": "get", "arguments": {"a": 2, "b": 1}}])]
    assert prefix_features(seq, 2)["repeated_call_fraction"] == .5
    assert prefix_features(seq, 1) == x


def test_unmatched_tool_and_missing_usage_fail():
    with pytest.raises(ValueError):
        prefix_features([action(), {"role": "tool", "id": "x", "requestor": "assistant"}], 1)
    with pytest.raises(ValueError):
        prefix_features([{"role": "assistant", "raw_data": {}, "usage": {}}], 1)


def test_heldout_outcome_does_not_change_own_prediction():
    rows = [{"task_id": str(i), "features": {"x": i % 2}, "failure": i % 2} for i in range(30)]
    result = crossfit(rows, ["x"])
    assert result["auroc"] == 1
    changed = deepcopy(rows)
    changed[0]["failure"] = 1
    assert crossfit(changed, ["x"])["per_task"][0]["prediction"] == result["per_task"][0]["prediction"]
