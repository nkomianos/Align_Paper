from copy import deepcopy
import math

from interaction_sprint.agentuq_decision_events import events, examples, fit, score, evaluate


def test_content_and_outcomes_are_not_features():
    messages = [{"role": "assistant", "content": "greeting"},
                {"role": "user", "content": "hidden preference"},
                {"role": "assistant", "usage": {}, "tool_calls": [{"name": "get", "arguments": {"secret": 1}}]},
                {"role": "tool", "requestor": "assistant", "content": "result"}]
    labels = events(messages)
    assert labels == ["user:message", 'agent:tools:["get"]', "tool:assistant"]
    changed = deepcopy(messages)
    changed[1]["content"] = "other preference"
    changed[2]["tool_calls"][0]["arguments"] = {"secret": 0}
    assert events(changed) == labels


def test_scope_excludes_system_events_but_preserves_context():
    seq = ["user:message", "agent:text", "user:message", 'agent:tools:["get"]', "tool:assistant", "agent:text"]
    assert examples(seq, "agent_given_last_agent") == [("<START>", "agent:text"), ("agent:text", 'agent:tools:["get"]'), ('agent:tools:["get"]', "agent:text")]
    assert examples(seq, "tool_given_last_agent") == [("agent:text", 'agent:tools:["get"]')]


def test_unseen_label_and_context_backoff():
    model = fit([[('a', 'b')]])
    scored = score(model, [('unseen', 'unseen')])
    assert scored["unknown"] == 1
    assert math.isfinite(scored["markov_bits"])
    assert scored["markov_bits"] == scored["unigram_bits"]


def test_test_task_excluded_from_training_counts():
    rows = [{"task_id": str(i), "labels": ["user:message", "agent:text"]} for i in range(20)]
    result = evaluate(rows, "all_events")
    for f in result["fits"]:
        assert not set(f["train_ids"]) & set(f["test_ids"])
        assert f["unigram"]["agent:text"] == len(f["train_ids"])


def test_decision_metric_can_detect_real_context_signal():
    # Agent choices are perfectly related to distinct prior observations.
    # The decision-only metric must not mechanically force a null result.
    seq = examples(["user:message", "agent:text", "tool:assistant", 'agent:tools:["get"]']*20,
                   "agent_given_last_event")
    result = score(fit([seq]), seq)
    assert result["markov_accuracy"] == 1
    assert result["unigram_accuracy"] == .5
    assert result["markov_bits"] < result["unigram_bits"]
