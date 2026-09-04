import numpy as np
import pytest

from interaction_sprint.hindsight_human_feedback import (
    arm_texts,
    cluster_bootstrap_gain,
    query_split,
)


def test_arm_views_exclude_scripted_query_from_user_feedback():
    turns = [
        {"role": "USER", "content": "scripted"},
        {"role": "BOT", "content": "assistant one"},
        {"role": "USER", "content": "participant one"},
        {"role": "BOT", "content": "assistant two"},
        {"role": "USER", "content": "participant two"},
    ]
    views = arm_texts("statement", "query", turns, 3)
    assert "participant one" in views["user_only"]
    assert "participant two" in views["user_only"]
    assert "scripted" not in views["user_only"]
    assert "assistant one" not in views["user_only"]
    assert "participant one" not in views["assistant_only"]
    assert all("statement" in value and "query" in value for value in views.values())


def test_query_split_is_deterministic_and_disjoint():
    first = query_split([f"q{i}" for i in range(9)])
    second = query_split(list(reversed([f"q{i}" for i in range(9)])))
    assert first == second
    assert first[0].isdisjoint(first[1])
    assert len(first[0]) == 3


def test_cluster_bootstrap_detects_strict_improvement():
    target = np.arange(12, dtype=float)
    reference = target + 2
    candidate = target.copy()
    groups = np.repeat(["a", "b", "c", "d"], 3)
    result = cluster_bootstrap_gain(target, reference, candidate, groups, draws=1000)
    assert result["mse_gain"] == 4
    assert result["relative_mse_gain"] == 1
    assert result["cluster_bootstrap_ci95_low"] > 0
    with pytest.raises(ValueError):
        cluster_bootstrap_gain(target, reference[:-1], candidate, groups)

