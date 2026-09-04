import math
from collections import Counter

import pytest

from interaction_sprint.hindsight_bifurcation_g0 import (
    analyze,
    dataset,
    endpoint_pair,
    fixed_points,
    reverse_kl_map,
)


def test_fresh_crossed_dataset():
    rows = dataset()
    assert len(rows) == 64
    assert len({row["id"] for row in rows}) == 64
    assert len({row["base_id"] for row in rows}) == 32
    assert Counter(row["split"] for row in rows) == {"development": 32, "confirmation": 32}
    for row in rows:
        assert row["options"][row["target"]] == row["preferred"]


def test_symmetric_threshold_and_fixed_control():
    weak = fixed_points(-1.5, 1.5)
    strong = fixed_points(-3.0, 3.0)
    assert len(weak) == 1 and weak[0]["stable"]
    assert len(strong) == 3
    assert [root["stable"] for root in strong] == [True, False, True]
    pair = endpoint_pair(-3.0, 3.0, strong[1]["probability"])
    assert pair["dynamic_separation"] > 0.8
    assert pair["fixed_marginal_separation"] < 0.05
    assert reverse_kl_map(0.5, -3.0, 3.0) == pytest.approx(0.5)


def test_asymmetry_can_remove_bifurcation():
    roots = fixed_points(-10.0, 1.0)
    assert len(roots) == 1
    assert roots[0]["probability"] < 0.01


def test_analysis_gate_is_prospective_and_requires_confirmation_coverage():
    rows = []
    for case in dataset():
        if case["target"] == 0:
            probabilities = [1 / (1 + math.exp(-3)), 1 / (1 + math.exp(3))]
        else:
            probabilities = [1 / (1 + math.exp(3)), 1 / (1 + math.exp(-3))]
        rows.append(dict(case, kind="hindsight", probabilities=probabilities, prediction=case["target"]))
    result = analyze(rows)
    assert result["status"] == "BIFURCATION_SCREEN_POSITIVE"
    assert result["splits"]["confirmation"]["bistable"] == 16
    for row in rows:
        if row["split"] == "confirmation" and row["domain"] == "screen brightness" and row["target"] == 1:
            row["probabilities"] = [0.99, 0.01]
            row["prediction"] = 0
    assert analyze(rows)["status"] == "BIFURCATION_SCREEN_NEGATIVE"
