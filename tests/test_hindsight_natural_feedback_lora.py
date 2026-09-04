import pytest

from interaction_sprint.hindsight_natural_feedback_lora import basin_side, natural_metrics


def test_basin_side():
    assert basin_side(0.6, 0.5) == 1
    assert basin_side(0.4, 0.5) == -1
    with pytest.raises(ValueError):
        basin_side(0.5, 0.5)


def test_natural_metrics_respect_predicted_side():
    contexts = []
    initial = []
    dynamic = []
    fixed = []
    for index, value in enumerate((0.4, 0.6, 0.3, 0.7)):
        key = f"x-{index}"
        contexts.append(dict(base_id=key, split="confirmation", bistable=True, middle=0.5))
        initial.append(dict(base_id=key, probability_one=value))
        dynamic.append(dict(base_id=key, probability_one=0.01 if value < 0.5 else 0.99))
        fixed.append(dict(base_id=key, probability_one=0.45 if value < 0.5 else 0.55))
    result = natural_metrics(contexts, initial, dynamic, fixed)
    assert result["n"] == 4
    assert result["positive_advantage_fraction"] == 1
    assert result["dynamic_basin_consistency"] == 1
    assert result["upper_median_signed_dynamic_advantage"] == pytest.approx(0.44)
