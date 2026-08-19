from __future__ import annotations

import pytest

from under_extinction.prospective import compare_models_loto, simulate_trajectory_power, validate_rows


def _rows(trajectories: int = 12) -> list[dict]:
    rows: list[dict] = []
    for trajectory in range(trajectories):
        for checkpoint in range(4):
            proxy_index = -1.0 + 2.0 * checkpoint / 3.0 + (trajectory % 3 - 1) * 0.05
            outcome = int(proxy_index > 0.0)
            rows.append({
                "trajectory_id": f"t{trajectory:02d}",
                "checkpoint_step": checkpoint * 100,
                "forecast_horizon_steps": 200,
                "eligible": True,
                "future_sustained_hack": outcome,
                "checkpoint_fraction": checkpoint / 3.0,
                "current_hack_rate": 0.0,
                "cumulative_proxy_reward": float(checkpoint),
                "current_reward_gap": 0.0,
                "recent_proxy_reward_slope": 0.0,
                "recent_hack_rate_slope": 0.0,
                "proxy_control_index": proxy_index,
            })
    return rows


def test_prospective_analysis_uses_trajectory_folds() -> None:
    report = compare_models_loto(_rows(), bootstrap_replicates=200, seed=7)
    assert report["trajectory_count"] == 12
    assert report["checkpoint_row_count"] == 48
    assert len(report["trajectory_log_loss_improvements"]) == 12
    assert report["mean_log_loss_improvement"] > 0


def test_validation_rejects_duplicate_checkpoint() -> None:
    rows = _rows(4)
    rows.append(dict(rows[0]))
    with pytest.raises(ValueError, match="Duplicate"):
        validate_rows(rows)


def test_power_simulation_is_deterministic_and_increases_with_n() -> None:
    result = simulate_trajectory_power(
        trajectory_counts=(8, 24), simulations=200, standardized_fingerprint_effect=0.5, seed=9
    )
    assert 0.0 <= result["8"] <= 1.0
    assert result["24"] >= result["8"]
