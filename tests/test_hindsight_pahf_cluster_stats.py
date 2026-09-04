import pytest

from interaction_sprint.hindsight_pahf_cluster_stats import (
    bootstrap_mean_interval,
    paired_cluster_summary,
)


def _rows(offset: float, *, bases: int = 8) -> list[dict[str, object]]:
    return [
        {
            "base_id": f"base-{base}",
            "rotation": rotation,
            "old_target_log_loss": 1.0 + offset + 0.01 * rotation,
            "old_target_correct": (base + rotation) % 2 == 0,
        }
        for base in range(bases) for rotation in range(4)
    ]


def test_clustered_summary_uses_base_not_rotation_as_unit() -> None:
    summary = paired_cluster_summary(_rows(0.0), _rows(-0.1), samples=1000)
    assert summary["aggregate"]["base_clusters"] == 8
    assert summary["aggregate"]["rotation_rows_per_arm"] == 32
    assert summary["aggregate"]["mean_old_target_nll_gain"] == pytest.approx(0.1)
    assert summary["decision"] == "ENDO_PAHF_CLUSTERED_COMPARISON_QUALIFIED"


def test_clustered_summary_rejects_duplicate_or_incomplete_rotations() -> None:
    anchor = _rows(0.0)
    with pytest.raises(ValueError, match="four rotations"):
        paired_cluster_summary(anchor[:-1], _rows(-0.1), samples=1000)
    with pytest.raises(ValueError, match="duplicate"):
        paired_cluster_summary(anchor + [anchor[0]], _rows(-0.1), samples=1000)


def test_bootstrap_interval_is_deterministic_and_order_invariant() -> None:
    values = [float(index) for index in range(20)]
    first = bootstrap_mean_interval(values, samples=1000, seed=7)
    second = bootstrap_mean_interval(list(reversed(values)), samples=1000, seed=7)
    assert first == pytest.approx(second, abs=0.2)
    assert first == bootstrap_mean_interval(values, samples=1000, seed=7)


def test_small_or_nonfinite_inputs_fail_closed() -> None:
    with pytest.raises(ValueError):
        bootstrap_mean_interval([1.0], samples=1000)
    with pytest.raises(ValueError):
        bootstrap_mean_interval([1.0, float("nan")], samples=1000)
