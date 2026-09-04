import pytest

from interaction_sprint.hindsight_gradient_power import run_power_audit


def test_power_audit_has_expected_grid_and_overlap():
    result = run_power_audit(seed=7, replicates=2, dimension=8)
    assert len(result["cells"]) == 28
    assert len(result["panel_overlap"]) == 8
    assert all(row[index] == 8 for index, row in enumerate(result["panel_overlap"]))
    assert max(value for i, row in enumerate(result["panel_overlap"])
               for j, value in enumerate(row) if i != j) <= 2


def test_power_audit_is_deterministic_and_bounded():
    left = run_power_audit(seed=11, replicates=2, dimension=8)
    right = run_power_audit(seed=11, replicates=2, dimension=8)
    assert left == right
    for cell in left["cells"]:
        assert 0 <= cell["qualified_rate"] <= 1
        assert all(0 <= value <= 1 for value in cell["gate_pass_rates"].values())


def test_power_audit_rejects_invalid_shape():
    with pytest.raises(ValueError):
        run_power_audit(replicates=0)
