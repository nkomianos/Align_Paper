"""Regression checks for corrections to the scientific statement."""
import importlib.util
from pathlib import Path
from fractions import Fraction as F

spec = importlib.util.spec_from_file_location(
    "corrected_theory", Path(__file__).resolve().parents[1]
    / "scripts/verify_hindsight_corrected_theory_20260905.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_deterministic_data_dependence_refutes_old_point_two_bound():
    risks = module.risk(F(2, 5))
    assert risks == (F(3, 25), F(4, 25))
    assert max(risks) < F(1, 5)
    assert max(risks) >= F(2, 15)


def test_observing_current_state_removes_restricted_decision_lower_bound():
    assert module.fixed_state_policy_value("E", (0, 1)) == 1
    assert module.fixed_state_policy_value("T", (0, 1)) == 1


def test_probability_clipping_does_not_preserve_unbiasedness():
    result = module.clipping_counterexample()
    assert result["unclipped_expectation"] == 0
    assert result["clipped_expectation"] == F(1, 4)


def test_appendix_exact_receipt():
    result = module.verify()
    assert result["status"] == "CORRECTED_THEORY_EXACT_CHECKS_PASS"
    assert result["balanced_delayed_tv"] == F(3, 10)
    assert result["state_dependent_delayed_tv"] == F(9, 50)
