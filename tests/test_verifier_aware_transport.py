from __future__ import annotations

import numpy as np

from verifier_aware_transport import Scenario, decode_categorical, lattice_uniforms, run_feasibility_gate


def test_arbitrary_bin_order_preserves_categorical_marginal() -> None:
    probabilities = (0.11, 0.23, 0.31, 0.35)
    order = (2, 0, 3, 1)
    uniforms = (np.arange(200_000) + 0.5) / 200_000
    observed = np.bincount(
        decode_categorical(probabilities, order, uniforms), minlength=len(probabilities)
    ) / len(uniforms)
    np.testing.assert_allclose(observed, probabilities, atol=1e-5, rtol=0.0)


def test_lattice_coordinates_are_spaced_and_in_unit_interval() -> None:
    uniforms = lattice_uniforms(4, 0.91)
    np.testing.assert_allclose(uniforms, (0.91, 0.16, 0.41, 0.66))
    assert np.all((uniforms >= 0.0) & (uniforms < 1.0))


def test_rejects_non_permutation_order() -> None:
    try:
        decode_categorical((0.4, 0.6), (0, 0), (0.2,))
    except ValueError as error:
        assert "permutation" in str(error)
    else:
        raise AssertionError("invalid order was accepted")


def test_frozen_synthetic_gate_rejects_an_insufficient_selectable_gain() -> None:
    report = run_feasibility_gate(trials=4_000)
    assert report.pass_gate is False
    assert report.marginal_max_error <= 0.01
    assert report.minimum_utility_gain < 0.05
    assert report.minimum_success_gain < 0.05


def test_scenario_rejects_non_unit_probability_mass() -> None:
    try:
        Scenario("bad", (0.4, 0.4), (0.0, 1.0), (0.0, 1.0), 2, 0.1)
    except ValueError as error:
        assert "sum to one" in str(error)
    else:
        raise AssertionError("invalid probability mass was accepted")
