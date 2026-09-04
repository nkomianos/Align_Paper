import numpy as np
from scipy.integrate import quad

from interaction_sprint.flow_ensemble_pilot import basis, field_parts, solve, moments, gaussian_kl, fit, vfd, energy


def test_partition_and_affine_parts():
    assert np.allclose(basis(np.linspace(0, 1, 20)).sum(1), 1)
    coeff = np.tile(np.array([[.1, .2], [.3, .4], [.5, .6]])[None], (6, 1, 1))
    a, b = field_parts(coeff, .23)
    assert np.allclose(a, [[.1, .3], [.2, .4]]) and np.allclose(b, [.5, .6])


def test_translation_and_null():
    zero = np.zeros((6, 3, 2)); move = zero.copy(); move[:, 2, 0] = .25
    s0, s1 = solve(zero), solve(move)
    m0, c0 = moments(s0, np.asarray(1.)); m1, c1 = moments(s1, np.asarray(1.))
    assert np.allclose(m1, [.25, 0]) and np.allclose(c1, np.eye(2))
    assert abs(gaussian_kl(m0, c0, m1, c1)-.03125) < 1e-10
    assert vfd([zero, zero], [s0, s0], 100) == 0
    # Constant translation is not an exact OT Gaussian-path velocity; do not assert VFD=KL here.
    assert vfd([zero, move], [s0, s1], 100) > 0


def test_fitting_deterministic_and_finite():
    data = np.random.default_rng(10).normal(size=(128, 2))
    a = fit(data, 9)
    assert np.array_equal(a, fit(data, 9)) and np.isfinite(a).all()
    x = data[:64]; y = data[64:]
    assert abs(energy(x, y)-energy(y, x)) < 1e-12
