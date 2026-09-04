import numpy as np
import pytest

from interaction_sprint.cache_verification_audit import (
    attention, corrected_attention, experiment, forward, layers_for, run,
    scalar_witness, verify,
)


def test_posthoc_correction_matches_direct_attention():
    rng = np.random.default_rng(17)
    q, k, v, old_k, old_v = [rng.normal(size=(7, 4)) for _ in range(5)]
    for seeds in ([], [0], [0, 3], list(range(7))):
        got = corrected_attention(q, k, v, old_k, old_v, seeds)
        direct = corrected_attention(q, k, v, old_k, old_v, seeds, explicit=True)
        np.testing.assert_allclose(got, direct, atol=1e-12)
        kp, vp = k.copy(), v.copy()
        kp[seeds], vp[seeds] = old_k[seeds], old_v[seeds]
        nonseeds = [i for i in range(7) if i not in seeds]
        np.testing.assert_allclose(got[nonseeds], attention(q, kp, vp)[0][nonseeds])


@pytest.mark.parametrize("n", [2, 3, 8])
@pytest.mark.parametrize("a", [-3., 0., 4.])
def test_two_layer_witness(n, a):
    row = scalar_witness(a, n)
    assert row["verified_seed"] == pytest.approx(a*(n-1)/n**2)
    assert row["clean_seed"] == 0


def test_zero_override_control():
    rng = np.random.default_rng(21)
    x = rng.normal(size=(5, 8))
    layers = layers_for(1, 4, 8)
    clean, cache = forward(x, layers)
    restored, _ = forward(x, layers, [0, 3], cache)
    np.testing.assert_allclose(clean, restored, atol=1e-12)


def test_depth_negative_control_and_positive_sweep():
    result = experiment()
    for row in result["summary"]:
        assert row["sensitive_over_1e_10"] == (0 if row["depth"] == 1 else 8)


def test_nonoverwrite_and_tamper(tmp_path):
    root = tmp_path / "run"
    run(root)
    assert verify(root)["verified"]
    with pytest.raises(FileExistsError):
        run(root)
    with (root / "result.json").open("a", encoding="utf-8") as f:
        f.write(" ")
    with pytest.raises(AssertionError):
        verify(root)
