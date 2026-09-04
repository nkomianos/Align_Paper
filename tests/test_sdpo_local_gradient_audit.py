from pathlib import Path
import numpy as np
import pytest
import torch
from scripts.sdpo_local_gradient_audit import local_statistics, released_full_loss


def loss_class():
    source = Path("artifacts/user_interactions_objective_audit_v1/online_sdpo_updater.py").read_bytes().replace(b"\r\n", b"\n")
    return released_full_loss(source)


def test_expected_gradient_and_variance_against_exhaustive_categories():
    z = torch.tensor([.2, -.3, .7, 1.2], dtype=torch.float64)
    q = torch.tensor([.9, .4, -.1, .2], dtype=torch.float64)
    p = z.softmax(0)
    a = q.log_softmax(0) - z.log_softmax(0)
    gradients = -a[:, None] * (torch.eye(4, dtype=z.dtype) - p)
    expected = (p[:, None] * gradients).sum(0)
    variance = (p[:, None] * (gradients-expected).square()).sum()
    result = local_statistics(z, q, 2, loss_class(), topk=2)
    assert result["expected_gradient_norm"] == pytest.approx(float(expected.norm()))
    assert result["sampled_gradient_trace_variance"] == pytest.approx(float(variance))
    assert result["observed_gradient_norm"] == pytest.approx(float(gradients[2].norm()))
    assert result["categorical_expectation_gradient_max_error"] < 1e-12
    assert result["released_topk_tail_reverse_kl"] <= result["full_reverse_kl"] + 1e-12


def test_equal_distributions_have_zero_signal():
    result = local_statistics([1., 2., 3., 4.], [1., 2., 3., 4.], 0, loss_class(), topk=2)
    assert result["expected_gradient_norm"] < 1e-12
    assert result["sampled_gradient_trace_variance"] < 1e-12
    assert result["observed_expected_gradient_cosine"] is None


def test_invalid_values_and_source_rejected():
    with pytest.raises(ValueError):
        local_statistics([0., np.nan, 1.], [0., 1., 2.], 0, loss_class(), topk=2)
    with pytest.raises(ValueError):
        released_full_loss(b"untrusted source")


def test_tail_clamp_is_explicit():
    result = local_statistics([100., 0., -1.], [90., 0., -1.], 0, loss_class(), topk=1)
    assert result["released_tail_clamp_active"]
    assert np.isfinite(result["released_topk_tail_reverse_kl"])
