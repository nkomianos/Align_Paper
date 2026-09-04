import pytest
import torch
from torch import nn

from latent_contract.cache_repair import (sample_positions, fit_head_map, apply_head_map,
                                        CacheRepairProjector, retune_projector, verify_head_map)


def test_sampling_is_fixed_unique_and_includes_endpoints():
    for length in (1, 4, 33, 1024):
        positions = sample_positions(length)
        assert len(positions) == min(length, 32)
        assert len(set(positions)) == len(positions)
        assert positions[0] == 0 and positions[-1] == length-1


@pytest.mark.parametrize("method", ["identity", "diagonal", "ridge", "orthogonal"])
def test_no_update_fits_identity_behavior(method):
    torch.manual_seed(10)
    x = torch.randn(2, 3, 100, 8)
    mapping = fit_head_map(x, x, method)
    torch.testing.assert_close(apply_head_map(x, **mapping), x, rtol=1e-5, atol=1e-6)


def test_diagonal_recovers_rescaling_on_new_samples():
    torch.manual_seed(11)
    x = torch.randn(1, 2, 1000, 4)
    scale = torch.tensor([2., .5, 1.5, 3.])
    y = x*scale + 2
    mapping = fit_head_map(x, y, "diagonal")
    unseen = torch.randn(1, 2, 40, 4)
    torch.testing.assert_close(apply_head_map(unseen, **mapping), unseen*scale+2, rtol=.02, atol=.02)


@pytest.mark.parametrize("method", ["ridge", "orthogonal"])
def test_dense_rotation_recovered_on_unseen_samples(method):
    torch.manual_seed(12)
    x = torch.randn(1, 2, 1000, 4)
    q, _ = torch.linalg.qr(torch.randn(2, 4, 4))
    bias = torch.randn(2, 4)
    y = apply_head_map(x, q, bias)
    mapping = fit_head_map(x, y, method)
    unseen = torch.randn(1, 2, 40, 4)
    torch.testing.assert_close(apply_head_map(unseen, **mapping), apply_head_map(unseen, q, bias), rtol=.02, atol=.02)


def test_rank_deficient_constant_cache_is_finite():
    x = torch.ones(1, 2, 10, 4)
    for method in ("diagonal", "ridge", "orthogonal"):
        mapping = fit_head_map(x, x+1, method)
        torch.testing.assert_close(apply_head_map(x, **mapping), x+1)
    with pytest.raises(ValueError):
        fit_head_map(x, x[:, :, :-1], "ridge")


def test_replay_accepts_nonunique_orthogonal_optimum_but_rejects_wrong_fit():
    x = torch.zeros(1, 1, 20, 4)
    x[0, 0, :, 0] = torch.linspace(-1,1,20)
    alternative = {"weight": torch.diag(torch.tensor([1.,-1.,-1.,-1.]))[None], "bias": torch.zeros(1,4)}
    verify_head_map(x, x, alternative, "orthogonal")
    broken = {"weight": torch.diag(torch.tensor([-1.,1.,1.,1.]))[None], "bias": torch.zeros(1,4)}
    with pytest.raises(ValueError):
        verify_head_map(x, x, broken, "orthogonal")
    ridge = fit_head_map(x,x,"ridge")
    ridge["weight"] += .1
    with pytest.raises(AssertionError):
        verify_head_map(x,x,ridge,"ridge")


class TinyProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 4)
        self.key_gate_logit = nn.Parameter(torch.tensor(1.0))

    def forward(self, source, target):
        return tuple(t + (self.key_gate_logit > 0)*self.projection(s) for s, t in zip(source, target))


def test_identity_wrapper_preserves_bf16_source_and_target():
    torch.manual_seed(15)
    source = torch.randn(1, 2, 10, 4).bfloat16()
    target = torch.randn_like(source)
    projector = TinyProjector().bfloat16().eval()
    mapping = fit_head_map(source, source, "identity")
    wrapper = CacheRepairProjector(projector, mapping, mapping)
    expected = projector((source, source), (target, target))
    actual = wrapper((source, source), (target, target))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_retuning_decreases_output_error_without_changing_teacher_or_gates():
    torch.manual_seed(16)
    projector = TinyProjector().eval()
    before = {n: t.clone() for n, t in projector.state_dict().items()}
    old = (torch.randn(1, 2, 160, 4), torch.randn(1, 2, 160, 4))
    new = tuple(x*1.2+.1 for x in old)
    target = tuple(torch.randn_like(x) for x in old)
    student, curve = retune_projector(projector, new, old, target, seed=9, steps=80, lr=.01)
    with torch.no_grad():
        desired = projector(old, target)
        initial = projector(new, target)
        final = student(new, target)
        before_error = sum((a-b).square().mean() for a, b in zip(initial, desired))
        after_error = sum((a-b).square().mean() for a, b in zip(final, desired))
    assert after_error < before_error * .1
    assert len(curve) == 80
    assert student.key_gate_logit == projector.key_gate_logit
    for name, tensor in projector.state_dict().items():
        torch.testing.assert_close(tensor, before[name], atol=0, rtol=0)


def test_actual_c2c_projector_retuning_uses_frozen_eval_gates():
    import transformers
    from pathlib import Path
    import sys
    if transformers.__version__ != "4.52.4":
        pytest.skip("published C2C API compatibility test")
    upstream = Path(__file__).resolve().parents[1] / "artifacts/c2c_upstream_audit_20260904"
    if not upstream.exists():
        pytest.skip("pinned upstream unavailable")
    sys.path.insert(0, str(upstream))
    from rosetta.model.projector import C2CProjector
    projector = C2CProjector(source_dim=4, target_dim=4, source_num_heads=2, target_num_heads=2,
                             hidden_dim=16, intermediate_dim=16, num_layers=3).eval()
    with torch.no_grad():
        projector.key_gate_logit.fill_(-1)
        projector.value_gate_logit.fill_(1)
    old = (torch.randn(1,2,40,4), torch.randn(1,2,40,4))
    new = tuple(x*1.1 for x in old)
    target = tuple(torch.randn_like(x) for x in old)
    student, curve = retune_projector(projector, new, old, target, seed=2, steps=3)
    assert len(curve) == 3 and not student.training
    assert student.key_gate_logit == -1 and student.value_gate_logit == 1
    with torch.no_grad():
        key, value = student(new, target)
    torch.testing.assert_close(key, target[0], atol=0, rtol=0)
    assert torch.isfinite(value).all()
