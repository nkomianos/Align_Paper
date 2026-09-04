"""Existing alignment/teacher-fitting baselines for an updated sender's cache.

Cache layout is B,H,N,D, as actually passed by the pinned C2C wrapper. These
are engineering baselines, not implementations of a claimed new method.
"""
import copy

import torch
from torch import nn

METHODS = ("identity", "diagonal", "ridge", "orthogonal")


def sample_positions(length, maximum=32):
    if length < 1 or maximum < 1:
        raise ValueError("empty cache or invalid sample count")
    # Integer arithmetic makes the predetermined sampling independent of device.
    count = min(length, maximum)
    return [0] if count == 1 else [i * (length-1)//(count-1) for i in range(count)]


def head_samples(cache):
    if cache.ndim != 4 or not torch.isfinite(cache).all():
        raise ValueError("expected finite B,H,N,D cache")
    b, h, n, d = cache.shape
    return cache.permute(1, 0, 2, 3).reshape(h, b*n, d).double()


def fit_head_map(updated, original, method, ridge=1e-3):
    """Fit y = x W + b independently per head; fit tensors contain no labels.

Ridge and diagonal fits regularize toward identity with lambda proportional to
each head's mean centered feature variance. Orthogonal fit permits reflections.
Keys are in post-RoPE cache space; this is not a RoPE-aware method reproduction.
"""
    if method not in METHODS or ridge <= 0 or updated.shape != original.shape:
        raise ValueError("invalid map request")
    x, y = head_samples(updated), head_samples(original)
    h, n, d = x.shape
    eye = torch.eye(d, device=x.device, dtype=x.dtype).expand(h, d, d)
    if method == "identity":
        return {"weight": eye.clone().float(), "bias": torch.zeros(h, d, device=x.device)}
    if n < 2:
        raise ValueError("need at least two calibration samples")
    mx, my = x.mean(1), y.mean(1)
    x, y = x-mx[:, None, :], y-my[:, None, :]
    xx = x.transpose(1, 2) @ x / n
    xy = x.transpose(1, 2) @ y / n
    lam = ridge * xx.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-8)
    if method == "ridge":
        weight = torch.linalg.solve(xx + lam[:, None, None]*eye, xy + lam[:, None, None]*eye)
    elif method == "diagonal":
        slopes = (xy.diagonal(dim1=-2, dim2=-1) + lam[:, None]) / (xx.diagonal(dim1=-2, dim2=-1) + lam[:, None])
        weight = torch.diag_embed(slopes)
    else:
        u, _, vh = torch.linalg.svd(xy)
        weight = u @ vh
    bias = my - torch.einsum("hd,hde->he", mx, weight)
    if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
        raise ValueError("nonfinite fitted repair")
    return {"weight": weight.float(), "bias": bias.float()}


def apply_head_map(cache, weight, bias):
    if cache.ndim != 4 or weight.shape != (cache.shape[1], cache.shape[3], cache.shape[3]) or bias.shape != (cache.shape[1], cache.shape[3]):
        raise ValueError("repair/cache shape mismatch")
    result = torch.einsum("bhnd,hde->bhne", cache.float(), weight.float()) + bias.float()[None, :, None, :]
    return result.to(cache.dtype)


def verify_head_map(updated, original, actual, method, ridge=1e-3):
    """Replay fits; handle nonunique orthogonal optima without false corruption."""
    expected = fit_head_map(updated, original, method, ridge=ridge)
    if set(actual) != set(expected):
        raise ValueError("map parameter coverage differs")
    if method != "orthogonal":
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=1e-4, atol=1e-5, equal_nan=False)
        return
    x, y = head_samples(updated), head_samples(original)
    h, _, d = x.shape
    w, b = actual["weight"].double(), actual["bias"].double()
    if w.shape != (h,d,d) or b.shape != (h,d) or not torch.isfinite(w).all() or not torch.isfinite(b).all():
        raise ValueError("invalid orthogonal map")
    torch.testing.assert_close(w.transpose(1,2) @ w, torch.eye(d, dtype=w.dtype, device=w.device).expand(h,d,d), rtol=1e-4, atol=1e-5)
    bias = y.mean(1)-torch.einsum("hd,hde->he", x.mean(1), w)
    torch.testing.assert_close(b, bias, rtol=1e-4, atol=1e-5)
    achieved = (x @ w+b[:,None,:]-y).square().mean((1,2))
    optimum = (x @ expected["weight"].double()+expected["bias"].double()[:,None,:]-y).square().mean((1,2))
    if torch.any(achieved > optimum+1e-5*y.square().mean((1,2)).clamp_min(1)):
        raise ValueError("orthogonal map does not attain the calibration optimum")


class CacheRepairProjector(nn.Module):
    """Apply a map before the *unchanged* released projector, never after it."""
    def __init__(self, projector, key_map, value_map):
        super().__init__()
        self.projector = projector
        for prefix, mapping in (("key", key_map), ("value", value_map)):
            self.register_buffer(prefix+"_weight", mapping["weight"].clone())
            self.register_buffer(prefix+"_bias", mapping["bias"].clone())

    def forward(self, source_kv, target_kv, **kwargs):
        corrected = (apply_head_map(source_kv[0], self.key_weight, self.key_bias),
                     apply_head_map(source_kv[1], self.value_weight, self.value_bias))
        return self.projector(corrected, target_kv, **kwargs)


def retune_projector(projector, updated_kv, original_kv, receiver_kv,
                     seed, steps=100, batch_tokens=64, lr=1e-4):
    """Ordinary output-matching baseline; deterministic eval gates and no labels.

Retune a float32 copy to match a frozen float32 teacher's cache outputs. Hard
gate logits stay fixed: eval-mode thresholding is not differentiable. Returned
student remains float32; the runner must save both this and the deployment cast.
"""
    teacher = copy.deepcopy(projector).float().eval().requires_grad_(False)
    student = copy.deepcopy(projector).float().eval()
    student.requires_grad_(True)
    for name, parameter in student.named_parameters():
        if "gate_logit" in name:
            parameter.requires_grad_(False)
    if steps < 1 or batch_tokens < 1 or lr <= 0:
        raise ValueError("invalid retuning budget")
    tensors = [*updated_kv, *original_kv, *receiver_kv]
    if any(t.ndim != 4 or t.shape[0] != 1 or not torch.isfinite(t).all() for t in tensors):
        raise ValueError("retune expects finite single-batch cache tensors")
    count = updated_kv[0].shape[2]
    if count < 2 or any(t.shape[2] != count for t in tensors):
        raise ValueError("unmatched token samples")
    device = next(student.parameters()).device
    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=lr, weight_decay=0)
    generator = torch.Generator().manual_seed(seed)
    curve = []
    for step in range(steps):
        indices = torch.randint(count, (batch_tokens,), generator=generator).to(device)
        def take(pair):
            return tuple(t.to(device=device, dtype=torch.float32).index_select(2, indices) for t in pair)
        old, new, target = take(original_kv), take(updated_kv), take(receiver_kv)
        with torch.no_grad():
            desired = teacher(old, target)
        actual = student(new, target)
        loss = sum((a-b).square().mean() / (b-t).square().mean().clamp_min(1e-6)
                   for a, b, t in zip(actual, desired, target)) / 2
        if not torch.isfinite(loss):
            raise ValueError("nonfinite retuning loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
        optimizer.step()
        curve.append({"step": step+1, "normalized_output_mse": float(loss.detach()),
                      "gradient_norm": float(norm), "sample_indices": indices.cpu().tolist()})
    return student, curve
