"""CPU arithmetic for fixed-prefix, fixed-feedback categorical logit gradients.

This is not a neural replay or an unbiased-gradient claim for feedback selected
by a sampled completion. Coordinates are independent local vocabulary logits.
"""
import ast
import hashlib
from types import SimpleNamespace
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

UPSTREAM_SHA = "6d9b92726fffa857e02d3a3773a309d8418433c50f33f91252e760b17fec6b8d"


def released_full_loss(source):
    if hashlib.sha256(source).hexdigest() != UPSTREAM_SHA:
        raise ValueError("upstream source digest differs")
    parent = next(n for n in ast.parse(source).body
                  if isinstance(n, ast.ClassDef) and n.name == "OnlineSDPOUpdater")
    names = {"_full_distillation_loss", "_add_tail", "_renorm_topk"}
    methods = [n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in methods} != names:
        raise ValueError("released methods missing")
    tree = ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(
        name="Released", bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[]))
    namespace = dict(torch=torch, F=F, Tuple=Tuple, Dict=Dict)
    exec(compile(tree, "pinned_released_topk", "exec"), namespace)
    return namespace["Released"]


def _cos(a, b):
    denominator = a.norm() * b.norm()
    return float(torch.dot(a, b) / denominator) if denominator > 1e-15 else None


def local_statistics(student_logits, teacher_logits, observed_token, loss_class, topk=20):
    z = torch.as_tensor(student_logits, dtype=torch.float64).clone().requires_grad_(True)
    qz = torch.as_tensor(teacher_logits, dtype=torch.float64)
    if z.ndim != 1 or qz.shape != z.shape or not torch.isfinite(z).all() or not torch.isfinite(qz).all():
        raise ValueError("invalid full-vocabulary logits")
    if not 0 <= observed_token < z.numel() or not 1 <= topk < z.numel():
        raise ValueError("invalid token or topk")
    lp, lq = z.log_softmax(0), qz.log_softmax(0)
    p = lp.exp().detach()
    advantage = (lq - lp).detach()
    expectation = p * ((p * advantage).sum() - advantage)
    second_moment = (p * advantage.square() * (1 - 2*p + p.square().sum())).sum()
    variance = second_moment - expectation.square().sum()
    if variance < -1e-10:
        raise ValueError("negative trace variance")
    kl = (lp.exp() * (lp - lq)).sum()
    full_gradient = torch.autograd.grad(kl, z, retain_graph=True)[0]
    error = (expectation - full_gradient).abs().max()
    if error > 1e-10:
        raise ValueError("categorical expectation and reverse-KL gradient differ")
    observed = advantage[observed_token] * p
    observed = observed.clone()
    observed[observed_token] -= advantage[observed_token]

    # Call the actual released top-k/tail routine with a one-position bridge.
    stub = SimpleNamespace(config=SimpleNamespace(distillation_topk=topk,
                                                  distillation_add_tail=True))
    stub._add_tail = loss_class._add_tail
    stub._renorm_topk = loss_class._renorm_topk
    def forward(name, *args, **kwargs):
        logits = (z if name == "base" else qz).reshape(1, 1, -1)
        selected = logits.log_softmax(-1)[..., observed_token]
        return selected, torch.ones_like(selected, dtype=torch.long), logits
    stub._compute_token_logprobs = forward
    tail_loss, _ = loss_class._full_distillation_loss(stub, "base", "teacher",
                                                    torch.tensor([[observed_token]]))
    tail_gradient = torch.autograd.grad(tail_loss, z)[0]
    selected_indices = lp.detach().topk(topk).indices
    ps = p[selected_indices].sum()
    qs = lq.exp()[selected_indices].sum()
    # The released helper clips the top-mass log at -1e-7 before adding a tail.
    # Report this rather than silently treating it as exact coarsening at mass 1.
    clamp_active = bool(ps.log() > -1e-7 or qs.log() > -1e-7)
    return dict(full_reverse_kl=float(kl.detach()),
                categorical_expectation_gradient_max_error=float(error),
                expected_gradient_norm=float(expectation.norm()),
                sampled_gradient_trace_variance=float(variance.clamp_min(0)),
                sampled_gradient_second_moment=float(second_moment),
                observed_advantage=float(advantage[observed_token]),
                observed_gradient_norm=float(observed.norm()),
                observed_expected_gradient_cosine=_cos(observed, expectation),
                topk=topk, topk_indices=selected_indices.tolist(),
                student_topk_mass=float(ps), teacher_on_student_topk_mass=float(qs),
                released_tail_clamp_active=clamp_active,
                released_topk_tail_reverse_kl=float(tail_loss.detach()),
                topk_full_gradient_cosine=_cos(tail_gradient, expectation),
                topk_full_gradient_l2_difference=float((tail_gradient-expectation).norm()),
                topk_gradient_norm=float(tail_gradient.norm()),
                scope="One fixed prefix and feedback; vocabulary-logit gradient only; per-position, no completion-length scaling.")
