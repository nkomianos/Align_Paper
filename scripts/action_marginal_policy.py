"""Exact finite-grammar policy primitives; not an unrestricted LM policy.

Offsets are frozen at initialization and shared by all estimators on a grammar.
The detached conditional weights in the marginal estimator implement a score
gradient, not differentiation through a sampling distribution.
"""
import torch


def _validate(scores, action_ids):
    if scores.ndim != 1 or action_ids.shape != scores.shape:
        raise ValueError("expected one score and action ID per spelling")
    if action_ids.dtype != torch.long or not torch.isfinite(scores).all():
        raise ValueError("invalid scores or action IDs")
    count = int(action_ids.max()) + 1
    if count < 2 or set(action_ids.tolist()) != set(range(count)):
        raise ValueError("action IDs must be contiguous, with at least two actions")
    return count


def action_logsumexp(scores, action_ids):
    count = _validate(scores, action_ids)
    return torch.stack([scores[action_ids == a].logsumexp(0) for a in range(count)])


def initial_offsets(initial_scores, action_ids, target_mass=None):
    masses = action_logsumexp(initial_scores.detach(), action_ids)
    if target_mass is None:
        target_mass = torch.ones_like(masses) / len(masses)
    if target_mass.shape != masses.shape or not (target_mass > 0).all():
        raise ValueError("target action masses must be positive")
    if not torch.isclose(target_mass.sum(), target_mass.new_tensor(1.0)):
        raise ValueError("target action masses must sum to one")
    return (target_mass.log() - masses).detach()


def policy(scores, action_ids, offsets):
    count = _validate(scores, action_ids)
    if offsets.shape != (count,) or offsets.requires_grad:
        raise ValueError("offsets must be frozen per-action constants")
    adjusted = scores + offsets[action_ids]
    strings = adjusted.log_softmax(0)
    return strings, action_logsumexp(strings, action_ids)


def pg_loss(string_logp, action_logp, action_ids, sampled_spelling, reward,
            estimator, baseline=0.5):
    if estimator == "spelling":
        score = string_logp[sampled_spelling]
    elif estimator == "marginal":
        score = action_logp[action_ids[sampled_spelling]]
    else:
        raise ValueError("unknown estimator")
    return -(float(reward) - baseline) * score


def exact_gradient_moments(scores, action_ids, offsets, action_rewards, parameters):
    """Enumerate the actual parameter-space covariance trace for one context.

    This expensive diagnostic is separate from training cost. Each score gradient
    includes the finite-grammar normalizer. No logits-only proxy is substituted.
    """
    strings, actions = policy(scores, action_ids, offsets)
    if action_rewards.shape != actions.shape:
        raise ValueError("reward shape differs from action set")
    probability = strings.detach().exp().double()
    moments = {}
    for estimator in ("spelling", "marginal"):
        gradients = []
        for index in range(len(scores)):
            loss = pg_loss(strings, actions, action_ids, index,
                           action_rewards[action_ids[index]], estimator)
            parts = torch.autograd.grad(loss, parameters, retain_graph=True,
                                        allow_unused=False)
            gradients.append(torch.cat([p.reshape(-1) for p in parts]).double())
        matrix = torch.stack(gradients)
        mean = (probability[:, None] * matrix).sum(0)
        trace = (probability * (matrix - mean).square().sum(1)).sum()
        moments[estimator] = {"mean": mean, "variance_trace": trace}
    return moments
