"""Pure target transforms; no claim of a new distillation objective."""
import torch


def corrected_target(base_logits, teacher_logits, reference_logits, beta=1.0, clip=10.0):
    if beta <= 0 or (clip is not None and clip <= 0):
        raise ValueError('beta and clipping scale must be positive')
    if base_logits.shape != teacher_logits.shape or base_logits.shape != reference_logits.shape:
        raise ValueError('logit shapes must match')
    if base_logits.ndim < 1 or base_logits.shape[-1] < 2:
        raise ValueError('last dimension must be the vocabulary')
    if not all(torch.isfinite(x).all() for x in [base_logits,teacher_logits,reference_logits]):
        raise ValueError('nonfinite input logits')
    # float64 CPU inputs retain precision for the analytic tests.
    dtype = torch.float64 if base_logits.dtype == torch.float64 else torch.float32
    base,teacher,reference = [x.to(dtype) for x in [base_logits,teacher_logits,reference_logits]]
    delta = teacher.log_softmax(-1)-reference.log_softmax(-1)
    delta = delta-delta.mean(-1,keepdim=True)
    if clip is not None:
        delta = clip*torch.tanh(delta/clip)
    return (base.log_softmax(-1)+delta/beta).log_softmax(-1)


def distribution_kl(log_p, log_q):
    return (log_p.exp()*(log_p-log_q)).sum(-1)
