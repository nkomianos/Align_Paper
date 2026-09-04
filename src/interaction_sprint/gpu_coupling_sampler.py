"""GPU-logit implementation of the frozen native-marginal sampler.

Counter noise is generated with the exact existing NumPy stream and transferred;
model logits never leave the accelerator. This is an auditable prototype, not an
optimized claim about GPU throughput. Shared bytes and fresh clocks retain the
same coupling construction; grouping does not restrict native token support.
"""
import numpy as np
import torch
from .byte_coupling_native import NativeSampler


class GPUNativeSampler(NativeSampler):
    def __init__(self, labels, groups, lengths, device):
        super().__init__(labels, groups, lengths)
        self.device = torch.device(device)
        self.group_index = torch.as_tensor(self.group_ids, device=self.device)

    def draw_gpu(self, logits, seed, clock, policy):
        z = logits.to(device=self.device, dtype=torch.float64)
        assert tuple(z.shape) == self.keys.shape
        if not bool(torch.isfinite(z).all()):
            raise ValueError('Nonfinite model logits')
        assert policy in ('independent', 'token_clock', 'byte_clock', 'byte_hierarchical')
        noise = torch.as_tensor(self.gumbel(self.keys, seed, clock), device=self.device)
        if policy == 'byte_hierarchical':
            probability = torch.softmax(z, dim=0)
            mass = torch.zeros(len(self.group_keys), dtype=torch.float64, device=self.device)
            mass.scatter_add_(0, self.group_index, probability)
            group_noise = torch.as_tensor(self.gumbel(self.group_keys, seed, clock), device=self.device)
            group = (mass.log() + group_noise).argmax()
            noise = noise.masked_fill(self.group_index != group, -torch.inf)
        token = int((z + noise).argmax())
        return token, float(z[token] - torch.logsumexp(z, dim=0))
