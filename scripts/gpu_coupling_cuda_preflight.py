"""CUDA/reference sampler parity without additional test dependencies."""
import json
import numpy as np
import torch
from interaction_sprint.byte_coupling_native import NativeSampler
from interaction_sprint.gpu_coupling_sampler import GPUNativeSampler

assert torch.cuda.is_available()
labels = [b'raw:a', b'raw:ab', b'raw:b', b'special:EOS', b'opaque:m:4']
groups = [b'byte:a', b'byte:a', b'byte:b', b'special:EOS', b'opaque:m:4']
native = NativeSampler(labels, groups, [1, 2, 1, 5, 0])
gpu = GPUNativeSampler(labels, groups, [1, 2, 1, 5, 0], 'cuda')
rng = np.random.default_rng(500)
errors = []
for policy in ('independent', 'token_clock', 'byte_clock', 'byte_hierarchical'):
    for seed in range(200):
        z = rng.normal(size=5) * 3
        expected = native.draw(z, seed, seed << 16, policy)
        token, logp = gpu.draw_gpu(torch.tensor(z, device='cuda'), seed, seed << 16, policy)
        assert token == expected
        error = abs(logp - (z[token] - np.logaddexp.reduce(z)))
        assert error < 1e-12
        errors.append(error)
print(json.dumps({'device': torch.cuda.get_device_name(), 'selections_matched': len(errors),
                  'max_log_probability_error': max(errors), 'torch': torch.__version__}))
