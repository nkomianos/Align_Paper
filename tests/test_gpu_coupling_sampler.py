import numpy as np
import pytest
import torch
from interaction_sprint.byte_coupling_native import NativeSampler
from interaction_sprint.gpu_coupling_sampler import GPUNativeSampler


@pytest.mark.parametrize('device', ['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
def test_all_policies_match_reference(device):
    labels=[b'raw:a',b'raw:ab',b'raw:b',b'special:EOS',b'opaque:m:4']
    groups=[b'byte:a',b'byte:a',b'byte:b',b'special:EOS',b'opaque:m:4']
    native=NativeSampler(labels,groups,[1,2,1,5,0])
    gpu=GPUNativeSampler(labels,groups,[1,2,1,5,0],device)
    rng=np.random.default_rng(500)
    for policy in ['independent','token_clock','byte_clock','byte_hierarchical']:
        for seed in range(200):
            z=rng.normal(size=5)*3
            clock=seed<<16
            expected=native.draw(z,seed,clock,policy)
            token,logp=gpu.draw_gpu(torch.tensor(z,device=device),seed,clock,policy)
            assert token==expected
            assert logp==pytest.approx(z[token]-np.logaddexp.reduce(z),abs=1e-12)


def test_hierarchical_native_frequencies():
    sampler=GPUNativeSampler([b'a',b'ab',b'b'],[b'a',b'a',b'b'],[1,2,1],'cpu')
    probability=np.array([.15,.55,.3]);z=torch.tensor(np.log(probability))
    counts=np.zeros(3)
    for seed in range(6000):
        token,_=sampler.draw_gpu(z,seed,0,'byte_hierarchical');counts[token]+=1
    assert np.max(np.abs(counts/6000-probability))<.025


def test_nonfinite_rejected():
    sampler=GPUNativeSampler([b'a'],[b'a'],[1],'cpu')
    with pytest.raises(ValueError):sampler.draw_gpu(torch.tensor([float('nan')]),1,0,'token_clock')
