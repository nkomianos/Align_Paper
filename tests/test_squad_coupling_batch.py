import numpy as np
import pytest
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM,LlamaConfig,LlamaForCausalLM
from interaction_sprint.byte_coupling_native import NativeSampler
from interaction_sprint.squad_coupling_runner import qualify_cache,batch_decode


class Tokens:
    eos_token_id=0
    def decode(self,ids,skip_special_tokens=False,**kwargs):
        return ''.join(chr(64+i) for i in ids if not(skip_special_tokens and i==0))


@pytest.mark.parametrize('family',['qwen','llama'])
def test_cached_batch_preserves_individual_rollouts(family):
    torch.set_num_threads(2);torch.manual_seed(914)
    config=dict(vocab_size=20,hidden_size=32,intermediate_size=48,num_hidden_layers=2,
                num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64,
                attention_dropout=0.,eos_token_id=0)
    model=Qwen3ForCausalLM(Qwen3Config(**config)) if family=='qwen' else LlamaForCausalLM(LlamaConfig(**config))
    model.eval();model.set_attn_implementation('eager')
    check=qualify_cache(model,[1,2,3])
    assert check['first_batch_serial_error']<1e-5 and check['next_cache_full_error']<1e-5
    assert check['next_argmax_agreement']
    raw=[chr(64+i).encode() for i in range(20)]
    sampler=NativeSampler(raw,raw,[1]*20); cfg={'max_new_tokens':5,'temperature':1.}
    for policy in ['independent','token_clock','byte_clock','byte_hierarchical']:
        batch,_=batch_decode(model,Tokens(),sampler,raw,[1,2,3],list(range(8)),policy,'x','m',cfg)
        for seed,row in enumerate(batch):
            solo,_=batch_decode(model,Tokens(),sampler,raw,[1,2,3],[seed],policy,'x','m',cfg)
            assert row['tokens']==solo[0]['tokens']
            assert np.allclose([e['log_probability'] for e in row['events']],
                               [e['log_probability'] for e in solo[0]['events']],atol=1e-5)
