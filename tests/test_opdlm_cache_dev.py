import json
import pytest
import torch
import transformers
from interaction_sprint.opdlm_cache_dev import Instrument,upstream

def test_loading_metadata_serialization():
    from interaction_sprint.opdlm_cache_dev import json_default
    assert json.loads(json.dumps({'missing_keys':set()},default=json_default))=={'missing_keys':[]}
    with pytest.raises(TypeError): json.dumps({'bad':object()},default=json_default)

def test_released_mask_and_attention_equivalence():
    torch.set_num_threads(2); torch.manual_seed(193)
    env=upstream()
    cfg=env['A2DQwen3Config'](vocab_size=32,hidden_size=32,intermediate_size=64,
        num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2,head_dim=8)
    cfg._attn_implementation='sdpa'
    model=transformers.AutoModelForMaskedLM.from_config(cfg).eval()
    x=torch.tensor([[1,2,3,4,5,6,7,8]])
    mask,pos=env['_prepare_for_sampling'](x,4,0)
    assert mask[0,0,0,3] and not mask[0,0,0,4]
    assert mask[0,0,4,7] and pos.tolist()==[list(range(8))]
    seed=5
    with torch.inference_mode():
        def f(t): return model(t,attention_mask=mask,position_ids=pos,use_cache=False).logits
        native=f(x)
        with Instrument(model) as inst:
            inst.seed=seed
            plain=f(x)
            torch.testing.assert_close(native,plain,atol=1e-6,rtol=1e-5)
            variants=[]
            for token in [9,10]:
                candidate=x.clone(); candidate[0,seed]=token
                inst.mode='collect'; inst.saved={}; f(candidate); inst.cache=inst.saved
                inst.mode='late_corrected'; late=f(x)
                torch.testing.assert_close(late[:,seed],native[:,seed],atol=1e-6,rtol=1e-5)
                variants.append(late)
            assert not torch.allclose(variants[0][:,6],variants[1][:,6])
        torch.testing.assert_close(native,f(x))
