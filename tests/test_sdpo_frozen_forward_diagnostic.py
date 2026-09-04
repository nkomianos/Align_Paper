import pytest
import torch
from interaction_sprint.sdpo_frozen_forward_diagnostic import select_cases, positions, fixed_feedback_moments, forward


def test_selection_fixed_strata_and_order():
    rows = [dict(id=f"c{i:02}", preference=style, score=dict(joint=(style=="plain")))
            for i,style in enumerate(["plain"]*8+["json"]*8+["table"]*8+["bullets"]*8)]
    selected = select_cases(list(reversed(rows)))
    assert [r["id"] for r in selected] == [f"c{i:02}" for i in list(range(8))+[8,9,10,16,17,18,24,25]]
    with pytest.raises(ValueError):
        select_cases(rows[:-1])


def test_punctuation_position_is_token_only():
    class Tokenizer:
        all_special_ids = [9]
        def decode(self, tokens, **kwargs):
            return {0:"Asset",1:":",2:"unit",3:"; zone",9:"<eos>"}[tokens[0]]
    assert positions(Tokenizer(), [0,1,2,9]) == [0,1]
    assert positions(Tokenizer(), [1,2,3,9]) == [0,2]
    with pytest.raises(ValueError):
        positions(Tokenizer(), [0,2,9])


def test_local_moments_match_explicit_enumeration():
    torch.manual_seed(2)
    z,q=torch.randn(25,dtype=torch.float64),torch.randn(25,dtype=torch.float64)
    p=z.softmax(-1); a=q.log_softmax(-1)-z.log_softmax(-1)
    g=a[:,None]*(torch.eye(25,dtype=torch.float64)-p[None,:])
    mean=(p[:,None]*g).sum(0)
    variance=(p[:,None]*(g-mean).square()).sum()
    actual=fixed_feedback_moments(z,q,3)
    assert actual["fixed_feedback_expected_ascent_norm"] == pytest.approx(float(mean.norm()),abs=1e-12)
    assert actual["fixed_feedback_variance_trace"] == pytest.approx(float(variance),abs=1e-12)
    assert actual["sampled_ascent_norm"] == pytest.approx(float(g[3].norm()),abs=1e-12)
    same=fixed_feedback_moments(z,z,3)
    assert same["fixed_feedback_expected_ascent_norm"] < 1e-14
    assert same["fixed_feedback_variance_trace"] < 1e-14


def test_forward_native_sequence_offset_and_no_grads():
    from transformers import Qwen3Config,Qwen3ForCausalLM
    torch.manual_seed(3)
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=24,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8)).eval()
    context,y=[1,7,8],[9,6,2]
    chosen,lp=forward(model,context,y,[0,1],"cpu")
    with torch.no_grad():
        expected=model(torch.tensor([context+y]),use_cache=False).logits[0,len(context)-1:-1].float()
    assert torch.allclose(chosen,expected[[0,1]],atol=1e-6)
    wanted=expected.log_softmax(-1).gather(-1,torch.tensor(y)[:,None])[:,0]
    assert torch.allclose(torch.tensor(lp),wanted,atol=1e-6)
    assert not chosen.requires_grad
    assert all(p.grad is None for p in model.parameters())
