import torch
from run_censor_adapter_gradients import sampled_token_logprobs


def test_temperature_score_gradient_matches_manual_softmax_derivative():
    logits=torch.tensor([[.1,.5,-.7],[.8,-.2,.3]],requires_grad=True)
    target=torch.tensor([1,2]);temperature=.8
    lp=sampled_token_logprobs(logits,target,temperature)
    derivative=torch.autograd.grad(lp.sum(),logits)[0]
    expected=(torch.nn.functional.one_hot(target,3)-(logits.detach()/temperature).softmax(-1))/temperature
    assert torch.allclose(derivative,expected,atol=1e-7)


def test_early_score_does_not_include_later_token_logprob():
    logits=torch.randn(5,7,requires_grad=True);target=torch.tensor([1,2,3,4,5])
    lp=sampled_token_logprobs(logits,target,.8)
    early=torch.autograd.grad(lp[:2].sum(),logits)[0]
    assert torch.count_nonzero(early[2:])==0
