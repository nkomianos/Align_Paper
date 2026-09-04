from pathlib import Path
import json
import pytest
import torch
from scripts.sdpo_local_gradient_audit_v2 import local_statistics,released_full_loss


def cls():
    return released_full_loss(Path('artifacts/user_interactions_objective_audit_v1/online_sdpo_updater.py').read_bytes().replace(b'\r\n',b'\n'))


def test_nonzero_near_saturated_cosine_retained_but_flagged():
    z=torch.tensor([32.,0.,-1.,-2.],dtype=torch.float64)
    q=torch.tensor([31.,0.,-.5,-2.],dtype=torch.float64)
    result=local_statistics(z,q,0,cls(),topk=2)
    p=z.softmax(0); a=q.log_softmax(0)-z.log_softmax(0)
    center=-p.clone(); center[0]+=1
    observed=-a[0]*center; expected=p*((p*a).sum()-a)
    expected_cosine=float((observed@expected)/(observed.norm()*expected.norm()))
    assert result['observed_expected_gradient_cosine']==pytest.approx(expected_cosine)
    assert result['sampled_cosine_numerically_small']
    assert result['topk_cosine_numerically_small']


def test_exact_zero_stays_undefined():
    result=local_statistics([1.,2.,3.,4.],[1.,2.,3.,4.],0,cls(),topk=2)
    assert result['observed_expected_gradient_cosine'] is None
    assert result['sampled_gradient_trace_variance']==0
    assert result['sampled_cosine_numerically_small']


def test_observed_call_two_regression_when_local_evidence_available():
    root=Path('retrieved/sdpo_frozen_forward_20260904T1007Z/sdpo_frozen_forward_20260904T1005Z')
    if not root.exists(): pytest.skip('optional immutable local forensic regression')
    torch.set_num_threads(4)
    base=torch.load(root/'logits/001.pt',weights_only=True)
    teacher=torch.load(root/'logits/002.pt',weights_only=True)
    case=json.loads((root/'cases.json').read_text())[0]
    token=case['completion_ids'][case['positions'][1]]
    result=local_statistics(base[1],teacher[1],token,cls())
    assert result['observed_expected_gradient_cosine']==pytest.approx(.5453208324282273,abs=2e-5)
    assert result['sampled_cosine_numerically_small']
