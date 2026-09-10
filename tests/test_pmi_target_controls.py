import pytest
import torch
from pmi_target_controls import corrected_target, distribution_kl


def test_analytic_null_reference_and_logit_shift_invariance():
    base=torch.tensor([0.8,0.2],dtype=torch.float64).log()
    reference=torch.tensor([0.5,0.5],dtype=torch.float64).log()
    expected=torch.tensor([16/17,1/17],dtype=torch.float64)
    observed=corrected_target(base,base,reference,clip=None)
    torch.testing.assert_close(observed.exp(),expected)
    torch.testing.assert_close(corrected_target(base+500,base-300,reference+30,clip=None),observed)
    assert distribution_kl(base,observed).item() > 0.11


def test_teacher_equals_reference_is_identity_even_with_clipping():
    base=torch.tensor([[1.,2.,-1.],[3.,-2.,1.]])
    other=torch.tensor([[2.,1.,3.],[-1.,2.,4.]])
    torch.testing.assert_close(corrected_target(base,other,other),base.log_softmax(-1))


def test_nonfinite_logits_rejected():
    with pytest.raises(ValueError,match='nonfinite'):
        corrected_target(torch.tensor([float('nan'),0.]),torch.zeros(2),torch.zeros(2))
