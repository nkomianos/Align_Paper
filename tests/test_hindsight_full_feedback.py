import torch
from interaction_sprint.hindsight_full_feedback import report_weights


def test_full_vocabulary_invalid_action_mass_is_not_renormalized():
    p=torch.tensor([.1,.2,.7],dtype=torch.float64,requires_grad=True)
    m=report_weights(p,torch.tensor([0,1]),1,.9)
    assert not m.requires_grad
    assert torch.allclose(m,torch.tensor([.09,.28,.63],dtype=torch.float64))
    assert torch.allclose(m.sum(),torch.tensor(1.,dtype=torch.float64))
    assert torch.equal(report_weights(p,torch.tensor([0,1]),0,0.),torch.tensor([1.,0.,0.],dtype=torch.float64))
