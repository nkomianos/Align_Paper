import torch
from interaction_sprint.hindsight_full_feedback import report_weights


def test_full_vocabulary_invalid_action_mass_is_not_renormalized():
    p=torch.tensor([.1,.2,.7],dtype=torch.float64,requires_grad=True)
    m=report_weights(p,torch.tensor([0,1]),1,.9)
    assert not m.requires_grad
    assert torch.allclose(m,torch.tensor([.09,.28,.63],dtype=torch.float64))
    assert torch.allclose(m.sum(),torch.tensor(1.,dtype=torch.float64))
    assert torch.equal(report_weights(p,torch.tensor([0,1]),0,0.),torch.tensor([1.,0.,0.],dtype=torch.float64))


def test_full_kl_equals_explicit_report_average_and_has_finite_gradient():
    z=torch.tensor([.2,-.3,.7,.1],dtype=torch.float64,requires_grad=True)
    lp=z.log_softmax(0);p=lp.exp();ab=torch.tensor([0,1])
    q=torch.tensor([[.6,.1,.2,.1],[.1,.6,.1,.2],[.45,.45,.05,.05]],dtype=torch.float64)
    m=report_weights(p,ab,1,.9)
    vectorized=(m[:,None]*p[None,:]*(lp[None,:]-q.log())).sum()
    explicit=sum(m[o]*sum(p[a]*(lp[a]-q[o,a].log()) for a in range(4)) for o in range(3))
    assert torch.allclose(vectorized,explicit,atol=1e-12,rtol=1e-12)
    grad=torch.autograd.grad(vectorized,z)[0]
    assert torch.isfinite(grad).all() and abs(float(grad.sum()))<1e-12
