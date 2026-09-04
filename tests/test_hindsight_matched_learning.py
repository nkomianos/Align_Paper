from collections import Counter
import torch
from interaction_sprint.hindsight_matched_learning import data_and_schedule, feedback_marginal


def test_balanced_labels_semantic_pairs_and_label_budget():
    cases,schedule=data_and_schedule();lookup={c['id']:c for c in cases}
    assert len(cases)==16 and len(schedule)==24
    assert sum(c['target'] for c in cases)==8
    for i in range(0,16,2):
        a,b=cases[i:i+2]
        assert a['options'][a['target']]==b['options'][b['target']]
    assert all(len(set(batch))==4 and sum(lookup[c]['anchor'] for c in batch)==1 for batch in schedule)
    assert set(Counter(c for b in schedule for c in b).values())=={6}
    assert sum(c['anchor'] for c in cases)==4


def test_marginal_does_not_differentiate_sampling_law():
    p=torch.tensor([.3,.7],requires_grad=True)
    assert not feedback_marginal(p,0,.9).requires_grad
    assert torch.allclose(feedback_marginal(p,0,.9),torch.tensor([.37,.63]))
    assert torch.equal(feedback_marginal(p,1,0),torch.tensor([0.,1.]))
