import numpy as np
import torch

from interaction_sprint.competence_pilot import dataset, schedule, project_ascent, qualifies


def test_disjoint_balanced_splits_and_labels():
    data = dataset()
    assert len({c['id'] for c in data}) == len(data) == 192
    seen = {}
    for split, count in [('warm', 64), ('qualify', 32), ('adapt', 32), ('eval', 64)]:
        rows = [c for c in data if c['split'] == split]
        assert len(rows) == count
        assert sum(c['answer'] for c in rows) == count//2
        for c in rows:
            key = (c['domain'], tuple(c['pair']))
            assert key not in seen or seen[key] == split
            seen[key] = split
            target = c['pair'][int(c['higher'])]
            assert f' {target} ' in c['options'][c['answer']]
    order = schedule([c for c in data if c['split'] == 'adapt'], 8)
    assert len(order) == 8 and all(len(b) == len({c['id'] for c in b}) == 8 for b in order)
    assert len({c['id'] for b in order[:4] for c in b}) == 32


def test_projection_is_feasible_and_minimal():
    a = torch.tensor([1., 2.]); g = torch.tensor([-3., 0.])
    projected = project_ascent(g, a)
    assert abs(float(torch.dot(projected, a))) < 1e-6
    assert torch.allclose(projected-g, -torch.dot(g, a)/torch.dot(a, a)*a)
    assert torch.equal(project_ascent(-g, a), -g)
    assert torch.equal(project_ascent(g, torch.zeros_like(g)), g)


def test_qualification_checks_capability_and_vocabulary():
    rows = [dict(domain='a', answer=0, probabilities=[.99, .01], AB_mass=.99)]*10
    assert qualifies(rows)
    assert not qualifies([{**r, 'AB_mass': .5} for r in rows])
    assert not qualifies([{**r, 'probabilities': [.1, .9]} for r in rows])
