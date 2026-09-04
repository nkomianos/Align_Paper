import math
from collections import Counter
import pytest
from interaction_sprint.hindsight_semantic_choice import dataset,distribution


def test_complete_crossed_fresh_design():
    rows=dataset()
    assert len(rows)==64 and len({r['id'] for r in rows})==64
    assert Counter(r['split'] for r in rows)==dict(development=32,confirmation=32)
    assert len({r['base_id'] for r in rows})==32
    domains=[{r['domain'] for r in rows if r['split']==s} for s in ('development','confirmation')]
    assert not domains[0]&domains[1]
    for r in rows:
        assert r['options'][r['target']]==r['preferred']
        matches=[p for p in rows if p['domain']==r['domain'] and p['wording']==r['wording'] and p['preferred']==r['preferred']]
        assert len(matches)==2 and {p['target'] for p in matches}=={0,1}


def test_distribution_is_joint_sequence_likelihood_not_length_average():
    assert distribution([math.log(.2)+math.log(.3),math.log(.12)])==pytest.approx([1/3,2/3])
    assert distribution([-10001,-10000])==pytest.approx([1/(1+math.e),math.e/(1+math.e)])
    with pytest.raises(ValueError):distribution([0,float('nan')])
