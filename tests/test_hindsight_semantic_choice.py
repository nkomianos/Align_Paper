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


def test_joint_gate_requires_control_without_erasing_primary_success():
    from interaction_sprint.hindsight_semantic_choice import summarize
    rows=[];seen=set()
    for c in dataset():
        if c['base_id'] not in seen:
            rows.append(dict(c,kind='base',probabilities=[.5,.5],prediction=0));seen.add(c['base_id'])
        for kind in ('direct','hindsight'):
            p=[.01,.01];p[c['target']]=.99
            rows.append(dict(c,kind=kind,probabilities=p,prediction=c['target']))
    assert summarize(rows)['status']=='CONSTRAINED_TEACHER_QUALIFIED'
    controls=[r for r in rows if r['kind']=='direct' and r['split']=='development']
    for r in controls[:6]:r['prediction']=1-r['target']
    result=summarize(rows)
    assert result['status']=='CONSTRAINED_TEACHER_UNQUALIFIED'
    assert result['metrics']['hindsight']['confirmation']['correct']==32
