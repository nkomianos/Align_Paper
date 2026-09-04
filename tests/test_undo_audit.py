import json
import numpy as np
from interaction_sprint.undo_audit import build, ARMS
from interaction_sprint.fixtures import reduce_ops
from interaction_sprint.run import prepare, inputs, select
from interaction_sprint.analyze import analyze


def fake(cases, key):
    result=[]
    for c in cases:
        p=np.full(4,.01/3); p[c['choices'].index(key[c['case_id']]['answer'])]=.99
        result.append({'case_id':c['case_id'],'choice_probability':p.tolist(),
            'log_prob':np.log(.9*p).tolist(),'choice_mass':.9,'predicted':key[c['case_id']]['answer']})
    return result


def test_fresh_audit_shape_and_reproducibility():
    cases,key=build()
    assert (cases,key)==build()
    assert len(cases)==864
    assert len(select(cases,'smoke'))==48
    assert len({c['case_id'] for c in cases})==len(cases)
    assert set(c['arm'] for c in cases)==set(ARMS)
    for c in cases:
        k=key[c['case_id']]
        assert reduce_ops({},k['operations'])==k['state']
        if c['arm']!='counterfactual':
            assert k['answer'] != key[c['pair_id']+'/counterfactual']['answer']


def test_no_answer_leak_in_reminder():
    cases,_=build()
    for c in cases:
        if c['arm']=='reminder':
            assert c['messages'][-1]['content'].endswith('A field with a cancelled or removed assignment must be UNSET.')


def test_replay_of_prepared_bytes(tmp_path):
    prepare('undo_audit', tmp_path/'new')
    assert len(inputs(tmp_path/'new')[1])==864


def test_successful_baseline_is_not_training_evidence():
    cases,key=build(); records=fake(cases,key)
    assert analyze('undo_audit',cases,key,records,'full')['decision']=='NO_FRESH_HISTORY_SIGNAL_PARK'
    for c,r in zip(cases,records):
        if c['arm']=='history' and c['depth']>=60:
            p=np.full(4,.01/3); p[c['choices'].index(key[c['case_id']]['stale_answer'])]=.99
            r.update(choice_probability=p.tolist(),log_prob=np.log(.9*p).tolist(),predicted=key[c['case_id']]['stale_answer'])
    assert analyze('undo_audit',cases,key,records,'full')['decision']=='SIMPLE_BASELINE_SUFFICIENT_DO_NOT_CLAIM_TRAINING_NEEDED'


def test_bad_control_is_invalid_not_paper_failure():
    cases,key=build(); records=fake(cases,key)
    for c,r in zip(cases,records):
        if c['arm']=='canonical':
            wrong=(c['choices'].index(r['predicted'])+1)%4
            p=np.full(4,.01/3);p[wrong]=.99
            r.update(choice_probability=p.tolist(),log_prob=np.log(.9*p).tolist(),predicted=c['choices'][wrong])
    assert analyze('undo_audit',cases,key,records,'full')['decision']=='INVALID_ROBUSTNESS_ASSAY'
