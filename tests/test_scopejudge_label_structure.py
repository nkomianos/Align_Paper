import json
from scripts.audit_scopejudge_label_structure import audit


def test_first_positive_batch_not_arbitrary_call_order():
    labels=[dict(step_id=s,tool_call_id=c,votes=v,
                 golden_label='out_of_scope' if v>=3 else 'in_scope')
            for s,c,v in [(1,'a',0),(2,'b',3),(2,'c',5),(3,'d',4)]]
    record=dict(extra={'scopejudge':dict(task_family='synthetic',labels=labels)},
                steps=[dict(step_id=s,tool_calls=[dict(tool_call_id=l['tool_call_id'])
                       for l in labels if l['step_id']==s]) for s in (1,2,3)])
    result=audit(json.dumps(record).encode())
    assert result['trajectories_with_majority_violation']==1
    assert result['first_positive_step_call_vote_histogram']=={3:1,5:1}
    assert result['positive_calls_strictly_after_first_positive_step']==1
