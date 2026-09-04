from scripts.audit_scopejudge_expert_boundaries import measure


def test_unanimous_boundary_agreement_and_batches():
    labels=[]
    for step in (1,1,2):
        labels.append(dict(step_id=step,**{f'reviewer_{j+1}':True for j in range(5)}))
    record=dict(steps=[dict(step_id=1),dict(step_id=2)],extra={'scopejudge':{'labels':labels}})
    result=measure([record],bootstrap=10)
    assert result['first']==dict(positive=10,total=10,agreement=1.0)
    assert result['later']==dict(positive=5,total=5,agreement=1.0)
    assert result['trajectory_bootstrap_95_interval']==[0.0,0.0]
