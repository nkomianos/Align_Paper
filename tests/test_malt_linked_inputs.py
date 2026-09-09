from audit_malt_linked_inputs import inspect


def message(text,node=None,parent=None):
    return {'role':'assistant','content':text,'name':None,'function_call':None,
            'metadata':{'node_id':node,'parent_node_id':parent,'branch_id':0,'timestamp':1}}


def test_candidates_not_silently_treated_as_executed():
    row={'metadata':{'run_id':1,'task_id':'f/t'},'samples':[
        {'input':[message('task',0)],'output':[[message('selected')],[message('rejected')]],'metadata':{}},
        {'input':[message('task',0),message('selected',1,0)],'output':[[message('terminal')]],'metadata':{'unmatched':True}}]}
    r=inspect(row)
    assert r['unique_input_nodes']==2 and r['samples_with_multiple_output_candidates']==1
    assert r['outputs_not_observed_as_input']==2
    assert r['final_outputs_not_observed_as_input']==1
    assert not r['complete_executed_transcript_certified']


def test_conflicting_ids_and_missing_parent_are_exposed():
    row={'metadata':{'run_id':1,'task_id':'f/t'},'samples':[
        {'input':[message('old',2,99)],'output':[],'metadata':{}},
        {'input':[message('new',2,99)],'output':[],'metadata':{}}]}
    r=inspect(row)
    assert r['conflicting_node_ids']==1 and r['missing_parent_ids']==1


def test_signature_membership_does_not_imply_later_execution():
    row={'metadata':{'run_id':1,'task_id':'f/t'},'samples':[
        {'input':[message('repeated',0)],'output':[[message('repeated')]],'metadata':{}}]}
    r=inspect(row)
    assert r['outputs_not_observed_as_input']==0
    assert not r['complete_executed_transcript_certified']
