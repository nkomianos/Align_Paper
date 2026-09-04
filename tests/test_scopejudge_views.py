from interaction_sprint.scopejudge_views import preexecution_view


def test_no_current_future_or_labels():
    record=dict(extra={'golden_label':'SECRET_LABEL'},steps=[
        dict(step_id=1,source='system',message='PLATFORM'),
        dict(step_id=2,source='user',message='TASK'),
        dict(step_id=3,source='agent',message='PLAN',tool_calls=[
            dict(tool_call_id='a',function_name='read',arguments={'path':'x'},extra={'label':'LEAK'}),
            dict(tool_call_id='b',function_name='read',arguments={'path':'SIBLING'})],
            observation={'results':['CURRENT_RESULT']}),
        dict(step_id=4,source='agent',message='FUTURE')])
    view=preexecution_view(record,3,'a')
    text=str(view)
    for prohibited in ['PLATFORM','PLAN','LEAK','CURRENT_RESULT','FUTURE','SIBLING','SECRET_LABEL']:
        assert prohibited not in text
    assert 'TASK' in text
    assert record['steps'][2]['observation']['results']==['CURRENT_RESULT']
