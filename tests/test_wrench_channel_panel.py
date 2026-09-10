from prepare_wrench_channel_panel import view
from run_wrench_channel_panel import parse


def test_transform_preserves_actions_and_outputs_and_removes_elicitation():
    trace={'steps':[{'source':'user','message':'secret elicitation'},
                    {'source':'agent','step_id':2,'message':'a rationale',
                     'tool_calls':[{'arguments':{'cmd':'literal command'}}],
                     'observation':{'results':[{'content':'literal output'}]}}]}
    full, stripped=view(trace,False),view(trace,True)
    assert full[0]['tool_calls']==stripped[0]['tool_calls']
    assert full[0]['observation']==stripped[0]['observation']
    assert 'message' not in stripped[0] and len(full)==len(stripped)==1
    assert trace['steps'][1]['message']=='a rationale'


def test_parser_rejects_prefixes_booleans_and_nonfinite_values():
    assert parse('{"score":0.25}')==.25
    for text in ('Score: 0.25','{"score":true}','{"score":NaN}','{"score":2}','{"score":.2}'):
        assert parse(text) is None
