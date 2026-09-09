from prepare_cheatbench_event_view import project


def test_metadata_does_not_enter_message_view():
    trace={'messages':[{'role':'user','content':'task'},{'role':'assistant','content':'answer','tool_calls':[{'id':'x'}]}],
           'extra':{'monitor_verdict':'cheating'},'result':{'reward':1}}
    view,meta=project(trace)
    assert view==trace['messages']
    assert meta['field']=='messages'


def test_unknown_fields_and_empty_records_fail_closed():
    assert project({'events':[]})[0] is None
    assert project({'events':[{'content':'x','unreviewed_payload':'y'}]})[0] is None


def test_only_inspected_cache_metadata_is_omitted():
    assert project({'messages':[{'content':'x','cache_control':{'type':'ephemeral'}}]})[0]==[{'content':'x'}]
    assert project({'messages':[{'content':'x','cache_control':{'unknown':'payload'}}]})[0] is None
