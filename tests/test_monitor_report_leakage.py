from prepare_cheatbench_event_view import project


def test_retrospective_report_is_rejected_as_a_whole():
    view,report=project({'events':[{'kind':'report_prelude','role':'system','content':'Status: Success'},
                                  {'role':'assistant','content':'Transcript plus retrospective statistics'}]})
    assert view is None
    assert report['status']=='RETROSPECTIVE_REPORT_SOURCE_REJECTED'


def test_genuine_observed_success_text_is_preserved():
    original=[{'role':'tool','content':'Unit test succeeded'}]
    view,report=project({'messages':original})
    assert view==original
    assert report['status']=='EVENT_VIEW_PREPARED'


def test_unresolved_raw_user_message_reference_is_rejected():
    view,report=project({'schema_version':'cheatbench.raw_trace.v1','events':[
        {'step':0,'src':'system','msg':'$31'}, {'step':1,'src':'user','msg':'$32'},
        {'step':2,'src':'agent','msg':'I will inspect the input','tools':[{'fn':'bash_command','cmd':'ls'}]}]})
    assert view is None
    assert report['status']=='UNRESOLVED_USER_MESSAGE_REFERENCE'


def test_currency_in_a_real_instruction_is_not_a_reference():
    events=[{'step':0,'src':'user','msg':'Calculate the tax on $32'}]
    view,report=project({'schema_version':'cheatbench.raw_trace.v1','events':events})
    assert view==events
    assert report['status']=='EVENT_VIEW_PREPARED'
