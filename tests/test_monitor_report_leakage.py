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
