from check_monitor_recovery_candidates import differences


def test_retained_command_change_is_not_a_missing_field():
    partial=[{'src':'agent','tools':[{'fn':'bash_command','cmd':'cat input.csv'}]}]
    full=[{'src':'agent','tools':[{'fn':'bash_command','cmd':'cat expected.csv'}]}]
    result=differences(partial,full,'steps')
    assert len(result)==1 and result[0]['kind']=='unexplained_difference'


def test_missing_argument_is_flagged_but_not_equated_to_observed_text():
    partial=[{'src':'user','msg':'$32'},{'src':'agent','tools':[{'fn':'think','cmd':''}]}]
    full=[{'src':'user','msg':'Transform the input'},{'src':'agent','tools':[{'fn':'think','cmd':'reasoning'}]}]
    result=differences(partial,full,'steps')
    assert [r['kind'] for r in result]==['serialized_reference','empty_tool_argument']


def test_changed_step_count_is_unexplained():
    assert differences([{'src':'user'}],[], 'steps')[0]['kind']=='unexplained_difference'
