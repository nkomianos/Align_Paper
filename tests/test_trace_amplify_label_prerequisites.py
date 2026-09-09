from audit_trace_amplify_label_prerequisites import assertion_conflicts


def test_detects_only_literal_conflict_on_same_call():
    result=assertion_conflicts('assert f(1) == 2\nassert f(1) == 3\nassert f(2) == 4')
    assert result['conflicting_literal_assertion_inputs']==1
    assert assertion_conflicts('assert f(1) == 2\nassert f(2) == 3')['conflicting_literal_assertion_inputs']==0


def test_unknown_assertion_is_not_claimed_satisfiable():
    assert assertion_conflicts('assert f(1) == expected')['conflicting_literal_assertion_inputs']==0
    assert assertion_conflicts('assert ?')['parseable'] is False


def test_equivalent_literal_values_are_not_false_conflicts():
    assert assertion_conflicts('assert f(1) == True\nassert f(1) == 1.0')['conflicting_literal_assertion_inputs']==0
