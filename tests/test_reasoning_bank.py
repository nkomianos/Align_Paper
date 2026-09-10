from run_reasoning_bank import answer


def test_numeric_answer_requires_explicit_terminal_marker():
    assert answer('intermediate 123') is None
    assert answer('there are 10. #### 1,234.50')=='1234.5'
    assert answer('#### -2')=='-2'
    assert answer('#### 2\nCorrection: #### 3')=='3'
