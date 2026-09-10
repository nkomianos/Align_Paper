from prepare_thinking_pmi_traces import select_positions, thinking_positions


def test_selection_ignores_ineligible_positions_and_breaks_ties_early():
    result=select_positions([100.,1.,2.,2.,200.],[1,2,3],'problem-a')
    assert result['high_entropy']==2
    assert result['random'] in [1,2,3]
    assert result==select_positions([100.,1.,2.,2.,200.],[1,2,3],'problem-a')
    assert select_positions([1.],[],'empty') is None


def test_only_explicit_thinking_span_is_eligible():
    ids=[10]+[1]*39+[11]+[1]*10
    assert thinking_positions(ids,10,11)==(list(range(32,40)),0,40)
    assert thinking_positions([1]*50,10,11)==([],None,50)
    assert thinking_positions([11,10]+[1]*50,10,11)==([],1,0)
    assert thinking_positions([10]+[1]*49,10,11)==(list(range(32,50)),0,50)
