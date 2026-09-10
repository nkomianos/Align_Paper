from prepare_thinking_pmi_traces import select_positions


def test_selection_ignores_ineligible_positions_and_breaks_ties_early():
    result=select_positions([100.,1.,2.,2.,200.],[1,2,3],'problem-a')
    assert result['high_entropy']==2
    assert result['random'] in [1,2,3]
    assert result==select_positions([100.,1.,2.,2.,200.],[1,2,3],'problem-a')
    assert select_positions([1.],[],'empty') is None
