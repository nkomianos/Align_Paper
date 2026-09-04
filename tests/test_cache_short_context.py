import numpy as np
from interaction_sprint.cache_short_context import views, ExcludingTokenizer, summarize, ARMS


def test_short_context_keeps_exactly_three_right_tokens_not_target():
    c={"position":2,"input_ids":[1,5,6,7,8,9,10,11,2]}
    full,short=views(c,99,2,3)
    assert full.tolist()==[[1,5,99,7,8,9,10,11,2]]
    assert short.tolist()==[[1,5,99,7,8,9,2]]
    c["input_ids"][2]=31
    f2,s2=views(c,99,2,3)
    assert (full==f2).all() and (short==s2).all()


def test_exclusion_precedes_selection():
    class T:
        def encode(self,text,**kwargs): return [1,2,3]
        def convert_ids_to_tokens(self,ids): return ["a"]*len(ids)
    t=ExcludingTokenizer(T(),["already seen"])
    assert t.encode("already seen")==[]
    assert t.encode("new sentence")==[1,2,3]


def test_numerical_gain_cannot_override_invalid_drafts():
    z=np.zeros((4,len(ARMS),3))
    z[:,:,0]=1
    for j in (1,2,7): z[:,j,1]=2
    cases=[{"gold_id":1,"article":str(i)} for i in range(4)]
    cfg={"cases":4,"equivalence_tolerance":1e-4,"minimum_recoverable_drafts":1,
         "minimum_net_clean_gain_cases":1,"minimum_lexical_fraction":.8,
         "minimum_initial_correct":1,"minimum_initial_wrong":1}
    result=summarize(z,cases,cfg,[{"draft_lexical":False}]*4)
    assert result["numerical_comparison_decision"]=="DEVELOPMENTAL_CLEAN_GAIN_INVESTIGATE_NOT_PAPER_GO"
    assert result["decision"]=="INVALID_DRAFT_DISTRIBUTION_DO_NOT_INTERPRET_GAIN"
