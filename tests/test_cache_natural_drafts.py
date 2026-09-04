from copy import deepcopy

import numpy as np
import pytest

from interaction_sprint.cache_natural_drafts import ARMS, select_cases, summarize, views


class Tokenizer:
    def encode(self, text, add_special_tokens=True):
        return list(range(26))

    def convert_ids_to_tokens(self, ids):
        return ["word"]*len(ids)


def test_selection_deterministic_and_article_limited():
    texts = ["= First =", "Sentence alpha. Sentence beta. Sentence gamma.",
             "= Second =", "Sentence delta. Sentence epsilon. Sentence zeta."]
    config = {"min_tokens":20,"max_tokens":96,"min_side_tokens":8,
              "selection_seed":1,"cases":4,"max_per_article":2}
    cases=select_cases(texts,Tokenizer(),config)
    assert cases==select_cases(texts,Tokenizer(),config)
    assert sum(c["article"]=="= First =" for c in cases)==2
    assert len({c["id"] for c in cases})==4
    assert all(c["input_ids"][c["position"]]==c["gold_id"] for c in cases)


def test_disclosed_gold_not_present_in_inference_inputs():
    case={"input_ids":[1,5,6,7,8,2],"position":2,"gold_id":6}
    full,prefix=views(case,99,2)
    assert full.tolist()==[[1,5,99,7,8,2]]
    assert prefix.tolist()==[[1,5,99,2]]
    changed=deepcopy(case); changed["gold_id"]=71; changed["input_ids"][2]=71
    f2,p2=views(changed,99,2)
    assert (full==f2).all() and (prefix==p2).all()


def test_recovery_and_regression_are_paired():
    # Native truth [1,1,1,1]; draft succeeds only at case 2.
    predictions=[[0,1,1,0,0,1,1,1], [0,1,1,1,1,1,1,1],
                 [1,0,0,1,1,0,0,0], [0,0,0,0,0,0,0,0]]
    z=np.zeros((4,len(ARMS),3))
    for i,row in enumerate(predictions):
        for j,p in enumerate(row): z[i,j,p]=1
    cases=[{"gold_id":1,"article":str(i//2)} for i in range(4)]
    config={"cases":4,"equivalence_tolerance":1e-4,"minimum_recoverable_drafts":1,
            "minimum_net_clean_gain_cases":1}
    out=summarize(z,cases,config)
    assert out["initially_wrong"]==3 and out["recoverable_by_fresh"]==2
    assert out["fresh_only_correct"]==1 and out["stale_diagonal_only_correct"]==1
    assert out["fresh_over_stale_diagonal_net_correct"]==0
    assert out["arms"]["stale_diagonal"]["retained_wrong_draft_on_fresh_recoverable"]==1
    assert out["decision"]=="NO_USEFUL_CLEAN_VERIFICATION_GAIN_HOLD_EXPANSION"
    z[0,2,1]=10
    with pytest.raises(AssertionError): summarize(z,cases,config)
