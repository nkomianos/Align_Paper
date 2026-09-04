from types import SimpleNamespace
import copy
import pytest
from scripts.verify_sdpo_single_profile_calibration import native_check,replay_judge
from scripts.audit_sdpo_single_profile_qualification import audit


def test_native_capture_checks_prefix_eos_and_decode():
    tokenizer=SimpleNamespace(decode=lambda ids,**kw:'text')
    capture=dict(input_ids=[[1,2]],output_ids=[[1,2,7,9]],generation_config=dict(max_new_tokens=8,do_sample=False,eos_token_id=9))
    assert native_check(capture,'text',[1,2],tokenizer,8,False)==2
    bad=copy.deepcopy(capture);bad['output_ids'][0][-1]=8
    with pytest.raises(ValueError,match='EOS'):native_check(bad,'text',[1,2],tokenizer,8,False)
    with pytest.raises(ValueError,match='decoded'):native_check(capture,'different',[1,2],tokenizer,8,False)


def test_pairwise_order_replay_and_tampered_decision():
    judge=SimpleNamespace(_decide_from_scores=lambda s:[0 if s['A'][0]>s['B'][0] else 1],
                          _invert_ab=lambda values:[1-x for x in values])
    row=dict(AB_BA_scores=[{'A':[-.1],'B':[-2.],'C':[-3.]},{'A':[-2.],'B':[-.1],'C':[-3.]}],decision=0)
    assert replay_judge(row,judge)['order_consistent']
    row['decision']=1
    with pytest.raises(ValueError,match='decision'):replay_judge(row,judge)


def fixture():
    records=[]
    for i in range(16):
        record={'id':str(i)}
        for condition in ('ordinary','explicit','teacher'):
            record[condition]=dict(style_satisfactory=(i<8 or condition!='ordinary'),material_content_error=False,reason='Source checked; style rubric applied.')
        records.append(record)
    judgments=dict(calibration_manifest_sha256='abc',reviewer_kind='assistant_model_audit',reviewer_identity='model reviewer',rubric='fixed source/style rubric',records=records)
    verified=dict(verified_receipts=True,cases=16,calibration_manifest_sha256='abc',records=[{'id':str(i)} for i in range(16)])
    return judgments,verified


def test_qualification_keeps_ambiguity_and_never_approves():
    judgments,verified=fixture();result=audit(judgments,verified)
    assert result['descriptive_engineering_thresholds_met'] and result['approval'] is False
    judgments['records'][0]['teacher']['style_satisfactory']=None
    result=audit(judgments,verified)
    assert not result['descriptive_engineering_thresholds_met']
    assert result['original_satisfactory']==dict(n=8,successes=7,rate=.875)
    assert len(result['ambiguous_judgments'])==1


def test_qualification_bound_to_original_case_ids():
    judgments,verified=fixture();judgments['records'][1]['id']='0'
    with pytest.raises(ValueError,match='IDs'):audit(judgments,verified)
