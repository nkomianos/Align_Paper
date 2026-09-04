import copy
from scripts.audit_sdpo_generation_config import replay

D=dict(do_sample=False,temperature=1.,top_k=50,top_p=1.,bos_token_id=None,eos_token_id=None,pad_token_id=None,decoder_start_token_id=None)
M={**D,'do_sample':True,'temperature':.6,'top_k':20,'top_p':.95,'bos_token_id':151643,'transformers_version':'4.51.0'}

def test_false_is_overridden_without_mutating_passed():
    passed=copy.deepcopy(D);result=replay(passed,M,D)
    assert result['do_sample'] is True and result['temperature']==.6
    assert passed==D

def test_simulator_custom_temperature_survives():
    result=replay({**D,'do_sample':True,'temperature':.7},M,D)
    assert result['temperature']==.7 and result['top_k']==20 and result['top_p']==.95

def test_explicit_kwarg_has_highest_priority():
    assert replay(D,M,D,do_sample=False)['do_sample'] is False

def test_disable_model_defaults_retains_greedy_but_fills_tokens():
    result=replay(D,M,D,use_model_defaults=False)
    assert result['do_sample'] is False and result['temperature']==1.
    assert result['bos_token_id']==151643
