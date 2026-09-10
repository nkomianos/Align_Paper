import pytest
from policy_value_replication_statistics import reversal_pvalue,holm_rejections,summarize_replication


def test_one_policy_effect_is_not_a_reversal():
    result=reversal_pvalue([32,0],[16,16],1)
    assert result['first_p']<1e-10
    assert result['joint_p']==result['second_p'] and result['joint_p']>.5


def test_relabeling_prefixes_preserves_frozen_direction_test():
    assert reversal_pvalue([29,3],[4,28],1)==reversal_pvalue([3,29],[28,4],-1)


def test_holm_uses_full_family_and_stops_at_first_failure():
    assert holm_rejections({'a':.001,'b':.03,'c':.04})==['a']


def test_duplicate_cases_are_not_independent_replicates():
    case={'id':'x','first':[32,0],'second':[0,32],'direction':1}
    with pytest.raises(ValueError):summarize_replication([case,case])
