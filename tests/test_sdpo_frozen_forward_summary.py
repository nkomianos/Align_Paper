import pytest
from scripts.summarize_sdpo_frozen_forward import describe


def test_describe_retains_denominators_and_range():
    result=describe([-2.,0.,1.])
    assert result['n']==3
    assert result['mean']==pytest.approx(-1/3)
    assert result['median']==0
    assert result['minimum']==-2
    assert result['maximum']==1
