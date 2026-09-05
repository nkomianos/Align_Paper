import math
import pytest
from verify_hindsight_pahf_reduced_dev_v2_portable import _same_numeric


def test_one_ulp_is_numerical_replay_not_a_new_decision():
    _same_numeric({'gain':math.nextafter(.2,1.),'gate':False},{'gain':.2,'gate':False},'result')


@pytest.mark.parametrize('actual,expected',[
    ({'gain':.2001},{'gain':.2}),
    ({'gain':.2,'gate':True},{'gain':.2,'gate':False}),
    ({'id':'altered'},{'id':'original'}),
    ({'gain':float('nan')},{'gain':.2}),
])
def test_meaningful_changes_fail(actual,expected):
    with pytest.raises(ValueError):_same_numeric(actual,expected,'result')
