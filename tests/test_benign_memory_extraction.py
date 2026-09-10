import pytest
from run_benign_memory_extraction import solve,parse,inputs


def test_incomplete_order_can_have_certain_decision():
    assert solve([1,7,9],[[0,1],[0,2]],5)=='YES'
    assert solve([1,3,9],[[0,1],[0,2]],5)=='CLARIFY'
    assert solve([9,3],[[0,1]],5)=='NO'


def test_cycles_rejected():
    with pytest.raises(ValueError):solve([1,2],[[0,1],[1,0]],1)


def test_fence_transport_and_paired_labels():
    assert parse('```json\n{"answer":"YES"}\n```')=={'answer':'YES'}
    assert parse('Preface {"answer":"YES"}') is None
    rows=inputs();assert len(rows)==48
    assert all(rows[i]['gold']==rows[i+1]['gold'] for i in range(0,48,2))
