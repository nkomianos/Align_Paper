import numpy as np
import pytest
from interaction_sprint.video_composite_replay import composite, replay


def test_endpoint_contracts():
    s=np.ones((4,5,3)); g=np.zeros_like(s)
    assert np.array_equal(composite(s,g,np.zeros((4,5))),s)
    assert np.array_equal(composite(s,g,np.ones((4,5))),g)


def test_support_floor_and_paired_contrast():
    s=np.ones((4,5,3)); ref=np.zeros_like(s); g=ref.copy()
    a=np.zeros((4,5)); a[:2]=1
    r=replay(s,g,ref,{'half':a,'all':np.ones_like(a)}, {'effect':np.ones_like(a,dtype=bool)})
    assert r['half']['effect']['mae']==.5
    assert r['half']['effect']['hard_support_error_floor']==.5
    assert r['all']['effect']['mae']==0
    assert np.array_equal(g,ref)


def test_soft_support_not_claimed_as_hard_floor():
    s=np.ones((2,2,3)); g=np.zeros_like(s); a=np.full((2,2),.5)
    r=replay(s,g,g,{'soft':a},{'all':np.ones((2,2),bool)})
    assert r['soft']['all']['mae']==.5
    assert r['soft']['all']['hard_support_error_floor']==0


def test_invalid_inputs():
    s=np.ones((2,2,3))
    with pytest.raises(ValueError): composite(s,s,np.ones((1,2)))
    with pytest.raises(ValueError): composite(s,s,np.full((2,2),np.nan))
    with pytest.raises(ValueError): replay(s,s,s,{'x':np.ones((2,2))},{'empty':np.zeros((2,2),bool)})
