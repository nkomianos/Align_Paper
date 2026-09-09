from render_malt_api_records import pack,unpack
import pytest


def test_roundtrip_preserves_candidates_repeats_nulls_and_metadata():
    m={'role':'assistant','content':'same','metadata':{'node_id':None}}
    samples=[{'input':[m,m],'output':[[m],[]],'metadata':{'unmatched':True}}]
    r=pack(samples)
    assert len(r['messages'])==1
    assert r['requests'][0]['output_candidates']==[[0],[]]
    assert unpack(r)==samples


def test_different_metadata_does_not_silently_merge_nodes():
    a={'content':'same','metadata':{'node_id':1}}
    b={'content':'same','metadata':{'node_id':2}}
    assert len(pack([{'input':[a,b],'output':[],'metadata':{}}])['messages'])==2
    with pytest.raises(ValueError):pack([{'input':[],'output':[],'metadata':{},'new_field':1}])
