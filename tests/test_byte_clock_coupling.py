import numpy as np
import pytest

from interaction_sprint.byte_clock_coupling import categorical, draw_model, mix64, noise, validate_model


def test_noise_and_marginals():
    seeds = mix64(np.arange(100000,dtype=np.uint64))
    clocks = np.zeros(len(seeds),dtype=int)
    g = noise(seeds,clocks,[b"a",b"b"],b"test")
    assert np.isfinite(g).all()
    for policy in ["independent","token_clock","byte_clock","byte_hierarchical"]:
        x = categorical([.15,.35,.5],[b"a",b"ab",b"b"],seeds,clocks,policy)
        assert np.max(abs(np.bincount(x,minlength=3)/len(x)-[.15,.35,.5])) < .006


def test_zero_mass_and_deterministic_branch():
    seeds = np.arange(1000,dtype=np.uint64)
    x = categorical([0,1,0],[b"a",b"ab",b"b"],seeds,np.zeros(1000),"byte_hierarchical")
    assert (x==1).all()


def test_token_id_permutation_invariance():
    seeds = np.arange(10000,dtype=np.uint64)
    labels = [b"x",b"xy",b"z"]
    order = np.array([2,0,1]); p=np.array([.2,.3,.5])
    for policy in ["token_clock","byte_clock","byte_hierarchical"]:
        a = categorical(p,labels,seeds,np.zeros(len(seeds)),policy)
        b = categorical(p[order],[labels[i] for i in order],seeds,np.zeros(len(seeds)),policy)
        assert (a==order[b]).all()


def test_adaptive_tree_native_marginal():
    model=[dict(tokens=["a","a"],p=.1),dict(tokens=["a","bb"],p=.2),dict(tokens=["bbb","a"],p=.7)]
    seeds=mix64(np.arange(100000,dtype=np.uint64))
    for policy in ["byte_clock","byte_hierarchical"]:
        out=draw_model(model,seeds,policy,0)
        assert np.max(abs(np.bincount(out,minlength=3)/len(out)-[.1,.2,.7])) < .006


def test_retokenization_controls():
    a=[dict(tokens=["answer:","red"],p=.55),dict(tokens=["answer:","blue"],p=.45)]
    b=[dict(tokens=["an","swer",":","r","ed"],p=.55),dict(tokens=["an","swer",":","b","lue"],p=.45)]
    seeds=mix64(np.arange(10000,dtype=np.uint64))
    assert (draw_model(a,seeds,"byte_hierarchical",0)==draw_model(b,seeds,"byte_hierarchical",1)).all()
    assert np.mean(draw_model(a,seeds,"token_clock",0)==draw_model(b,seeds,"token_clock",1)) < .55


def test_invalid_inputs():
    with pytest.raises(ValueError):
        validate_model([dict(tokens=["a"],p=.5),dict(tokens=["a","b"],p=.5)])
    with pytest.raises(ValueError):
        categorical([1],[b""],np.array([1]),np.array([0]),"byte_clock")
