import numpy as np
from interaction_sprint.byte_coupling_native import byte_decoder, NativeSampler, byte_event_clock


def test_byte_alphabet_round_trip():
    decoder=byte_decoder(); assert len(decoder)==256
    assert sorted(decoder.values())==list(range(256))
    assert decoder['a']==97 and decoder['Ġ']==32


def test_native_sampler_group_probability():
    s=NativeSampler([b'raw:a',b'raw:aa',b'raw:b'],[b'byte:a',b'byte:a',b'byte:b'],[1,2,1])
    out=[s.draw(np.log([.1,.3,.6]),i,0,'byte_hierarchical') for i in range(10000)]
    assert np.max(abs(np.bincount(out,minlength=3)/10000-[.1,.3,.6])) < .015


def test_greedy_native_distribution_unchanged():
    s=NativeSampler([b'a',b'b'],[b'a',b'b'],[1,1])
    assert s.draw([2,1],100,0,'greedy')==0


def test_silent_event_clock_does_not_reuse_noise():
    clocks=[byte_event_clock(0,0),byte_event_clock(0,1),byte_event_clock(0,2),byte_event_clock(1,0)]
    assert len(set(clocks))==4
    s=NativeSampler([b'raw:a',b'opaque:0'],[b'byte:a',b'opaque:0'],[1,0])
    assert s.draw([0,1],10,clocks[1],'greedy')==1
