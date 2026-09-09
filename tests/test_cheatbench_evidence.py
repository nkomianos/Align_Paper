import pytest
from audit_cheatbench_evidence import resolve,locate_snippet


def test_paths_and_repeated_matches_are_preserved():
    obj={'events':[{'content':'abc abc'},{'content':'abc'}]}
    assert resolve(obj,'events[0].content')=='abc abc'
    hits=locate_snippet(obj,'abc')
    assert [(h['path'],h['char_start']) for h in hits]==[('events[0].content',0),('events[0].content',4),('events[1].content',0)]
    assert locate_snippet(obj,'')==[]
    with pytest.raises(IndexError):resolve(obj,'events[2].content')
