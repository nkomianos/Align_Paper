import pytest
from interaction_sprint.belief_reconstruction_views import parse_transcript,prefix,build_prompt


def test_nested_prefix_and_future_exclusion():
    turns=parse_transcript('USER: early\nBOT: reply\nUSER: later\nBOT: future',4)
    assert prefix(turns,2)[:len(prefix(turns,1))]==prefix(turns,1)
    assert 'future' not in str(build_prompt('statement',turns,2))
    assert 'reply' not in str(build_prompt('statement',turns,2,user_only=True))
    assert 'reply' in str(build_prompt('statement',turns,2))


def test_invalid_boundaries_fail_without_echoing_text():
    for text,n in [('USER: secret\nUSER: another',2),('USER: secret',2),('secret',1)]:
        with pytest.raises(ValueError) as error:parse_transcript(text,n)
        assert 'secret' not in str(error.value)


def test_no_empty_turns_or_missing_prefix():
    with pytest.raises(ValueError):parse_transcript('USER:\nBOT: yes',2)
    with pytest.raises(ValueError):prefix([dict(role='USER',content='x')],2)
