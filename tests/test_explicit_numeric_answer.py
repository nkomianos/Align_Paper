import pytest
from explicit_numeric_answer import answer


@pytest.mark.parametrize('raw,expected', [
    ('4/3', '4/3'), ('1,234.50', '2469/2'), ('-2', '-2'),
    ('$3 / 2$', '3/2'), ('-0', '0'), ('2.5/0.5', '5'),
])
def test_exact_whole_values(raw, expected):
    assert answer('Reasoning\n#### ' + raw) == expected


@pytest.mark.parametrize('raw', ['4/0', '4/', '4x', '4 meters', '1e3',
                                    '4 + 2', '4,5', '3.', '2 or 3', '$4', '2'*257])
def test_no_numeric_prefix_or_heading_acceptance(raw):
    assert answer('#### ' + raw) is None


def test_last_marker_is_authoritative_even_when_malformed():
    assert answer('#### 2\nCorrection:\n#### 4/3') == '4/3'
    assert answer('#### 2\nCorrection:\n#### 4/') is None
    assert answer('#### 2\n#### ') is None
    assert answer('some prose #### 4') is None
