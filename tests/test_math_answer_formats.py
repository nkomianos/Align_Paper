from fractions import Fraction
from audit_math_answer_formats import numeric, extract


def test_no_numeric_prefix_false_positives():
    for text in ('4x', '4/0', '4 meters', '1e3', '4 + 2', '4,5'):
        assert numeric(text) is None
    assert numeric('4/3') == Fraction(4, 3)
    assert numeric(r'\frac{4}{3}') == Fraction(4, 3)


def test_nested_box_and_later_final_answer():
    assert extract(r'\boxed{\frac{4}{3}}')['value'] == '4/3'
    assert extract('Earlier '+r'\boxed{4}'+'\n#### 3')['value'] == '3'
    assert extract('#### 4/3')['value'] == '4/3'


def test_incomplete_box_and_symbolic_value_remain_unresolved():
    assert extract(r'\boxed{\frac{4}{3}')['kind'] == 'unextracted'
    value = extract(r'\boxed{\sqrt{2}}')
    assert value['kind'] == 'boxed' and value['value'] is None
