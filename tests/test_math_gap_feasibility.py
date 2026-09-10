import itertools
from fractions import Fraction
from audit_math_gap_feasibility import gap_bound


def test_interval_bound_equals_exhaustive_binary_completions():
    def completions(xs):
        for bits in itertools.product((0,1),repeat=xs.count(None)):
            it=iter(bits)
            yield [next(it) if x is None else x for x in xs]
    for first,second in itertools.product(itertools.product((0,1,None),repeat=2),repeat=2):
        expected=max(abs(Fraction(sum(a),len(a))-Fraction(sum(b),len(b)))
                     for a in completions(first) for b in completions(second))
        assert gap_bound(first,second)[2]==expected
