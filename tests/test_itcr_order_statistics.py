"""Enumerate discrete i.i.d. populations to exercise ties and small calibration."""
from fractions import Fraction
from itertools import product
import math

from audit_itcr_released_data import lower_threshold, upper_threshold


def test_repaired_ranks_include_ties_and_empty_calibration():
    for n in range(6):
        for alpha in [Fraction(1, 10), Fraction(1, 4), Fraction(1, 2)]:
            lower_errors = upper_errors = 0
            for sample in product(range(3), repeat=n+1):
                calibration, test = sample[:-1], sample[-1]
                lower_errors += test < lower_threshold(calibration, alpha)
                upper_errors += test > upper_threshold(calibration, alpha)
            total = 3 ** (n+1)
            assert Fraction(lower_errors, total) <= alpha
            assert Fraction(upper_errors, total) <= alpha


def test_marginal_first_bad_scores_allow_no_bad_sentinel():
    # Uniform i.i.d. mixture over two finite first-bad scores and no-bad graphs.
    for n in range(6):
        alpha = Fraction(1, 4)
        errors = 0
        for sample in product([0.5, 0.75, math.inf], repeat=n+1):
            threshold = lower_threshold(sample[:-1], alpha)
            errors += math.isfinite(sample[-1]) and sample[-1] < threshold
        assert Fraction(errors, 3 ** (n+1)) <= alpha
