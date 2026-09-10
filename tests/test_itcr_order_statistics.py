"""Enumerate discrete i.i.d. populations to exercise ties and small calibration."""
from fractions import Fraction
from itertools import product

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
