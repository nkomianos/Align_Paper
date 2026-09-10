"""Prospective exact-number transport parser; no symbolic equivalence or eval."""
import re
from fractions import Fraction

VERSION = 'explicit-numeric-v2'
_NUMBER = r'[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'


def answer(text):
    """Parse the entire last #### answer line, including exact rational values.

    A malformed final marker invalidates extraction; never fall back to an older
    valid answer. Headings, units, expressions and incomplete fractions fail closed.
    """
    matches = list(re.finditer(r'(?m)^####[ \t]+([^\r\n]*)', text))
    if not matches:
        return None
    raw = matches[-1][1].strip()
    if len(raw) > 256:
        return None
    if raw.startswith('$') and raw.endswith('$') and len(raw) > 1:
        raw = raw[1:-1].strip()
    if re.fullmatch(_NUMBER, raw):
        return str(Fraction(raw.replace(',', '')))
    match = re.fullmatch(r'(' + _NUMBER + r')\s*/\s*(' + _NUMBER + r')', raw)
    if match:
        numerator, denominator = [Fraction(s.replace(',', '')) for s in match.groups()]
        return str(numerator / denominator) if denominator else None
    return None
