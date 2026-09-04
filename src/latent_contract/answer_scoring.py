"""Explicit option-label scoring for prospective communication comparisons.

Existing frozen reports are unchanged. Do not infer labels from stray letters
in prose. A clear label may be followed by option text after punctuation.
"""
import re


def option_label(text):
    match = re.match(r'^(?:The correct answer is\s*:?\s*)?([ABCD])(?:[.)](?:\s|$)|\s*$)',
                     text.strip(), flags=re.IGNORECASE)
    return match.group(1).upper() if match else None
