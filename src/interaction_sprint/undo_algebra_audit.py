"""Executable semantic audit, not a model experiment or a novel theorem.

Our fixture uses overwrite/clear registers, not stack-valued undo.  Universal
equivalence therefore means equality as state transformers for every initial
state, rather than equality only when all registers start empty.
"""
from itertools import product

from .fixtures import reduce_ops


def normal_form(operations):
    """Last operation per field, ordered by field; clear is not identity."""
    last = {}
    for kind, field, value in operations:
        if kind not in ("set", "clear"):
            raise ValueError(kind)
        last[field] = (kind, field, value if kind == "set" else None)
    return tuple(last[field] for field in sorted(last))


def relation_audit():
    initial = {"x": "prior"}
    purported_identity = [("set", "x", "new"), ("clear", "x", None)]
    after = reduce_ops(initial, purported_identity)
    alphabet = [("set", "x", "a"), ("set", "x", "b"),
                ("clear", "x", None), ("set", "y", "a"),
                ("clear", "y", None)]
    checked = 0
    for length in range(6):
        for history in product(alphabet, repeat=length):
            for start in ({}, {"x": "prior"}, {"x": "a", "y": "prior"}):
                assert reduce_ops(start, history) == reduce_ops(start, normal_form(history))
                checked += 1
    return {"false_identity_initial": initial, "false_identity_final": after,
            "identity_holds": initial == after, "normal_form_checks": checked,
            "claim": "Finite implementation checks, not evidence about LLM learning."}


if __name__ == "__main__":
    import json
    print(json.dumps(relation_audit(), indent=2))
