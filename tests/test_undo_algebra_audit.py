from interaction_sprint.fixtures import reduce_ops
from interaction_sprint.undo_algebra_audit import normal_form, relation_audit


def test_clear_is_not_general_inverse():
    assert reduce_ops({"x": "prior"}, [("set", "x", "new"), ("clear", "x", None)]) == {}
    assert normal_form([("set", "x", "new"), ("clear", "x", None)]) == (("clear", "x", None),)


def test_last_write_and_independence():
    history = [("set", "x", "a"), ("set", "y", "b"), ("set", "x", "c")]
    assert normal_form(history) == (("set", "x", "c"), ("set", "y", "b"))
    assert normal_form(history + [("clear", "x", None)]) == (("clear", "x", None), ("set", "y", "b"))


def test_exhaustive_short_transformer_check():
    report = relation_audit()
    assert report["identity_holds"] is False
    assert report["normal_form_checks"] == 11718
