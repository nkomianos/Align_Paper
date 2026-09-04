from scripts.audit_smoke_interfaces import leading_letter
from scripts.diagnose_lc0_text_interface import final_answer
from scripts.run_lc0_reasoning_development import SPEC, final_answer as development_answer


def test_leading_letter_is_explicit_and_not_semantic_rescue():
    assert leading_letter("A. code_7288", "A-D") == "A"
    assert leading_letter(" B ", "A-D") == "B"
    assert leading_letter("Answer is A", "A-D") is None
    assert leading_letter("APPLE", "A-D") is None
    assert leading_letter("E", "A-D") is None


def test_reasoning_must_end_and_final_format_stays_strict():
    assert final_answer("I think A", True) is None
    assert final_answer("reasoning\n</think>\n\nB", True) == "B"
    assert final_answer("reasoning\n</think>\nB. code_1234", True) is None
    assert final_answer("C", False) == "C"
    assert final_answer("C. code_1234", False) is None


def test_development_uses_fresh_pairs_and_fixed_budget():
    assert SPEC["pair_ids"] == list(range(4, 12))
    assert SPEC["max_new_tokens"] == 512
    assert development_answer("unfinished reasoning with A") is None
    assert development_answer("reasoning</think>\nA") == "A"
    assert development_answer("reasoning</think>\nA. code_1234") is None
