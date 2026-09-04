import pytest

from latent_contract.answer_scoring import option_label
from scripts.run_c2c_text_baseline import messages, SPEC


@pytest.mark.parametrize("text,expected", [("A", "A"), ("D. weather", "D"),
    ("The correct answer is B. Fe", "B"), ("The correct answer is: C", "C"),
    ("A or B", None), ("Use a bar graph.", None), ("The answer might be A.", None)])
def test_explicit_labels_not_stray_letters(text, expected):
    assert option_label(text) == expected


def test_context_transfer_preserves_upstream_conversation_structure():
    turns = messages("Q?", "Q? Choices...", "Fact.")
    assert [m["role"] for m in turns] == ["user", "assistant", "user"]
    assert turns[0]["content"] == SPEC["background_prompt"].format(question="Q?")
    assert turns[1]["content"] == "Fact."
    assert turns[2]["content"] == "Q? Choices..."
    assert len(messages("Q?", "not included")) == 1
