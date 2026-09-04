from scripts.audit_c2c_answer_format import explicit_answer


def test_prefix_diagnostic_accepts_explicit_label_with_option_text():
    assert explicit_answer("The correct answer is D. weather") == "D"
    assert explicit_answer("A.") == "A"


def test_conservative_diagnostic_does_not_infer_labels_from_prose():
    assert explicit_answer("Use a bar graph.") is None
    assert explicit_answer("The best answer could be A or B.") is None
    assert explicit_answer("D. weather") is None
