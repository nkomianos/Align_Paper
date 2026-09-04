from interaction_sprint.hindsight_generation_audit import classify


def test_whitespace_is_only_strict_relaxation():
    assert classify(' \nA\n')['strict_label'] == 0
    assert classify('Answer: B')['strict_label'] is None
    assert classify('Answer: B')['unique_mentioned_label'] == 1


def test_ambiguous_or_embedded_labels_are_not_recovered():
    assert classify('A or B')['unique_mentioned_label'] is None
    assert classify('ABC')['unique_mentioned_label'] is None
    assert classify('')['strict_label'] is None


def test_unique_mention_is_not_a_semantic_judge():
    # Explicitly documents a limitation; exploratory parser must not be
    # presented as semantic correctness or used to repair the original gate.
    assert classify('Do not choose A')['unique_mentioned_label'] == 0
