from interaction_sprint.hindsight_pahf_interface import (
    REHEARSAL_COUNT,
    context_specs,
    select_rehearsal_rows,
    summarize_scores,
)


def _record(index: int) -> dict[str, object]:
    return {
        "id": f"row-{index}",
        "prompt": "A) alpha\nB) beta\nC) gamma\nD) none",
        "old_target": "A",
        "new_target": "B",
        "immediate_followup": "I prefer B.",
        "delayed_expression_followup": "I prefer A.",
        "delayed_transition_followup": "I prefer B.",
    }


def test_rehearsal_selection_is_deterministic_and_unique() -> None:
    records = [_record(index) for index in range(40)]
    first = select_rehearsal_rows(records)
    second = select_rehearsal_rows(list(reversed(records)))
    assert first == second
    assert len(first) == REHEARSAL_COUNT
    assert len({row["id"] for row in first}) == REHEARSAL_COUNT


def test_context_targets_separate_expression_and_transition() -> None:
    contexts = context_specs(_record(0))
    assert [(row["context"], row["target"]) for row in contexts] == [
        ("immediate", "B"),
        ("delayed_expression", "A"),
        ("delayed_transition", "B"),
    ]
    assert "I prefer A." in contexts[1]["text"]
    assert "I prefer B." in contexts[2]["text"]


def test_summary_qualification_is_recomputed() -> None:
    scores = []
    for context in ("immediate", "delayed_expression", "delayed_transition"):
        for index in range(REHEARSAL_COUNT):
            scores.append({
                "context": context,
                "normalized_correct": index < 28,
                "normalized_target_probability": 0.8,
                "full_vocabulary_choice_mass": 0.2,
            })
    summary = summarize_scores(scores)
    assert summary["decision"] == "CPU_REHEARSAL_ONLY_INTERFACE_QUALIFIED"
    for row in scores[:17]:
        row["full_vocabulary_choice_mass"] = 0.0
    summary = summarize_scores(scores)
    assert summary["decision"] == "CPU_REHEARSAL_ONLY_INTERFACE_UNQUALIFIED"
