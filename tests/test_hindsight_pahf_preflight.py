from interaction_sprint.hindsight_pahf_preflight import (
    BASE_COUNT,
    build_jobs,
    select_base_panel,
    summarize_preflight,
)


def _variant(base: int, rotation: int) -> dict[str, object]:
    letters = "ABCD"
    return {
        "id": f"base-{base}-r{rotation}",
        "base_id": f"base-{base}",
        "label_rotation": rotation,
        "prompt": "A) alpha\nB) beta\nC) gamma\nD) none",
        "old_target": letters[rotation],
        "new_target": letters[(rotation + 1) % 4],
        "immediate_followup": "new",
        "delayed_expression_followup": "old",
    }


def test_base_selection_keeps_all_rotations_and_is_deterministic() -> None:
    records = [_variant(base, rotation) for base in range(20) for rotation in range(4)]
    first = select_base_panel(records)
    second = select_base_panel(list(reversed(records)))
    assert first == second
    assert len(first) == BASE_COUNT * 4
    assert len({row["base_id"] for row in first}) == BASE_COUNT
    for base_id in {row["base_id"] for row in first}:
        assert {row["label_rotation"] for row in first if row["base_id"] == base_id} == {0, 1, 2, 3}


def test_jobs_cover_two_contexts_and_each_label() -> None:
    records = [_variant(0, rotation) for rotation in range(4)]
    jobs = build_jobs(records)
    assert len(jobs) == 8
    assert {job["context"] for job in jobs} == {"immediate", "delayed_expression"}
    for context in ("immediate", "delayed_expression"):
        assert {job["target"] for job in jobs if job["context"] == context} == set("ABCD")


def test_summary_requires_every_context_label_cell() -> None:
    scores = []
    for context in ("immediate", "delayed_expression"):
        for target in "ABCD":
            for index in range(BASE_COUNT):
                scores.append({
                    "context": context,
                    "target": target,
                    "normalized_correct": index < 14,
                    "normalized_target_probability": 0.8,
                    "full_vocabulary_choice_mass": 0.5,
                })
    assert summarize_preflight(scores)["qualified"]
    scores[0]["normalized_correct"] = False
    assert not summarize_preflight(scores)["qualified"]
