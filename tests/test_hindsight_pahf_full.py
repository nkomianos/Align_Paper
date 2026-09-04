from interaction_sprint.hindsight_pahf_balanced import build_balanced_partitions
from interaction_sprint.hindsight_pahf_full import (
    EXPECTED_FULL_LEARNING_BASES,
    build_full_learning_base_partitions,
)


def _rows(count: int, *, evolved: bool, evaluation: bool = False):
    return [
        {
            "product": f"product-{i}",
            "Option A": ["a1", "a2", "a3"],
            "Option B": ["b1", "b2", "b3"],
            "Option C": ["c1", "c2", "c3"],
            "User": f"user-{i % 20}",
            "Task": f"{'eval' if evaluation else 'learn'} task {i}",
            "gt": ("C" if evaluation else "B") if evolved else "A",
        }
        for i in range(count)
    ]


def test_pinned_full_learning_repair_preserves_evaluation_and_balances_labels() -> None:
    base = build_full_learning_base_partitions(
        _rows(630, evolved=False),
        _rows(630, evolved=True),
        _rows(400, evolved=False, evaluation=True),
        _rows(400, evolved=True, evaluation=True),
    )
    partitions = build_balanced_partitions(base)
    assert base["invariants"]["learning"]["n"] == EXPECTED_FULL_LEARNING_BASES
    assert len(partitions["learning"]) == 4 * EXPECTED_FULL_LEARNING_BASES
    assert len(partitions["development"]) == 384
    assert len(partitions["confirmation"]) == 1024
    for split, bases in (("learning", 630), ("development", 96), ("confirmation", 256)):
        report = partitions["invariants"][split]
        assert report["base_records"] == bases
        assert set(report["old_target_counts"].values()) == {bases}
        assert set(report["new_target_counts"].values()) == {bases}
