"""Prospective full-learning repair for the EndoPAHF external assay."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from interaction_sprint.hindsight_pahf_balanced import build_balanced_partitions
from interaction_sprint.hindsight_pahf_pairs import (
    CONFIRMATION_SELECTION,
    DEVELOPMENT_SELECTION,
    build_changed_pairs,
    disjoint_hash_select,
    invariants,
    load_json_records,
)


EXPECTED_FULL_LEARNING_BASES = 630


def build_full_learning_base_partitions(
    learning_before: Sequence[Mapping[str, Any]],
    learning_after: Sequence[Mapping[str, Any]],
    evaluation_before: Sequence[Mapping[str, Any]],
    evaluation_after: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    """Keep every public learning pair while preserving the v1 evaluation split."""
    learning = build_changed_pairs(
        learning_before, learning_after, partition="learning"
    )
    if len(learning) != EXPECTED_FULL_LEARNING_BASES:
        raise ValueError(
            f"pinned source produced {len(learning)} rather than "
            f"{EXPECTED_FULL_LEARNING_BASES} learning pairs"
        )
    # Ordering is made independent of the source-file row order.  No outcomes
    # enter this ordering, and every available learning record is retained.
    learning = sorted(learning, key=lambda row: str(row["id"]))
    evaluation_pool = build_changed_pairs(
        evaluation_before, evaluation_after, partition="evaluation"
    )
    evaluation = disjoint_hash_select(
        evaluation_pool,
        (
            ("development", DEVELOPMENT_SELECTION),
            ("confirmation", CONFIRMATION_SELECTION),
        ),
    )
    return {
        "learning": learning,
        "development": evaluation["development"],
        "confirmation": evaluation["confirmation"],
        "invariants": {
            "learning_pool": invariants(learning),
            "evaluation_pool": invariants(evaluation_pool),
            "learning": invariants(learning),
            "development": invariants(evaluation["development"]),
            "confirmation": invariants(evaluation["confirmation"]),
        },
    }


def build_full_learning_balanced_partitions(source_root: Path) -> dict[str, object]:
    shopping = source_root / "data" / "shopping"
    base = build_full_learning_base_partitions(
        load_json_records(shopping / "phase1.json"),
        load_json_records(shopping / "phase3.json"),
        load_json_records(shopping / "phase2.json"),
        load_json_records(shopping / "phase4.json"),
    )
    balanced = build_balanced_partitions(base)  # type: ignore[arg-type]
    balanced["base_invariants"] = base["invariants"]
    return balanced
