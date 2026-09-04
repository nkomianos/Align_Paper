"""Construct natural PAHF surfaces for expression/transition evaluation."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from interaction_sprint.pahf_source_audit import SHOPPING_SURFACE_FIELDS, stable_identifier


OPTION_LETTERS = ("A", "B", "C", "D")
LEARNING_SELECTION = 128
DEVELOPMENT_SELECTION = 96
CONFIRMATION_SELECTION = 256


def normalize_gt(value: object) -> str:
    label = "D" if value is None else str(value)
    if label not in OPTION_LETTERS:
        raise ValueError(f"invalid shopping target: {value!r}")
    return label


def _prompt(row: Mapping[str, Any]) -> str:
    product = str(row["product"])
    options = {
        letter: f"{product} with {' and '.join(str(value) for value in row[f'Option {letter}'])}"
        for letter in OPTION_LETTERS[:3]
    }
    options["D"] = "Do not buy any of these options"
    return (
        f"{row['User']}: {row['Task']}\n"
        + "\n".join(f"{letter}) {options[letter]}" for letter in OPTION_LETTERS)
        + "\nReply with exactly A, B, C, or D."
    )


def preference_followup(row: Mapping[str, Any], label: str) -> str:
    label = normalize_gt(None if label == "D" else label)
    if label == "D":
        return "None of these options fits my preference; I would rather not buy one."
    features = " and ".join(str(value) for value in row[f"Option {label}"])
    return f"My preference is Option {label}, the one with {features}."


def _assistant_response(label: str) -> str:
    if label == "D":
        return "I recommend not buying any of these options."
    return f"I recommend Option {label}."


def build_changed_pairs(
    original: Sequence[Mapping[str, Any]],
    evolved: Sequence[Mapping[str, Any]],
    *,
    partition: str,
) -> list[dict[str, object]]:
    if len(original) != len(evolved):
        raise ValueError("paired files have different row counts")
    pairs: list[dict[str, object]] = []
    for source_index, (before, after) in enumerate(zip(original, evolved)):
        if any(before[field] != after[field] for field in SHOPPING_SURFACE_FIELDS):
            continue
        old_target = normalize_gt(before["gt"])
        new_target = normalize_gt(after["gt"])
        if old_target == new_target:
            continue
        surface_sha = stable_identifier(before, SHOPPING_SURFACE_FIELDS)
        immediate = preference_followup(before, new_target)
        pairs.append({
            "id": f"pahf-{partition}-{surface_sha[:20]}",
            "partition": partition,
            "source_index": source_index,
            "surface_sha256": surface_sha,
            "transition": f"{old_target}->{new_target}",
            "old_target": old_target,
            "new_target": new_target,
            "prompt": _prompt(before),
            "assistant_response": _assistant_response(new_target),
            "immediate_followup": immediate,
            "expression_persistent_target": old_target,
            "transition_persistent_target": new_target,
            "delayed_expression_followup": preference_followup(before, old_target),
            "delayed_transition_followup": immediate,
        })
    ids = [str(pair["id"]) for pair in pairs]
    if len(ids) != len(set(ids)):
        raise ValueError("surface-derived pair IDs are not unique")
    return pairs


def hash_select(
    records: Sequence[Mapping[str, object]], count: int, *, salt: str,
) -> list[dict[str, object]]:
    if count < 1 or count > len(records):
        raise ValueError("selection count outside available records")
    ordered = sorted(
        records,
        key=lambda row: hashlib.sha256(f"{row['id']}|{salt}".encode("utf-8")).hexdigest(),
    )
    return [dict(row) for row in ordered[:count]]


def disjoint_hash_select(
    records: Sequence[Mapping[str, object]],
    counts: Sequence[tuple[str, int]],
) -> dict[str, list[dict[str, object]]]:
    remaining = [dict(record) for record in records]
    selected: dict[str, list[dict[str, object]]] = {}
    for name, count in counts:
        chosen = hash_select(remaining, count, salt=f"endo-pahf-{name}-v1")
        selected[name] = chosen
        chosen_ids = {str(row["id"]) for row in chosen}
        remaining = [row for row in remaining if str(row["id"]) not in chosen_ids]
    flat = [str(row["id"]) for group in selected.values() for row in group]
    if len(flat) != len(set(flat)):
        raise AssertionError("selection overlap")
    return selected


def invariants(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    transitions = Counter(str(row["transition"]) for row in records)
    return {
        "n": len(records),
        "unique_ids": len({str(row["id"]) for row in records}),
        "transition_counts": dict(sorted(transitions.items())),
        "ordinary_log_world_mismatches": sum(
            row["immediate_followup"] != row["delayed_transition_followup"]
            for row in records
        ),
        "expression_delayed_target_mismatches": sum(
            row["old_target"] != row["expression_persistent_target"]
            for row in records
        ),
        "transition_delayed_target_mismatches": sum(
            row["new_target"] != row["transition_persistent_target"]
            for row in records
        ),
        "expression_transition_delayed_followup_differences": sum(
            row["delayed_expression_followup"] != row["delayed_transition_followup"]
            for row in records
        ),
    }


def load_json_records(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"expected list of objects: {path}")
    return value


def build_pinned_partitions(source_root: Path) -> dict[str, object]:
    shopping = source_root / "data" / "shopping"
    learning_pool = build_changed_pairs(
        load_json_records(shopping / "phase1.json"),
        load_json_records(shopping / "phase3.json"),
        partition="learning",
    )
    evaluation_pool = build_changed_pairs(
        load_json_records(shopping / "phase2.json"),
        load_json_records(shopping / "phase4.json"),
        partition="evaluation",
    )
    learning = hash_select(
        learning_pool, LEARNING_SELECTION, salt="endo-pahf-learning-v1",
    )
    evaluation = disjoint_hash_select(evaluation_pool, (
        ("development", DEVELOPMENT_SELECTION),
        ("confirmation", CONFIRMATION_SELECTION),
    ))
    report = {
        "learning_pool": invariants(learning_pool),
        "evaluation_pool": invariants(evaluation_pool),
        "learning": invariants(learning),
        "development": invariants(evaluation["development"]),
        "confirmation": invariants(evaluation["confirmation"]),
    }
    return {
        "learning": learning,
        "development": evaluation["development"],
        "confirmation": evaluation["confirmation"],
        "invariants": report,
    }

