"""Counterbalance EndoPAHF option labels without changing option semantics."""
from __future__ import annotations

from collections import Counter
import hashlib
from typing import Mapping, Sequence

from interaction_sprint.hindsight_pahf_pairs import OPTION_LETTERS


ROTATIONS = (0, 1, 2, 3)


def _parse_prompt(prompt: str) -> tuple[list[str], dict[str, str], list[str]]:
    lines = prompt.splitlines()
    positions: dict[str, int] = {}
    options: dict[str, str] = {}
    for letter in OPTION_LETTERS:
        matches = [index for index, line in enumerate(lines) if line.startswith(f"{letter}) ")]
        if len(matches) != 1:
            raise ValueError(f"prompt does not contain exactly one {letter} option")
        positions[letter] = matches[0]
        options[letter] = lines[matches[0]][3:]
    ordered_positions = [positions[letter] for letter in OPTION_LETTERS]
    if ordered_positions != list(range(ordered_positions[0], ordered_positions[0] + 4)):
        raise ValueError("prompt options are not one ordered contiguous block")
    start = ordered_positions[0]
    return lines[:start], options, lines[start + 4:]


def _mapping(rotation: int) -> dict[str, str]:
    if rotation not in ROTATIONS:
        raise ValueError("invalid rotation")
    # New position i displays the original option i + rotation.
    return {
        original: OPTION_LETTERS[(index - rotation) % 4]
        for index, original in enumerate(OPTION_LETTERS)
    }


def _preference(label: str, text: str) -> str:
    return f"My latest preference is Option {label}: {text}."


def _recommendation(label: str, text: str) -> str:
    return f"I recommend Option {label}: {text}."


def rotate_record(record: Mapping[str, object], rotation: int) -> dict[str, object]:
    header, original_options, tail = _parse_prompt(str(record["prompt"]))
    mapping = _mapping(rotation)
    displayed = {
        new_letter: original_options[OPTION_LETTERS[(new_index + rotation) % 4]]
        for new_index, new_letter in enumerate(OPTION_LETTERS)
    }
    prompt = "\n".join(
        header
        + [f"{letter}) {displayed[letter]}" for letter in OPTION_LETTERS]
        + tail
    )
    old_target = mapping[str(record["old_target"])]
    new_target = mapping[str(record["new_target"])]
    immediate = _preference(new_target, displayed[new_target])
    result = dict(record)
    result.update({
        "id": f"{record['id']}-rotation-{rotation}",
        "base_id": str(record["id"]),
        "label_rotation": rotation,
        "source_transition": str(record["transition"]),
        "transition": f"{old_target}->{new_target}",
        "old_target": old_target,
        "new_target": new_target,
        "prompt": prompt,
        "assistant_response": _recommendation(new_target, displayed[new_target]),
        "immediate_followup": immediate,
        "expression_persistent_target": old_target,
        "transition_persistent_target": new_target,
        "delayed_expression_followup": _preference(old_target, displayed[old_target]),
        "delayed_transition_followup": immediate,
        "permuted_surface_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    })
    return result


def counterbalance(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    balanced = [rotate_record(record, rotation) for record in records for rotation in ROTATIONS]
    ids = [str(record["id"]) for record in balanced]
    if len(ids) != len(set(ids)):
        raise ValueError("counterbalanced IDs are not unique")
    return balanced


def balanced_invariants(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    base_counts = Counter(str(row["base_id"]) for row in records)
    rotations = Counter(int(row["label_rotation"]) for row in records)
    old_targets = Counter(str(row["old_target"]) for row in records)
    new_targets = Counter(str(row["new_target"]) for row in records)
    return {
        "n": len(records),
        "unique_ids": len({str(row["id"]) for row in records}),
        "base_records": len(base_counts),
        "four_rotations_per_base": all(count == 4 for count in base_counts.values()),
        "rotation_counts": {str(key): rotations[key] for key in ROTATIONS},
        "old_target_counts": {key: old_targets[key] for key in OPTION_LETTERS},
        "new_target_counts": {key: new_targets[key] for key in OPTION_LETTERS},
        "ordinary_log_world_mismatches": sum(
            row["immediate_followup"] != row["delayed_transition_followup"]
            for row in records
        ),
        "expression_transition_probe_differences": sum(
            row["delayed_expression_followup"] != row["delayed_transition_followup"]
            for row in records
        ),
    }


def build_balanced_partitions(
    partitions: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, object]:
    result = {name: counterbalance(partitions[name]) for name in (
        "learning", "development", "confirmation"
    )}
    result["invariants"] = {
        name: balanced_invariants(result[name])
        for name in ("learning", "development", "confirmation")
    }
    return result
