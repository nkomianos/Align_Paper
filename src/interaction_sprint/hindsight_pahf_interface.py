"""Deterministic helpers for the EndoPAHF CPU interface rehearsal."""
from __future__ import annotations

import hashlib
from typing import Mapping, Sequence


OPTION_LETTERS = ("A", "B", "C", "D")
REHEARSAL_COUNT = 32
REHEARSAL_SALT = "endo-pahf-interface-rehearsal-v1"


def select_rehearsal_rows(
    records: Sequence[Mapping[str, object]], count: int = REHEARSAL_COUNT,
) -> list[dict[str, object]]:
    if count < 1 or count > len(records):
        raise ValueError("invalid rehearsal count")
    ordered = sorted(
        records,
        key=lambda row: hashlib.sha256(
            f"{row['id']}|{REHEARSAL_SALT}".encode("utf-8")
        ).hexdigest(),
    )
    return [dict(row) for row in ordered[:count]]


def preference_query(prompt: str, feedback: str) -> str:
    """Render a deliberately explicit extraction query, not a training prompt."""
    return (
        f"{prompt}\n\n"
        "The following is the user's latest recorded preference statement:\n"
        f"{feedback}\n\n"
        "Using that latest recorded preference, answer the original choice now. "
        "Reply with exactly A, B, C, or D."
    )


def context_specs(row: Mapping[str, object]) -> tuple[dict[str, str], ...]:
    return (
        {
            "context": "immediate",
            "target": str(row["new_target"]),
            "text": preference_query(str(row["prompt"]), str(row["immediate_followup"])),
        },
        {
            "context": "delayed_expression",
            "target": str(row["old_target"]),
            "text": preference_query(
                str(row["prompt"]), str(row["delayed_expression_followup"])
            ),
        },
        {
            "context": "delayed_transition",
            "target": str(row["new_target"]),
            "text": preference_query(
                str(row["prompt"]), str(row["delayed_transition_followup"])
            ),
        },
    )


def summarize_scores(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    expected = {"immediate", "delayed_expression", "delayed_transition"}
    contexts: dict[str, dict[str, float | int]] = {}
    for context in sorted(expected):
        selected = [row for row in rows if row["context"] == context]
        if not selected:
            raise ValueError(f"missing context: {context}")
        contexts[context] = {
            "n": len(selected),
            "correct": sum(bool(row["normalized_correct"]) for row in selected),
            "mean_normalized_target_probability": sum(
                float(row["normalized_target_probability"]) for row in selected
            ) / len(selected),
            "mean_full_vocabulary_choice_mass": sum(
                float(row["full_vocabulary_choice_mass"]) for row in selected
            ) / len(selected),
            "min_full_vocabulary_choice_mass": min(
                float(row["full_vocabulary_choice_mass"]) for row in selected
            ),
        }
    if set(contexts) != expected:
        raise ValueError("unexpected contexts")
    gates = {
        "each_context_at_least_28_of_32_correct": all(
            value["n"] == REHEARSAL_COUNT and value["correct"] >= 28
            for value in contexts.values()
        ),
        "each_context_mean_target_probability_at_least_point_70": all(
            value["mean_normalized_target_probability"] >= 0.70
            for value in contexts.values()
        ),
        "each_context_mean_choice_mass_at_least_point_10": all(
            value["mean_full_vocabulary_choice_mass"] >= 0.10
            for value in contexts.values()
        ),
    }
    return {
        "contexts": contexts,
        "gates": gates,
        "decision": (
            "CPU_REHEARSAL_ONLY_INTERFACE_QUALIFIED"
            if all(gates.values())
            else "CPU_REHEARSAL_ONLY_INTERFACE_UNQUALIFIED"
        ),
        "paper_green_light": False,
    }
