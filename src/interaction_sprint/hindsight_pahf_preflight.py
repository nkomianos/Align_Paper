"""Prospective capable-model qualification for counterbalanced EndoPAHF."""
from __future__ import annotations

import hashlib
from typing import Mapping, Sequence

from interaction_sprint.hindsight_neural_anchor import hindsight_user_text
from interaction_sprint.hindsight_pahf_interface import OPTION_LETTERS


BASE_COUNT = 16
SELECTION_SALT = "endo-pahf-capable-preflight-v2-exact-hindsight"
CONTEXTS = ("immediate", "delayed_expression")


def select_base_panel(
    records: Sequence[Mapping[str, object]], count: int = BASE_COUNT,
) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in records:
        grouped.setdefault(str(row["base_id"]), []).append(row)
    eligible = {
        base_id: rows for base_id, rows in grouped.items()
        if {int(row["label_rotation"]) for row in rows} == {0, 1, 2, 3}
        and len(rows) == 4
    }
    if count < 1 or count > len(eligible):
        raise ValueError("invalid base-panel count")
    selected_ids = sorted(
        eligible,
        key=lambda base_id: hashlib.sha256(
            f"{base_id}|{SELECTION_SALT}".encode("utf-8")
        ).hexdigest(),
    )[:count]
    return [
        dict(row)
        for base_id in selected_ids
        for row in sorted(eligible[base_id], key=lambda value: int(value["label_rotation"]))
    ]


def build_jobs(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    jobs: list[dict[str, object]] = []
    for row in records:
        for context, target_field, feedback_field in (
            ("immediate", "new_target", "immediate_followup"),
            ("delayed_expression", "old_target", "delayed_expression_followup"),
        ):
            jobs.append({
                "id": str(row["id"]),
                "base_id": str(row["base_id"]),
                "rotation": int(row["label_rotation"]),
                "context": context,
                "target": str(row[target_field]),
                "text": hindsight_user_text(
                    str(row["prompt"]), str(row[feedback_field])
                ),
            })
    return jobs


def summarize_preflight(scores: Sequence[Mapping[str, object]]) -> dict[str, object]:
    cells: dict[str, dict[str, float | int]] = {}
    for context in CONTEXTS:
        for target in OPTION_LETTERS:
            selected = [
                row for row in scores
                if row["context"] == context and row["target"] == target
            ]
            if not selected:
                raise ValueError(f"missing cell: {context}/{target}")
            cells[f"{context}/{target}"] = {
                "n": len(selected),
                "correct": sum(bool(row["normalized_correct"]) for row in selected),
                "mean_normalized_target_probability": sum(
                    float(row["normalized_target_probability"]) for row in selected
                ) / len(selected),
                "mean_full_vocabulary_choice_mass": sum(
                    float(row["full_vocabulary_choice_mass"]) for row in selected
                ) / len(selected),
            }
    gates = {
        "every_context_label_cell_is_14_of_16_correct": all(
            value["n"] == BASE_COUNT and value["correct"] >= 14
            for value in cells.values()
        ),
        "every_context_label_cell_mean_target_probability_at_least_point_70": all(
            value["mean_normalized_target_probability"] >= 0.70
            for value in cells.values()
        ),
        "every_context_label_cell_mean_choice_mass_at_least_point_10": all(
            value["mean_full_vocabulary_choice_mass"] >= 0.10
            for value in cells.values()
        ),
    }
    qualified = all(gates.values())
    return {
        "decision": (
            "ENDO_PAHF_CAPABLE_INTERFACE_QUALIFIED"
            if qualified else "ENDO_PAHF_CAPABLE_INTERFACE_UNQUALIFIED"
        ),
        "qualified": qualified,
        "cells": cells,
        "gates": gates,
        "paper_green_light": False,
    }
