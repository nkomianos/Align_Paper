"""Prepare observation-only EndoPAHF views for the official SLIFT baseline.

The SLIFT learner may see the original task, logged response, and immediate
feedback.  It must not see the expression/transition world label, delayed
probe, or old/new evaluation targets.  This module makes that boundary
executable and also emits outcome-blind FIX/SPEC role-sensitivity views.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

from interaction_sprint.hindsight_pahf_g2_v2 import (
    G2_LEARNING_BASES,
    build_g2_anchor_panels,
    build_g2_schedules,
)


SLIFT_RELEASE_URL = "https://anonymous.4open.science/api/repo/SLIFT/zip"
SLIFT_RELEASE_PAGE = "https://anonymous.4open.science/r/SLIFT"
SLIFT_RELEASE_LAST_UPDATE_UTC = "2026-07-31T07:56:51.365Z"
# The service regenerates ZIP metadata on each request, so the container bytes
# are not stable. Pin the canonical member-path/content tree instead.
SLIFT_RELEASE_TREE_SHA256 = (
    "dd6608719f7e46985e88c9b9e674bed076d1755b9090cff80bffca93ed87739e"
)

RAW_KEYS = {"id", "task", "logged_response", "feedback"}
PROCESSED_KEYS = {
    "id",
    "task",
    "logged_response",
    "components",
    "fix_components",
    "spec_components",
}
TASK_KEYS = {"id", "task"}
FORBIDDEN_ALGORITHM_KEYS = {
    "world",
    "old_target",
    "new_target",
    "expression_persistent_target",
    "transition_persistent_target",
    "delayed_expression_followup",
    "delayed_transition_followup",
    "partition",
    "source_index",
    "transition",
    "source_transition",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    lines = [
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in rows
    ]
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def _required_text(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a nonempty string")
    return value


def slift_raw_interaction(row: Mapping[str, object]) -> dict[str, object]:
    """Map one EndoPAHF record to SLIFT's released raw-input schema."""
    return {
        "id": _required_text(row, "id"),
        "task": _required_text(row, "prompt"),
        "logged_response": _required_text(row, "assistant_response"),
        "feedback": _required_text(row, "immediate_followup"),
    }


def slift_processed_interaction(
    row: Mapping[str, object], *, role: str,
) -> dict[str, object]:
    """Construct a role-sensitivity record using no persistent-state label."""
    if role not in {"FIX", "SPEC"}:
        raise ValueError("role sensitivity is restricted to FIX or SPEC")
    raw = slift_raw_interaction(row)
    feedback = str(raw["feedback"])
    return {
        "id": raw["id"],
        "task": raw["task"],
        "logged_response": raw["logged_response"],
        "components": [{"index": 0, "text": feedback, "role": role}],
        "fix_components": [feedback] if role == "FIX" else [],
        "spec_components": [feedback] if role == "SPEC" else [],
    }


def slift_inference_task(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "id": _required_text(row, "id"),
        "task": _required_text(row, "prompt"),
    }


def select_g2_global_rows(
    learning_rows: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    """Use exactly G2 v2's outcome-blind, one-rotation-per-base schedule."""
    panels = build_g2_anchor_panels(learning_rows)
    schedules = build_g2_schedules(learning_rows, panels)
    selected_ids = [
        str(row_id) for batch in schedules["global"] for row_id in batch
    ]
    by_id = {str(row["id"]): row for row in learning_rows}
    if len(by_id) != len(learning_rows) or set(selected_ids) - set(by_id):
        raise ValueError("learning rows do not match the frozen G2 schedule")
    selected = [by_id[row_id] for row_id in selected_ids]
    if len(selected) != G2_LEARNING_BASES:
        raise AssertionError("SLIFT view must contain one row per learning base")
    return selected


def _assert_exact_keys(rows: Sequence[Mapping[str, object]], allowed: set[str]) -> None:
    for row in rows:
        if set(row) != allowed:
            raise ValueError(f"algorithm-visible row keys differ: {sorted(row)}")
        if set(row) & FORBIDDEN_ALGORITHM_KEYS:
            raise ValueError("algorithm-visible row contains a forbidden target key")


def build_slift_g2b_payloads(
    learning_rows: Sequence[Mapping[str, object]],
    development_rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, bytes], dict[str, object]]:
    """Build deterministic raw, role-sensitivity, and DEV task-only files."""
    selected = select_g2_global_rows(learning_rows)
    raw = [slift_raw_interaction(row) for row in selected]
    all_fix = [slift_processed_interaction(row, role="FIX") for row in selected]
    all_spec = [slift_processed_interaction(row, role="SPEC") for row in selected]
    development = [slift_inference_task(row) for row in development_rows]
    _assert_exact_keys(raw, RAW_KEYS)
    _assert_exact_keys(all_fix, PROCESSED_KEYS)
    _assert_exact_keys(all_spec, PROCESSED_KEYS)
    _assert_exact_keys(development, TASK_KEYS)

    raw_bytes = _canonical_jsonl(raw)
    fix_bytes = _canonical_jsonl(all_fix)
    spec_bytes = _canonical_jsonl(all_spec)
    development_bytes = _canonical_jsonl(development)
    schedule = [
        {
            "base_id": _required_text(row, "base_id"),
            "id": _required_text(row, "id"),
            "label_rotation": int(row["label_rotation"]),
        }
        for row in selected
    ]
    schedule_bytes = (
        json.dumps(schedule, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    payloads = {
        # Deliberate duplicates make the observational-equivalence boundary
        # independently checksum-testable by downstream launchers.
        "learning_expression.jsonl": raw_bytes,
        "learning_transition.jsonl": raw_bytes,
        "learning_all_fix.jsonl": fix_bytes,
        "learning_all_spec.jsonl": spec_bytes,
        "development_tasks.jsonl": development_bytes,
        "learning_schedule.json": schedule_bytes,
    }
    base_counts = Counter(_required_text(row, "base_id") for row in selected)
    rotation_counts = Counter(int(row["label_rotation"]) for row in selected)
    metadata: dict[str, object] = {
        "decision": "ENDO_PAHF_SLIFT_G2B_INPUTS_PREPARED",
        "learning_rows": len(selected),
        "learning_unique_bases": len(base_counts),
        "maximum_rows_per_learning_base": max(base_counts.values()),
        "learning_rotation_counts": {
            str(key): rotation_counts[key] for key in sorted(rotation_counts)
        },
        "development_rows": len(development),
        "expression_transition_raw_sha256_equal": (
            sha256_bytes(payloads["learning_expression.jsonl"])
            == sha256_bytes(payloads["learning_transition.jsonl"])
        ),
        "confirmation_opened": False,
        "persistent_target_visible_to_algorithm": False,
        "role_sensitivity_uses_persistent_target": False,
        "paper_green_light": False,
    }
    if (
        metadata["learning_rows"] != 630
        or metadata["learning_unique_bases"] != 630
        or metadata["maximum_rows_per_learning_base"] != 1
        or metadata["development_rows"] != 384
        or max(rotation_counts.values()) - min(rotation_counts.values()) > 1
        or metadata["expression_transition_raw_sha256_equal"] is not True
    ):
        raise AssertionError(f"invalid SLIFT G2b preparation: {metadata}")
    return payloads, metadata
