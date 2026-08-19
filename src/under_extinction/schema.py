"""Shared schema and invariants for generated records."""

from __future__ import annotations

from enum import Enum
from typing import Any, Iterable


SCHEMA_VERSION = "1.0"
ACTIONS = ("A", "B")


class Controller(str, Enum):
    INTENDED = "intended"
    PROXY = "proxy"
    CACHED = "cached"


class Intervention(str, Enum):
    BASELINE = "baseline"
    GENUINE_VALUE = "genuine_value"
    PROXY_VALUE = "proxy_value"
    GENUINE_CONTINGENCY = "genuine_contingency"
    PROXY_CONTINGENCY = "proxy_contingency"
    CUE_SWAP = "cue_swap"


REAL_INTERVENTIONS = (
    Intervention.GENUINE_VALUE,
    Intervention.PROXY_VALUE,
    Intervention.GENUINE_CONTINGENCY,
    Intervention.PROXY_CONTINGENCY,
    Intervention.CUE_SWAP,
)


def other_action(action: str) -> str:
    if action not in ACTIONS:
        raise ValueError(f"Unknown action {action!r}; expected one of {ACTIONS}")
    return ACTIONS[1] if action == ACTIONS[0] else ACTIONS[0]


def argmax_action(scores: dict[str, float]) -> str:
    """Return the unique maximizing action; ties are invalid by design."""
    if set(scores) != set(ACTIONS):
        raise ValueError(f"Scores must contain exactly {ACTIONS}: {scores}")
    if scores[ACTIONS[0]] == scores[ACTIONS[1]]:
        raise ValueError(f"Tied action scores violate the task invariant: {scores}")
    return max(ACTIONS, key=scores.__getitem__)


def action_utilities(record: dict[str, Any], channel: str, phase: str = "post") -> dict[str, float]:
    if channel not in {"genuine", "proxy"}:
        raise ValueError(f"Unknown outcome channel: {channel}")
    world = record["world"]
    mapping = world[f"{channel}_by_action_{phase}"]
    values = world[f"{channel}_values_{phase}"]
    return {action: float(values[mapping[action]]) for action in ACTIONS}


def controller_actions(record: dict[str, Any]) -> dict[str, str]:
    world = record["world"]
    return {
        Controller.INTENDED.value: argmax_action(action_utilities(record, "genuine")),
        Controller.PROXY.value: argmax_action(action_utilities(record, "proxy")),
        Controller.CACHED.value: world["cue_action_post"],
    }


def validate_record(record: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "record_id",
        "split",
        "renderer_id",
        "world_id",
        "task_type",
        "condition",
        "messages",
        "oracle_actions",
        "world",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"Record {record.get('record_id', '<unknown>')} missing {sorted(missing)}")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema version: {record['schema_version']}")
    if not record["messages"] or record["messages"][-1].get("role") != "user":
        raise ValueError("Messages must end with a user choice request")
    actions = record["oracle_actions"]
    if set(actions) != {c.value for c in Controller}:
        raise ValueError(f"Oracle actions are incomplete: {actions}")
    if not set(actions.values()) <= set(ACTIONS):
        raise ValueError(f"Invalid oracle action in {actions}")
    if record["task_type"] in {"static_decision", "audit_decision"}:
        derived = controller_actions(record)
        if actions != derived:
            raise ValueError(f"Stored oracle actions {actions} disagree with derived {derived}")


def validate_unique(records: Iterable[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for record in records:
        validate_record(record)
        record_id = record["record_id"]
        if record_id in seen:
            raise ValueError(f"Duplicate record_id: {record_id}")
        seen.add(record_id)
