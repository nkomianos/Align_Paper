"""Two exact, resettable tool worlds and a conservative JSON plan parser.

The scientific object is the canonical state delta, not an embedding and not an
LLM judgment.  Read-only calls, aliases, and redundant successful calls can
therefore map different action strings to exactly the same environment effect.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import re
from typing import Any, Mapping, Sequence


ALIASES = {
    "book_slot": "create_event",
    "schedule_meeting": "create_event",
    "reschedule": "move_event",
    "delete_event": "cancel_event",
    "hold_stock": "reserve_stock",
    "unhold_stock": "release_stock",
    "move_stock": "transfer_stock",
}
READ_ONLY = {"check_calendar", "list_events", "check_stock", "list_reservations"}


@dataclass(frozen=True)
class Action:
    tool: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class Execution:
    effect: str
    final_state: Mapping[str, Any]
    valid: bool
    error: str | None = None


def _json_object(text: str) -> Any:
    """Extract the first balanced JSON list/object without accepting prose."""

    decoder = json.JSONDecoder()
    starts = [index for index, char in enumerate(text) if char in ("[", "{")]
    for start in starts:
        try:
            value, _ = decoder.raw_decode(text[start:])
            return value
        except json.JSONDecodeError:
            continue
    raise ValueError("no JSON plan found")


def parse_plan(text: str) -> tuple[Action, ...]:
    """Parse a plan of at most four tool calls.

    Markdown fences are tolerated, but a natural-language reconstruction is
    deliberately not.  Invalid generations become an explicit invalid effect.
    """

    value = _json_object(re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE))
    if isinstance(value, Mapping) and "actions" in value:
        value = value["actions"]
    if isinstance(value, Mapping):
        value = [value]
    if not isinstance(value, list) or not 1 <= len(value) <= 4:
        raise ValueError("plan must be a JSON list containing one to four actions")
    actions: list[Action] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("each action must be a JSON object")
        tool = item.get("tool", item.get("name"))
        arguments = item.get("arguments", item.get("args", {}))
        if not isinstance(tool, str) or not tool or not isinstance(arguments, Mapping):
            raise ValueError("each action needs a tool string and argument object")
        actions.append(Action(tool=tool.strip(), arguments=dict(arguments)))
    return tuple(actions)


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    return value


def _effect(before: Mapping[str, Any], after: Mapping[str, Any]) -> str:
    changes: dict[str, Any] = {}
    keys = sorted(set(before) | set(after))
    for key in keys:
        left, right = before.get(key), after.get(key)
        if _canonical(left) != _canonical(right):
            changes[key] = {"before": _canonical(left), "after": _canonical(right)}
    return json.dumps(changes, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _integer(arguments: Mapping[str, Any], key: str) -> int:
    value = arguments.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _string(arguments: Mapping[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _calendar_step(state: dict[str, Any], action: Action) -> None:
    tool = ALIASES.get(action.tool, action.tool)
    if tool in READ_ONLY:
        return
    events = state.setdefault("events", {})
    if tool == "create_event":
        event_id = _string(action.arguments, "event_id")
        start = _integer(action.arguments, "start")
        duration = _integer(action.arguments, "duration")
        person = _string(action.arguments, "person")
        if event_id in events or duration <= 0:
            raise ValueError("event already exists or duration is invalid")
        for event in events.values():
            if event["person"] == person and max(start, event["start"]) < min(start + duration, event["start"] + event["duration"]):
                raise ValueError("calendar conflict")
        events[event_id] = {"person": person, "start": start, "duration": duration}
        return
    event_id = _string(action.arguments, "event_id")
    if event_id not in events:
        raise ValueError("unknown event")
    if tool == "move_event":
        start = _integer(action.arguments, "start")
        candidate = dict(events[event_id])
        candidate["start"] = start
        for other_id, event in events.items():
            if other_id != event_id and event["person"] == candidate["person"] and max(start, event["start"]) < min(start + candidate["duration"], event["start"] + event["duration"]):
                raise ValueError("calendar conflict")
        events[event_id] = candidate
        return
    if tool == "cancel_event":
        del events[event_id]
        return
    raise ValueError(f"unknown calendar tool: {action.tool}")


def _inventory_step(state: dict[str, Any], action: Action) -> None:
    tool = ALIASES.get(action.tool, action.tool)
    if tool in READ_ONLY:
        return
    stock = state.setdefault("stock", {})
    reservations = state.setdefault("reservations", {})
    if tool == "reserve_stock":
        reservation_id = _string(action.arguments, "reservation_id")
        sku = _string(action.arguments, "sku")
        quantity = _integer(action.arguments, "quantity")
        if reservation_id in reservations or quantity <= 0 or stock.get(sku, 0) < quantity:
            raise ValueError("invalid reservation")
        stock[sku] -= quantity
        reservations[reservation_id] = {"sku": sku, "quantity": quantity}
        return
    if tool == "release_stock":
        reservation_id = _string(action.arguments, "reservation_id")
        if reservation_id not in reservations:
            raise ValueError("unknown reservation")
        reservation = reservations.pop(reservation_id)
        stock[reservation["sku"]] = stock.get(reservation["sku"], 0) + reservation["quantity"]
        return
    if tool == "transfer_stock":
        source = _string(action.arguments, "source_sku")
        target = _string(action.arguments, "target_sku")
        quantity = _integer(action.arguments, "quantity")
        if source == target or quantity <= 0 or stock.get(source, 0) < quantity:
            raise ValueError("invalid transfer")
        stock[source] -= quantity
        stock[target] = stock.get(target, 0) + quantity
        return
    raise ValueError(f"unknown inventory tool: {action.tool}")


def execute_plan(domain: str, initial_state: Mapping[str, Any], plan: Sequence[Action]) -> Execution:
    before = deepcopy(dict(initial_state))
    state = deepcopy(before)
    try:
        if not plan:
            raise ValueError("empty plan")
        for action in plan:
            if domain == "calendar":
                _calendar_step(state, action)
            elif domain == "inventory":
                _inventory_step(state, action)
            else:
                raise ValueError(f"unknown domain: {domain}")
    except (KeyError, TypeError, ValueError) as exc:
        return Execution(effect=f"INVALID:{type(exc).__name__}:{exc}", final_state=before, valid=False, error=str(exc))
    return Execution(effect=_effect(before, state), final_state=_canonical(state), valid=True)


def action_signature(plan: Sequence[Action], *, canonical_tools: bool = False) -> str:
    rows = []
    for action in plan:
        tool = ALIASES.get(action.tool, action.tool) if canonical_tools else action.tool
        rows.append({"tool": tool, "arguments": _canonical(action.arguments)})
    return json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
