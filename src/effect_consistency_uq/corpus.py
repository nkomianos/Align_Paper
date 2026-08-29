"""Deterministic public corpus and sequestered exact-effect answer key."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from under_extinction.io import read_jsonl, write_jsonl

from .environment import Action, execute_plan


@dataclass(frozen=True)
class EffectCase:
    task_id: str
    domain: str
    split: str
    interface: str
    stratum: str
    instruction: str
    initial_state: Mapping[str, Any]
    prompt: str


CALENDAR_TOOLS = {
    "canonical": ["check_calendar", "create_event", "move_event", "cancel_event"],
    "alias_rich": ["check_calendar", "list_events", "create_event", "book_slot", "schedule_meeting", "move_event", "reschedule", "cancel_event", "delete_event"],
}
INVENTORY_TOOLS = {
    "canonical": ["check_stock", "reserve_stock", "release_stock", "transfer_stock"],
    "alias_rich": ["check_stock", "list_reservations", "reserve_stock", "hold_stock", "release_stock", "unhold_stock", "transfer_stock", "move_stock"],
}


def _schema(domain: str, interface: str) -> str:
    names = CALENDAR_TOOLS[interface] if domain == "calendar" else INVENTORY_TOOLS[interface]
    return ", ".join(names)


def render_prompt(case: EffectCase) -> str:
    return (
        "You control a deterministic tool environment. Return only one JSON list with one to four calls. "
        "Each call has keys tool and arguments. Read-only checks are allowed. Do not explain.\n"
        f"DOMAIN: {case.domain}\nAVAILABLE TOOLS: {_schema(case.domain, case.interface)}\n"
        f"CURRENT STATE: {json.dumps(case.initial_state, sort_keys=True)}\n"
        f"USER GOAL: {case.instruction}\nPLAN:"
    )


def _calendar_case(index: int, rng: random.Random, split: str, interface: str) -> tuple[EffectCase, tuple[Action, ...]]:
    people = ["Ari", "Bo", "Chen", "Dara", "Eli"]
    person = people[index % len(people)]
    existing = {
        f"e{index}a": {"person": person, "start": 9, "duration": 1},
        f"e{index}b": {"person": people[(index + 1) % len(people)], "start": 13, "duration": 2},
    }
    operation = index % 3
    if operation == 0:
        event_id, start, duration = f"new{index}", 10 + (index % 2), 1
        instruction = f"Schedule event {event_id} with {person} at hour {start} for {duration} hour."
        oracle = (Action("create_event", {"event_id": event_id, "person": person, "start": start, "duration": duration}),)
    elif operation == 1:
        event_id, start = f"e{index}a", 15 + (index % 2)
        instruction = f"Move event {event_id} to hour {start}; keep its attendee and duration unchanged."
        oracle = (Action("move_event", {"event_id": event_id, "start": start}),)
    else:
        event_id = f"e{index}b"
        instruction = f"Cancel event {event_id}."
        oracle = (Action("cancel_event", {"event_id": event_id}),)
    state = {"events": existing, "timezone": "UTC"}
    stratum = "alias_equivalence" if interface == "alias_rich" else "argument_sensitive"
    case = EffectCase(f"cal-{index:04d}", "calendar", split, interface, stratum, instruction, state, "")
    return EffectCase(**{**asdict(case), "prompt": render_prompt(case)}), oracle


def _inventory_case(index: int, rng: random.Random, split: str, interface: str) -> tuple[EffectCase, tuple[Action, ...]]:
    sku_a, sku_b = f"SKU-{index % 17:02d}", f"SKU-{(index + 5) % 17:02d}"
    stock = {sku_a: 8 + index % 5, sku_b: 3 + index % 4}
    reservations = {f"old{index}": {"sku": sku_b, "quantity": 1}}
    operation = index % 3
    if operation == 0:
        rid, quantity = f"r{index}", 2 + index % 3
        instruction = f"Reserve {quantity} units of {sku_a} under reservation {rid}."
        oracle = (Action("reserve_stock", {"reservation_id": rid, "sku": sku_a, "quantity": quantity}),)
    elif operation == 1:
        rid = f"old{index}"
        instruction = f"Release reservation {rid} back into stock."
        oracle = (Action("release_stock", {"reservation_id": rid}),)
    else:
        quantity = 1 + index % 2
        instruction = f"Transfer {quantity} units from {sku_a} to {sku_b}."
        oracle = (Action("transfer_stock", {"source_sku": sku_a, "target_sku": sku_b, "quantity": quantity}),)
    state = {"stock": stock, "reservations": reservations, "warehouse": "west"}
    stratum = "alias_equivalence" if interface == "alias_rich" else "argument_sensitive"
    case = EffectCase(f"inv-{index:04d}", "inventory", split, interface, stratum, instruction, state, "")
    return EffectCase(**{**asdict(case), "prompt": render_prompt(case)}), oracle


def build_corpus(public_path: str | Path, answer_key_path: str | Path, *, count_per_domain: int = 160, seed: int = 20260829) -> tuple[EffectCase, ...]:
    """Write prompts and labels separately; refuse to overwrite either file."""

    if count_per_domain < 40 or count_per_domain % 4:
        raise ValueError("count_per_domain must be a multiple of four and at least 40")
    public, key = Path(public_path), Path(answer_key_path)
    if public.exists() or key.exists():
        raise FileExistsError("refusing to overwrite a frozen corpus or answer key")
    rng = random.Random(seed)
    cases: list[EffectCase] = []
    labels: list[dict[str, Any]] = []
    for domain, builder in (("calendar", _calendar_case), ("inventory", _inventory_case)):
        for index in range(count_per_domain):
            split = "DEV" if index < count_per_domain // 4 else "TEST"
            interface = "alias_rich" if index % 2 else "canonical"
            case, oracle = builder(index, rng, split, interface)
            execution = execute_plan(domain, case.initial_state, oracle)
            if not execution.valid:
                raise RuntimeError(f"generated invalid oracle for {case.task_id}: {execution.error}")
            cases.append(case)
            labels.append({"task_id": case.task_id, "oracle_effect": execution.effect})
    write_jsonl(public, (asdict(case) for case in cases))
    write_jsonl(key, labels)
    return tuple(cases)


def load_cases(path: str | Path) -> tuple[EffectCase, ...]:
    cases = tuple(EffectCase(**row) for row in read_jsonl(path))
    ids = [case.task_id for case in cases]
    if not cases or len(ids) != len(set(ids)):
        raise ValueError("effect-consistency corpus must be non-empty with unique IDs")
    return cases
