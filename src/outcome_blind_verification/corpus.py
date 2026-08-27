"""Generate a deterministic, answer-separated arithmetic process-audit corpus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .schema import TraceItem


def build_g0_items(*, seed: int = 260826, pairs: int = 240) -> tuple[TraceItem, ...]:
    """Create matched valid/invalid derivations with the same correct outcome.

    The final numeric outcome is absent from both derivation variants and exists
    only in the dedicated final-answer field.  Invalid variants introduce an
    unjustified premise that algebraically cancels, preserving the outcome.
    """

    if pairs < 20 or pairs % 4:
        raise ValueError("pairs must be a multiple of four and at least 20")
    rng = np.random.default_rng(seed)
    items: list[TraceItem] = []
    names = ("marbles", "seeds", "cards", "stickers")
    for index in range(pairs):
        initial = int(rng.integers(35, 160))
        removed = int(rng.integers(4, min(30, initial - 4)))
        answer = initial - removed
        noun = names[index % len(names)]
        split = "test" if index % 5 == 0 else "development"
        problem = f"A box initially contains {initial} {noun}. Exactly {removed} are removed. How many remain?"
        valid = (
            f"Let T denote the number remaining. Since {removed} were removed from {initial}, "
            f"T + {removed} = {initial}. Subtracting {removed} from both sides gives T = {initial} - {removed}."
        )
        mode = index % 4
        if mode == 0:
            invalid = (
                f"Let T denote the number remaining. Since {removed} were removed from {initial}, "
                f"assume T + {removed + 1} = {initial + 1}. Subtracting {removed + 1} from both sides gives T = {initial + 1} - {removed + 1}."
            )
        elif mode == 1:
            invalid = (
                f"Let T denote the number remaining. The box started with {initial}. "
                f"Without deriving a removal equation, set T = {initial} - {removed}."
            )
        elif mode == 2:
            invalid = (
                f"Let T denote the number remaining. First write T + {removed} = {initial}. "
                f"Because {removed + 2} were removed, subtract {removed + 2} from both sides and set T = {initial} - {removed}."
            )
        else:
            invalid = (
                f"Let T denote the number remaining. Since {removed} were removed from {initial}, "
                f"T + {removed} = {initial}. Add an unsupported correction of 1 to both sides, then conclude T = {initial} - {removed}."
            )
        answer_text = f"T = {answer}"
        prefix = f"g0-{index:04d}"
        items.extend((
            TraceItem(f"{prefix}-valid", problem, valid, answer_text, True, split),
            TraceItem(f"{prefix}-invalid", problem, invalid, answer_text, False, split),
        ))
    return tuple(items)


def write_g0_corpus(*, destination: Path, seed: int = 260826, pairs: int = 240) -> dict[str, str]:
    """Write public runner input and a separate local-only answer key."""

    destination.mkdir(parents=True, exist_ok=False)
    items = build_g0_items(seed=seed, pairs=pairs)
    runner_data = destination / "runner_data.jsonl"
    answer_key = destination / "private_answer_key.jsonl"
    for path, records in (
        (runner_data, (item.runner_record() for item in items)),
        (answer_key, (item.answer_key_record() for item in items)),
    ):
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    manifest = {
        "schema_version": "1.0",
        "kind": "outcome_blind_process_verification_g0",
        "seed": str(seed),
        "pairs": str(pairs),
        "runner_data_sha256": hashlib.sha256(runner_data.read_bytes()).hexdigest(),
        "private_answer_key_sha256": hashlib.sha256(answer_key.read_bytes()).hexdigest(),
    }
    (destination / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
