"""Deterministic public-prompt ingestion for SENTRY source generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from under_extinction.io import canonical_json


@dataclass(frozen=True)
class Prompt:
    """A public, trait-free source prompt with a content-addressed identifier."""

    prompt_id: str
    question: str
    source: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _identifier(question: str, source: str) -> str:
    return hashlib.sha256(canonical_json({"question": question, "source": source}).encode("utf-8")).hexdigest()


def load_public_prompts(path: str | Path, *, source: str, limit: int) -> tuple[Prompt, ...]:
    """Read the upstream ``question`` schema and fail on duplicate/blank rows."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    prompts: list[Prompt] = []
    seen: set[str] = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("question"), str):
            raise ValueError(f"missing string question at line {line_number}")
        question = row["question"].strip()
        if not question:
            raise ValueError(f"blank question at line {line_number}")
        identifier = _identifier(question, source)
        if identifier in seen:
            raise ValueError("duplicate public prompt")
        seen.add(identifier)
        prompts.append(Prompt(identifier, question, source))
        if len(prompts) == limit:
            break
    if len(prompts) != limit:
        raise ValueError("public source did not contain the requested prompt budget")
    return tuple(prompts)


def disjoint_assignment(prompts: Iterable[Prompt], *, seed: int, calibration_fraction: float = 0.5) -> dict[str, str]:
    """Assign prompt IDs before source generation; no row may cross partitions."""

    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be strictly between zero and one")
    assignments: dict[str, str] = {}
    for prompt in prompts:
        draw = int.from_bytes(hashlib.sha256(f"{seed}|{prompt.prompt_id}".encode()).digest()[:8], "big") / 2**64
        assignments[prompt.prompt_id] = "calibration" if draw < calibration_fraction else "sealed"
    if len(set(assignments.values())) != 2:
        raise ValueError("split seed produced an empty partition")
    return assignments
