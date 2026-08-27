"""Strict wire format for paired outcome-visibility experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


class Verdict(str, Enum):
    VALID = "PROCESS_VALID"
    INVALID = "PROCESS_INVALID"
    UNPARSEABLE = "UNPARSEABLE"


@dataclass(frozen=True)
class TraceItem:
    """An item whose process label is never passed to the model runner."""

    item_id: str
    problem: str
    derivation: str
    final_answer: str
    process_valid: bool
    split: str

    def __post_init__(self) -> None:
        if not self.item_id or not self.problem or not self.derivation or not self.final_answer:
            raise ValueError("item_id, problem, derivation, and final_answer are required")
        if self.split not in {"development", "test"}:
            raise ValueError("split must be development or test")

    def runner_record(self) -> dict[str, str]:
        """The unlabelled record allowed into an inference job."""

        return {
            "item_id": self.item_id,
            "problem": self.problem,
            "derivation": self.derivation,
            "final_answer": self.final_answer,
            "split": self.split,
        }

    def answer_key_record(self) -> dict[str, object]:
        return {"item_id": self.item_id, "process_valid": self.process_valid, "split": self.split}

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
