"""Prompts that differ only in final-outcome visibility."""

from __future__ import annotations

from .schema import TraceItem


def build_prompt(item: TraceItem, *, outcome_visible: bool, label_order: tuple[str, str]) -> str:
    """Construct a balanced forced-verdict prompt.

    The derivation is unchanged.  Only the dedicated final-answer field is
    replaced by a sentinel; this avoids pretending that internal derivation
    symbols have been hidden when they have not.
    """

    return build_prompt_from_fields(
        problem=item.problem,
        derivation=item.derivation,
        final_answer=item.final_answer,
        outcome_visible=outcome_visible,
        label_order=label_order,
    )


def build_prompt_from_fields(
    *,
    problem: str,
    derivation: str,
    final_answer: str,
    outcome_visible: bool,
    label_order: tuple[str, str],
) -> str:
    """Prompt builder for unlabelled runner records."""

    if set(label_order) != {"PROCESS_VALID", "PROCESS_INVALID"}:
        raise ValueError("label_order must contain each semantic verdict exactly once")
    outcome = final_answer if outcome_visible else "[OUTCOME WITHHELD FOR PROCESS REVIEW]"
    return (
        "You are auditing whether a proposed derivation is logically and mathematically valid. "
        "Judge the derivation itself, not whether its stated final answer is correct.\n\n"
        f"Problem:\n{problem}\n\n"
        f"Proposed derivation:\n{derivation}\n\n"
        f"Stated final answer:\n{outcome}\n\n"
        f"Reply with exactly one label: {label_order[0]} or {label_order[1]}."
    )
