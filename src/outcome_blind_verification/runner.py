"""Inference and private-key analysis for the outcome-blinding G0 experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

from .analysis import GateReport, assess_gate
from .prompts import build_prompt_from_fields
from .schema import Verdict


def _read_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON on line {line_number} of {path}") from error


def _label_order(item_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(item_id.encode("utf-8")).digest()
    return ("PROCESS_VALID", "PROCESS_INVALID") if digest[0] % 2 == 0 else ("PROCESS_INVALID", "PROCESS_VALID")


def parse_verdict(text: str) -> Verdict:
    """Accept only an exact verdict token; prose is an auditable format failure."""

    normalized = text.strip()
    if normalized == Verdict.VALID.value:
        return Verdict.VALID
    if normalized == Verdict.INVALID.value:
        return Verdict.INVALID
    return Verdict.UNPARSEABLE


def _validate_runner_record(record: Mapping[str, object]) -> dict[str, str]:
    forbidden = {"process_valid", "label", "gold", "target"} & set(record)
    if forbidden:
        raise ValueError(f"runner input leaks a process label: {sorted(forbidden)}")
    required = {"item_id", "problem", "derivation", "final_answer", "split"}
    if set(record) != required:
        raise ValueError(f"runner record keys must be exactly {sorted(required)}")
    result = {key: str(record[key]) for key in required}
    if result["split"] not in {"development", "test"}:
        raise ValueError("runner split must be development or test")
    return result


def run_inference(
    *,
    runner_data: Path,
    output: Path,
    model_name: str,
    max_new_tokens: int = 8,
) -> None:
    """Run both paired arms without opening the private process labels."""

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:  # pragma: no cover - exercised on the GPU host
        raise RuntimeError("install the project's gpu optional dependencies before inference") from error
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
    model.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for raw_record in _read_jsonl(runner_data):
            record = _validate_runner_record(raw_record)
            row: dict[str, object] = {"item_id": record["item_id"], "model": model_name, "arms": {}}
            order = _label_order(record["item_id"])
            for arm, visible in (("visible", True), ("blind", False)):
                prompt = build_prompt_from_fields(
                    problem=record["problem"],
                    derivation=record["derivation"],
                    final_answer=record["final_answer"],
                    outcome_visible=visible,
                    label_order=order,
                )
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.inference_mode():
                    generated = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
                completion = tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                row["arms"][arm] = {
                    "label_order": list(order),
                    "completion": completion,
                    "verdict": parse_verdict(completion).value,
                }
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def analyze_responses(*, answer_key: Path, responses: Path, split: str = "test") -> GateReport:
    """Join raw responses to labels locally, after inference is complete."""

    if split not in {"development", "test"}:
        raise ValueError("analysis split must be development or test")
    key = {
        str(row["item_id"]): bool(row["process_valid"])
        for row in _read_jsonl(answer_key)
        if row.get("split") == split
    }
    visible: dict[str, Verdict] = {}
    blind: dict[str, Verdict] = {}
    for row in _read_jsonl(responses):
        item_id = str(row["item_id"])
        if item_id not in key:
            continue
        arms = row.get("arms")
        if not isinstance(arms, dict) or set(arms) != {"visible", "blind"}:
            raise ValueError(f"response row {item_id} lacks paired arms")
        visible[item_id] = Verdict(str(arms["visible"]["verdict"]))
        blind[item_id] = Verdict(str(arms["blind"]["verdict"]))
    return assess_gate(key, visible, blind)
