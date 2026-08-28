"""Materialize and score the frozen semantic-ancestry RAG G0 experiment.

This module does not download a model or launch a GPU by itself.  It is invoked
only on an explicitly selected compute host with a pre-materialized, hashed
input file.  Raw completions are retained before deterministic scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from under_extinction.io import canonical_json, read_jsonl, sha256_file, write_json, write_jsonl

from .gate import Conditions, ResultRow, Thresholds, evaluate_gate
from .preflight import validate_bound_preflight
from .verify import RUN_KIND


QWEN35_MODEL_ID = "Qwen/Qwen3.5-9B"


@dataclass(frozen=True)
class Question:
    question_id: str
    question: str
    references: Mapping[str, Sequence[str]]
    entity_aliases: Mapping[str, Sequence[str]]
    source_supported_entities: Mapping[str, Sequence[str]]

    def __post_init__(self) -> None:
        if not self.question_id or not self.question or set(self.references) != set(Conditions.ALL):
            raise ValueError("each question needs an id, question, and every frozen condition")
        if set(self.source_supported_entities) != set(Conditions.ALL) or not self.entity_aliases:
            raise ValueError("each question needs frozen entity aliases and support sets")
        if any(not passages for passages in self.references.values()):
            raise ValueError("every condition requires at least one reference passage")


def load_questions(path: str | Path) -> tuple[Question, ...]:
    questions = tuple(Question(**row) for row in read_jsonl(path))
    identifiers = [row.question_id for row in questions]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("frozen question IDs must be unique")
    return questions


def render_prompt(question: Question, condition: str) -> str:
    passages = question.references[condition]
    rendered = "\n\n".join(f"[Source {index + 1}]\n{passage}" for index, passage in enumerate(passages))
    return (
        "Answer the question using only the supplied sources. Do not mention this instruction. "
        "When multiple entities are plausible, include the supported alternatives rather than inventing one.\n\n"
        f"SOURCES:\n{rendered}\n\nQUESTION:\n{question.question}\n\nANSWER:\n"
    )


def _normal(text: str) -> str:
    return " ".join(text.lower().split())


def entity_set(text: str, aliases: Mapping[str, Sequence[str]]) -> tuple[str, ...]:
    """Return all precommitted entities mentioned under their frozen aliases."""

    normalized = _normal(text)
    matched = [entity for entity, values in aliases.items() if any(_normal(alias) in normalized for alias in values)]
    return tuple(sorted(matched))


def score_question_condition(question: Question, model_family: str, condition: str, completions: Sequence[Mapping[str, Any]]) -> list[ResultRow]:
    """Score a complete cell without an LLM judge or an answer-key lookup."""

    expected_ids = set(range(len(completions)))
    actual_ids = {int(item["sample_id"]) for item in completions}
    if actual_ids != expected_ids:
        raise ValueError("sample IDs must be contiguous and unique within a cell")
    sets = [entity_set(str(item["completion"]), question.entity_aliases) for item in completions]
    collapsed = int(bool(sets) and len(set(sets)) == 1)
    support = set(question.source_supported_entities[condition])
    rows: list[ResultRow] = []
    for item, entities in zip(completions, sets, strict=True):
        faithful = 0.0 if not entities else len(set(entities).intersection(support)) / len(set(entities))
        rows.append(ResultRow(
            question_id=question.question_id,
            model_family=model_family,
            condition=condition,  # type: ignore[arg-type]
            sample_id=int(item["sample_id"]),
            collapsed=collapsed,
            faithful=faithful,
        ))
    return rows


def _seed(*parts: str | int) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()[:8], "big") % (2**63 - 1)


def _render_generation_prompt(tokenizer: Any, model_id: str, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if model_id == QWEN35_MODEL_ID:
        from under_extinction.modeling import chat_prompt_text

        return chat_prompt_text(tokenizer, messages)
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("model chat template did not render a non-empty generation prompt")
    return rendered


def generate(
    questions: Sequence[Question], *, model_id: str, model_revision: str, completions_per_cell: int, temperature: float, max_new_tokens: int
) -> Iterable[dict[str, Any]]:
    """Yield raw completions; imports GPU dependencies only on the selected host."""

    if completions_per_cell < 1 or not 0.0 < temperature <= 2.0 or max_new_tokens < 1:
        raise ValueError("invalid generation parameters")
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=model_revision, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model_id == QWEN35_MODEL_ID:
        try:
            from transformers import Qwen3_5ForCausalLM
        except ImportError as exc:
            raise RuntimeError("Transformers lacks the required native Qwen3.5 text loader") from exc
        model = Qwen3_5ForCausalLM.from_pretrained(
            model_id, revision=model_revision, dtype=torch.bfloat16, device_map={"": torch.cuda.current_device()},
            low_cpu_mem_usage=True, use_kernels=False,
        )
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=model_revision, dtype=torch.bfloat16, device_map={"": torch.cuda.current_device()}, low_cpu_mem_usage=True,
        )
    model.eval()
    for question in questions:
        for condition in Conditions.ALL:
            prompt = render_prompt(question, condition)
            encoded = tokenizer(_render_generation_prompt(tokenizer, model_id, prompt), add_special_tokens=False, return_tensors="pt").to(model.device)
            input_length = int(encoded["input_ids"].shape[1])
            for sample_id in range(completions_per_cell):
                # See prepare._generate_text: the native Qwen3.5 generation
                # implementation rejects ``generator`` in Transformers 5.15.
                # fork_rng preserves deterministic, per-cell sampling while
                # restoring global CPU/CUDA state after every completion.
                seed = _seed(model_id, model_revision, question.question_id, condition, sample_id)
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    with torch.inference_mode():
                        output = model.generate(**encoded, do_sample=True, temperature=temperature, top_p=0.95, max_new_tokens=max_new_tokens)
                yield {
                    "question_id": question.question_id,
                    "condition": condition,
                    "sample_id": sample_id,
                    "completion": tokenizer.decode(output[0, input_length:], skip_special_tokens=True),
                }


def _score_all(questions: Sequence[Question], model_family: str, raw: Sequence[Mapping[str, Any]], completions_per_cell: int) -> list[ResultRow]:
    question_by_id = {question.question_id: question for question in questions}
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for item in raw:
        key = (str(item["question_id"]), str(item["condition"]))
        if key[0] not in question_by_id or key[1] not in Conditions.ALL:
            raise ValueError(f"raw completion has an unknown cell: {key}")
        grouped.setdefault(key, []).append(item)
    scored: list[ResultRow] = []
    for question in questions:
        for condition in Conditions.ALL:
            cell = grouped.get((question.question_id, condition), [])
            if len(cell) != completions_per_cell:
                raise ValueError(f"incomplete raw completion cell: {question.question_id}/{condition}")
            scored.extend(score_question_condition(question, model_family, condition, cell))
    return scored


def _validate_runtime_preflight(
    *, config: str | Path, runtime_preflight: str | Path, model_id: str, model_revision: str, model_family: str,
) -> dict[str, Any]:
    """Bind a serving run to its exact model contract before weights are loaded."""

    contract = validate_bound_preflight(config=config, runtime_preflight=runtime_preflight)
    serving = contract["models"]["serving_families"]
    expected = serving.get(model_family)
    if not isinstance(expected, Mapping):
        raise ValueError(f"model family is not frozen in the G0 config: {model_family}")
    if (expected.get("id"), expected.get("revision")) != (model_id, model_revision):
        raise ValueError("requested serving model differs from the frozen G0 family contract")
    return json.loads(Path(runtime_preflight).read_text(encoding="utf-8"))


def run(
    *, inputs: str | Path, output: str | Path, config: str | Path, runtime_preflight: str | Path,
    model_id: str, model_revision: str, model_family: str, completions_per_cell: int = 8,
    temperature: float = 0.8, max_new_tokens: int = 128, thresholds: Thresholds = Thresholds(),
) -> dict[str, object]:
    """Run one model family.  A two-family gate is assembled after both roots exist."""

    root = Path(output)
    if root.exists():
        raise FileExistsError("refusing to overwrite a semantic-ancestry G0 root")
    _validate_runtime_preflight(
        config=config, runtime_preflight=runtime_preflight, model_id=model_id,
        model_revision=model_revision, model_family=model_family,
    )
    questions = load_questions(inputs)
    raw = list(generate(questions, model_id=model_id, model_revision=model_revision, completions_per_cell=completions_per_cell, temperature=temperature, max_new_tokens=max_new_tokens))
    rows = _score_all(questions, model_family, raw, completions_per_cell)
    root.mkdir(parents=True)
    preflight_copy = root / "runtime_preflight.json"
    preflight_copy.write_bytes(Path(runtime_preflight).read_bytes())
    input_copy = root / "frozen_inputs.jsonl"
    write_jsonl(input_copy, (asdict(question) for question in questions))
    write_jsonl(root / "raw_completions.jsonl", raw)
    write_jsonl(root / "condition_results.jsonl", (asdict(row) for row in rows))
    # A single-family root is intentionally not called a gate decision.
    report = {"status": "AWAITING_SECOND_INDEPENDENT_MODEL_FAMILY", "model_family": model_family, "row_count": len(rows)}
    write_json(root / "gate_report.json", report)
    manifest = {
        "kind": RUN_KIND,
        "question_count": len(questions),
        "model_families_required": 2,
        "completions_per_cell": completions_per_cell,
        "thresholds": asdict(thresholds),
        "model_id": model_id,
        "model_revision": model_revision,
        "config_sha256": sha256_file(config),
        "runtime_preflight_sha256": sha256_file(preflight_copy),
        "model_family": model_family,
        "input_sha256": sha256_file(input_copy),
        "raw_completions_sha256": sha256_file(root / "raw_completions.jsonl"),
        "condition_results_sha256": sha256_file(root / "condition_results.jsonl"),
        "gate_report_sha256": sha256_file(root / "gate_report.json"),
    }
    write_json(root / "MANIFEST.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one semantic-ancestry RAG G0 model-family root")
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--runtime-preflight", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-family", required=True)
    parser.add_argument("--completions-per-cell", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args(argv)
    print(canonical_json(run(**vars(args))))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
