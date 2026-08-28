"""Prepare one shared, frozen answer-history corpus before serving-model G0."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from under_extinction.io import canonical_json, read_jsonl, sha256_file, write_json, write_jsonl

from .corpus import BaseQuestion
from .preflight import validate_bound_preflight
from .retrieval import history_aware_select, mmr_select
from .runner import Question


QWEN35_MODEL_ID = "Qwen/Qwen3.5-9B"


def _load_base(path: str | Path) -> tuple[BaseQuestion, ...]:
    records = tuple(BaseQuestion(**row) for row in read_jsonl(path))
    if not records or len({record.question_id for record in records}) != len(records):
        raise ValueError("base packets must be non-empty with unique question IDs")
    return records


def _seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()[:8], "big") % (2**63 - 1)


def _render_generation_prompt(tokenizer: Any, model_id: str, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if model_id == QWEN35_MODEL_ID:
        # Qwen3.5 is a hybrid multimodal release, but this gate deliberately
        # uses its native text-only loader and disables its optional reasoning
        # mode.  The shared helper attests the exact chat-template behavior.
        from under_extinction.modeling import chat_prompt_text

        return chat_prompt_text(tokenizer, messages)
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("model chat template did not render a non-empty generation prompt")
    return rendered


def _generate_text(model: Any, tokenizer: Any, model_id: str, prompt: str, *, seed: int, max_new_tokens: int = 128) -> str:
    import torch

    encoded = tokenizer(_render_generation_prompt(tokenizer, model_id, prompt), add_special_tokens=False, return_tensors="pt").to(model.device)
    input_length = int(encoded["input_ids"].shape[1])
    generator = torch.Generator(device=model.device).manual_seed(seed)
    with torch.inference_mode():
        output = model.generate(**encoded, do_sample=True, temperature=0.8, top_p=0.95, max_new_tokens=max_new_tokens, generator=generator)
    return tokenizer.decode(output[0, input_length:], skip_special_tokens=True).strip()


def _load_model(model_id: str, revision: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model_id == QWEN35_MODEL_ID:
        try:
            from transformers import Qwen3_5ForCausalLM
        except ImportError as exc:
            raise RuntimeError("Transformers lacks the required native Qwen3.5 text loader") from exc
        model = Qwen3_5ForCausalLM.from_pretrained(
            model_id, revision=revision, dtype=torch.bfloat16, device_map={"": torch.cuda.current_device()},
            low_cpu_mem_usage=True, use_kernels=False,
        )
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, dtype=torch.bfloat16, device_map={"": torch.cuda.current_device()}, low_cpu_mem_usage=True,
        )
    model.eval()
    return model, tokenizer


def _base_prompt(question: BaseQuestion) -> str:
    sources = "\n\n".join(f"[Source {index + 1}]\n{value}" for index, value in enumerate(question.base_references))
    return f"Use only these sources to answer briefly.\n\n{sources}\n\nQUESTION:\n{question.question}\n\nANSWER:\n"


def _rewrite_prompt(text: str) -> str:
    return (
        "Rewrite the following response as a neutral source passage. Preserve every named entity and claim; "
        "do not add any facts, entities, rankings, or citations. Return only the passage.\n\nRESPONSE:\n" + text
    )


def _independent_summary_prompt(question: BaseQuestion) -> str:
    sources = "\n\n".join(f"[Source {index + 1}]\n{value}" for index, value in enumerate(question.base_references))
    return (
        "Write one neutral source passage from the source packets below. Preserve named entities and factual claims, "
        "but do not add rankings, citations, or facts. You have not seen any prior answer. Return only the passage.\n\n"
        f"SOURCE PACKETS:\n{sources}"
    )


def materialize_question(base: BaseQuestion, *, ancestor_answer: str, cross_rewrite: str, style_only: str, independent_summary: str) -> Question:
    """Build every condition from frozen transformations, without author metadata."""

    pool = (*base.base_references, cross_rewrite)
    mmr = mmr_select(base.question, pool, limit=len(base.base_references))
    history = history_aware_select(base.question, pool, (ancestor_answer,), limit=len(base.base_references))
    references = {
        "baseline": base.base_references,
        "self_ancestor": (*base.base_references, ancestor_answer),
        "cross_ancestor": (*base.base_references, cross_rewrite),
        "style_only": (*base.base_references, style_only),
        "independent_summary": (*base.base_references, independent_summary),
        "mmr": tuple(pool[index] for index in mmr.indices),
        "history_aware": tuple(pool[index] for index in history.indices),
    }
    support = {
        condition: tuple(
            entity for entity, aliases in base.entity_aliases.items()
            if any(alias.lower() in "\n".join(passages).lower() for alias in aliases)
        )
        for condition, passages in references.items()
    }
    return Question(
        question_id=base.question_id,
        question=base.question,
        references=references,
        entity_aliases=base.entity_aliases,
        source_supported_entities=support,
    )


def prepare(
    *, base_packets: str | Path, destination: str | Path, config: str | Path, runtime_preflight: str | Path,
    ancestor_model_id: str, ancestor_model_revision: str,
    rewriter_model_id: str, rewriter_model_revision: str,
) -> dict[str, Any]:
    """Generate one shared response-history corpus; never overwrite evidence."""

    target = Path(destination)
    if target.exists():
        raise FileExistsError("refusing to overwrite prepared ancestry-RAG inputs")
    contract = validate_bound_preflight(config=config, runtime_preflight=runtime_preflight)
    for name, model_id, revision in (
        ("ancestor", ancestor_model_id, ancestor_model_revision),
        ("rewriter", rewriter_model_id, rewriter_model_revision),
    ):
        expected = contract["models"][name]
        if (expected["id"], expected["revision"]) != (model_id, revision):
            raise ValueError(f"requested {name} model differs from the frozen G0 contract")
    base = _load_base(base_packets)
    ancestor_model, ancestor_tokenizer = _load_model(ancestor_model_id, ancestor_model_revision)
    answers = [
        _generate_text(ancestor_model, ancestor_tokenizer, ancestor_model_id, _base_prompt(question), seed=_seed("ancestor", ancestor_model_revision, question.question_id))
        for question in base
    ]
    rewriter_model, rewriter_tokenizer = _load_model(rewriter_model_id, rewriter_model_revision)
    prepared: list[Question] = []
    for index, question in enumerate(base):
        cross = _generate_text(rewriter_model, rewriter_tokenizer, rewriter_model_id, _rewrite_prompt(answers[index]), seed=_seed("cross", rewriter_model_revision, question.question_id))
        independent = _generate_text(rewriter_model, rewriter_tokenizer, rewriter_model_id, _independent_summary_prompt(question), seed=_seed("independent_summary", rewriter_model_revision, question.question_id))
        prepared.append(materialize_question(
            question,
            ancestor_answer=answers[index],
            cross_rewrite=cross,
            style_only=answers[(index + 1) % len(answers)],
            independent_summary=independent,
        ))
    target.mkdir(parents=True)
    preflight_copy = target / "runtime_preflight.json"
    preflight_copy.write_bytes(Path(runtime_preflight).read_bytes())
    inputs = target / "frozen_inputs.jsonl"
    write_jsonl(inputs, (asdict(question) for question in prepared))
    write_jsonl(target / "ancestor_answers.jsonl", ({"question_id": question.question_id, "answer": answer} for question, answer in zip(base, answers, strict=True)))
    manifest = {
        "kind": "semantic_ancestry_rag_input_preparation",
        "base_packets_sha256": sha256_file(base_packets),
        "config_sha256": sha256_file(config),
        "runtime_preflight_sha256": sha256_file(preflight_copy),
        "frozen_inputs_sha256": sha256_file(inputs),
        "ancestor_answers_sha256": sha256_file(target / "ancestor_answers.jsonl"),
        "ancestor_model_id": ancestor_model_id,
        "ancestor_model_revision": ancestor_model_revision,
        "rewriter_model_id": rewriter_model_id,
        "rewriter_model_revision": rewriter_model_revision,
        "question_count": len(prepared),
    }
    write_json(target / "PREPARATION.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare frozen answer-history inputs for semantic-ancestry RAG G0")
    parser.add_argument("--base-packets", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--runtime-preflight", required=True)
    parser.add_argument("--ancestor-model-id", required=True)
    parser.add_argument("--ancestor-model-revision", required=True)
    parser.add_argument("--rewriter-model-id", required=True)
    parser.add_argument("--rewriter-model-revision", required=True)
    args = parser.parse_args(argv)
    print(canonical_json(prepare(**vars(args))))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
