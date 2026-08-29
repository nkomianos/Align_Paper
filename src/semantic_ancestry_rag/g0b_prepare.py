"""Materialize role-separated, style-matched inputs for semantic-ancestry G0b.

The module generates every transformed document before the corresponding
serving-model evaluation begins.  It never reuses the developmental G0 inputs.
"""

from __future__ import annotations

import argparse
import gc
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from under_extinction.io import canonical_json, read_jsonl, sha256_file, write_json, write_jsonl

from .corpus import BaseQuestion
from .g0b import G0BCell, materialize_question
from .g0b_preflight import validate_bound_preflight
from .prepare import _base_prompt, _generate_text, _independent_summary_prompt, _load_model, _rewrite_prompt, _seed


KIND = "semantic_ancestry_rag_g0b_input_preparation"


def _load_base(path: str | Path) -> tuple[BaseQuestion, ...]:
    records = tuple(BaseQuestion(**row) for row in read_jsonl(path))
    if not records or len({record.question_id for record in records}) != len(records):
        raise ValueError("base packets must be non-empty with unique question IDs")
    return records


def _cell_name(cell: G0BCell) -> str:
    return f"{cell.serving_model}__rewrite_{cell.rewriter_model}__shadow_{cell.shadow_answer_model}"


def _clear_model_memory() -> None:
    """Clear cached CUDA allocations after a local model reference is dropped."""

    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):  # CPU test environments never invoke generation.
        pass


def _model_spec(contract: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    models = contract["models"]
    collections = (models["serving_families"], models["external_models"])
    for collection in collections:
        if role in collection:
            spec = collection[role]
            if isinstance(spec, Mapping):
                return spec
    raise ValueError(f"unknown frozen G0b model role: {role}")


def _generate_answers(base: Sequence[BaseQuestion], model_spec: Mapping[str, Any], purpose: str) -> list[str]:
    model_id, revision = str(model_spec["id"]), str(model_spec["revision"])
    model, tokenizer = _load_model(model_id, revision)
    try:
        return [
            _generate_text(
                model, tokenizer, model_id, _base_prompt(question),
                seed=_seed("g0b", purpose, revision, question.question_id),
            )
            for question in base
        ]
    finally:
        del model, tokenizer
        _clear_model_memory()


def _rewrite_all(
    base: Sequence[BaseQuestion], *, ancestor_answers: Sequence[str], shadow_answers: Sequence[str],
    model_spec: Mapping[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    model_id, revision = str(model_spec["id"]), str(model_spec["revision"])
    model, tokenizer = _load_model(model_id, revision)
    cross, style, summaries = [], [], []
    try:
        for question, ancestor, shadow in zip(base, ancestor_answers, shadow_answers, strict=True):
            # The two rewrite calls intentionally share this exact prompt
            # constructor; the immutable manifest records the shared model.
            cross.append(_generate_text(model, tokenizer, model_id, _rewrite_prompt(ancestor), seed=_seed("g0b", "cross", revision, question.question_id)))
            style.append(_generate_text(model, tokenizer, model_id, _rewrite_prompt(shadow), seed=_seed("g0b", "style", revision, question.question_id)))
            summaries.append(_generate_text(model, tokenizer, model_id, _independent_summary_prompt(question), seed=_seed("g0b", "summary", revision, question.question_id)))
    finally:
        del model, tokenizer
        _clear_model_memory()
    return cross, style, summaries


def prepare(*, base_packets: str | Path, destination: str | Path, config: str | Path, runtime_preflight: str | Path) -> dict[str, Any]:
    """Build all four G0b frozen input cells without overwriting evidence."""

    target = Path(destination)
    if target.exists():
        raise FileExistsError("refusing to overwrite G0b prepared inputs")
    contract = validate_bound_preflight(config=config, runtime_preflight=runtime_preflight)
    base = _load_base(base_packets)
    if len(base) != int(contract["question_count"]):
        raise ValueError("base-packet count differs from frozen G0b contract")
    cells = tuple(G0BCell(**row) for row in contract["role_cells"])
    target.mkdir(parents=True)
    preflight_copy = target / "runtime_preflight.json"
    preflight_copy.write_bytes(Path(runtime_preflight).read_bytes())
    base_copy = target / "base_packets.jsonl"
    base_copy.write_bytes(Path(base_packets).read_bytes())
    cell_manifest: dict[str, Any] = {}
    for cell in cells:
        name = _cell_name(cell)
        cell_root = target / "cells" / name
        if cell_root.exists():  # Defensive: target was freshly created, but preserve if interrupted manually.
            raise FileExistsError(f"refusing to overwrite G0b cell: {cell_root}")
        ancestor_spec = _model_spec(contract, cell.ancestor_model)
        shadow_spec = _model_spec(contract, cell.shadow_answer_model)
        rewriter_spec = _model_spec(contract, cell.rewriter_model)
        ancestor = _generate_answers(base, ancestor_spec, f"ancestor:{name}")
        shadow = _generate_answers(base, shadow_spec, f"shadow:{name}")
        cross, style, summary = _rewrite_all(
            base, ancestor_answers=ancestor, shadow_answers=shadow, model_spec=rewriter_spec,
        )
        questions = [
            materialize_question(
                question, ancestor_answer=ancestor[index], shadow_answer=shadow[index],
                cross_rewrite=cross[index], style_rewrite=style[index], independent_summary=summary[index],
            )
            for index, question in enumerate(base)
        ]
        cell_root.mkdir(parents=True)
        inputs = cell_root / "frozen_inputs.jsonl"
        write_jsonl(inputs, (asdict(question) for question in questions))
        transformations = cell_root / "transformations.jsonl"
        write_jsonl(transformations, (
            {
                "question_id": question.question_id,
                "ancestor_answer": ancestor[index],
                "shadow_answer": shadow[index],
                "cross_rewrite": cross[index],
                "style_rewrite": style[index],
                "independent_summary": summary[index],
            }
            for index, question in enumerate(base)
        ))
        roles = asdict(cell)
        write_json(cell_root / "ROLES.json", roles)
        cell_manifest[name] = {
            "roles": roles,
            "inputs_sha256": sha256_file(inputs),
            "transformations_sha256": sha256_file(transformations),
            "roles_sha256": sha256_file(cell_root / "ROLES.json"),
        }
    manifest = {
        "kind": KIND,
        "config_sha256": sha256_file(config),
        "runtime_preflight_sha256": sha256_file(preflight_copy),
        "base_packets_sha256": sha256_file(base_copy),
        "question_count": len(base),
        "cells": cell_manifest,
    }
    write_json(target / "PREPARATION.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare role-separated semantic-ancestry RAG G0b inputs")
    parser.add_argument("--base-packets", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--runtime-preflight", required=True)
    args = parser.parse_args(argv)
    print(canonical_json(prepare(**vars(args))))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
