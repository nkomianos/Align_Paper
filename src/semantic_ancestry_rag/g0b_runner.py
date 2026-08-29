"""Durable execution of one fully role-separated semantic-ancestry G0b cell."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from under_extinction.io import canonical_json, sha256_file, write_json, write_jsonl

from .g0b import G0BCell
from .g0b_preflight import validate_bound_preflight
from .runner import _score_all, generate, load_questions


RUN_KIND = "semantic_ancestry_rag_g0b_cell"


def _cell_from_preparation(prepared_root: str | Path, cell_name: str, config: str | Path, runtime_preflight: str | Path) -> tuple[Path, Path, G0BCell]:
    root = Path(prepared_root)
    manifest = json.loads((root / "PREPARATION.json").read_text(encoding="utf-8"))
    if manifest.get("kind") != "semantic_ancestry_rag_g0b_input_preparation":
        raise ValueError("not a completed G0b preparation root")
    if manifest.get("config_sha256") != sha256_file(config):
        raise ValueError("prepared inputs are not bound to this frozen G0b config")
    preflight_copy = root / "runtime_preflight.json"
    if sha256_file(preflight_copy) != manifest.get("runtime_preflight_sha256"):
        raise ValueError("prepared-root preflight hash mismatch")
    if sha256_file(runtime_preflight) != sha256_file(preflight_copy):
        raise ValueError("requested preflight differs from input-preparation preflight")
    cells = manifest.get("cells")
    if not isinstance(cells, Mapping) or not isinstance(cells.get(cell_name), Mapping):
        raise ValueError("requested G0b cell is absent from the frozen preparation")
    cell_root = root / "cells" / cell_name
    roles_path, inputs = cell_root / "ROLES.json", cell_root / "frozen_inputs.jsonl"
    role_record = cells[cell_name]
    if sha256_file(roles_path) != role_record.get("roles_sha256") or sha256_file(inputs) != role_record.get("inputs_sha256"):
        raise ValueError("prepared G0b cell evidence hash mismatch")
    roles = G0BCell(**json.loads(roles_path.read_text(encoding="utf-8")))
    contract = validate_bound_preflight(config=config, runtime_preflight=runtime_preflight)
    if asdict(roles) not in contract["role_cells"]:
        raise ValueError("prepared G0b roles are not an allowed frozen config cell")
    return inputs, cell_root, roles


def run(
    *, prepared_root: str | Path, cell_name: str, output: str | Path, config: str | Path, runtime_preflight: str | Path,
    model_id: str, model_revision: str, completions_per_cell: int = 4, temperature: float = 0.8, max_new_tokens: int = 128,
) -> dict[str, Any]:
    """Run one cell, checkpoint every completion, and never overwrite a root."""

    root = Path(output)
    if root.exists():
        raise FileExistsError("refusing to overwrite a semantic-ancestry G0b cell root")
    inputs, prepared_cell, roles = _cell_from_preparation(prepared_root, cell_name, config, runtime_preflight)
    contract = validate_bound_preflight(config=config, runtime_preflight=runtime_preflight)
    expected = contract["models"]["serving_families"].get(roles.serving_model)
    if not isinstance(expected, Mapping) or (expected.get("id"), expected.get("revision")) != (model_id, model_revision):
        raise ValueError("G0b serving model differs from the frozen role cell")
    if completions_per_cell != int(contract["completions_per_cell"]):
        raise ValueError("G0b completions-per-cell differs from the frozen contract")
    questions = load_questions(inputs)
    if len(questions) != int(contract["question_count"]):
        raise ValueError("G0b input count differs from frozen contract")
    root.mkdir(parents=True)
    input_copy, roles_copy, transformations_copy, preflight_copy = (
        root / "frozen_inputs.jsonl", root / "ROLES.json", root / "transformations.jsonl", root / "runtime_preflight.json",
    )
    input_copy.write_bytes(inputs.read_bytes())
    roles_copy.write_bytes((prepared_cell / "ROLES.json").read_bytes())
    transformations_copy.write_bytes((prepared_cell / "transformations.jsonl").read_bytes())
    preflight_copy.write_bytes(Path(runtime_preflight).read_bytes())
    running = {
        "kind": RUN_KIND,
        "status": "INCOMPLETE_DO_NOT_ANALYZE",
        "cell_name": cell_name,
        "records_completed": 0,
        "input_sha256": sha256_file(input_copy),
        "roles_sha256": sha256_file(roles_copy),
    }
    running_path, partial = root / "RUNNING.json", root / "raw_completions.partial.jsonl"
    write_json(running_path, running)
    raw: list[dict[str, Any]] = []
    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        for record in generate(
            questions, model_id=model_id, model_revision=model_revision, completions_per_cell=completions_per_cell,
            temperature=temperature, max_new_tokens=max_new_tokens,
        ):
            raw.append(record)
            handle.write(canonical_json(record) + "\n")
            handle.flush()
            running["records_completed"] = len(raw)
            write_json(running_path, running)
    rows = _score_all(questions, cell_name, raw, completions_per_cell)
    partial.replace(root / "raw_completions.jsonl")
    running_path.unlink()
    write_jsonl(root / "condition_results.jsonl", (asdict(row) for row in rows))
    report = {"status": "AWAITING_FOUR_CELL_G0B_ASSEMBLY", "cell_name": cell_name, "row_count": len(rows)}
    write_json(root / "cell_report.json", report)
    manifest = {
        "kind": RUN_KIND,
        "cell_name": cell_name,
        "roles": asdict(roles),
        "model_id": model_id,
        "model_revision": model_revision,
        "question_count": len(questions),
        "completions_per_cell": completions_per_cell,
        "config_sha256": sha256_file(config),
        "runtime_preflight_sha256": sha256_file(preflight_copy),
        "input_sha256": sha256_file(input_copy),
        "roles_sha256": sha256_file(roles_copy),
        "transformations_sha256": sha256_file(transformations_copy),
        "raw_completions_sha256": sha256_file(root / "raw_completions.jsonl"),
        "condition_results_sha256": sha256_file(root / "condition_results.jsonl"),
        "cell_report_sha256": sha256_file(root / "cell_report.json"),
    }
    write_json(root / "MANIFEST.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one role-separated semantic-ancestry RAG G0b cell")
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--cell-name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--runtime-preflight", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--completions-per-cell", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args(argv)
    print(canonical_json(run(**vars(args))))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
