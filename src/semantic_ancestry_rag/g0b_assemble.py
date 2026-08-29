"""Assemble four independently durable semantic-ancestry G0b cells."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from under_extinction.io import canonical_json, read_jsonl, sha256_file, write_json, write_jsonl

from .g0b import G0BCell, validate_role_plan
from .g0b_preflight import load_contract, validate_bound_preflight
from .g0b_runner import RUN_KIND
from .gate import ResultRow, Thresholds, evaluate_gate
from .runner import _score_all, load_questions


AGGREGATE_KIND = "semantic_ancestry_rag_g0b_aggregate"

# A recovered aggregate must retain every file whose digest is bound by the
# source cell manifest. Keep this mapping shared by loading and copying so a
# future aggregate cannot silently omit verification-critical evidence.
CELL_EVIDENCE_DIGEST_KEYS = (
    ("frozen_inputs.jsonl", "input_sha256"),
    ("ROLES.json", "roles_sha256"),
    ("transformations.jsonl", "transformations_sha256"),
    ("raw_completions.jsonl", "raw_completions_sha256"),
    ("condition_results.jsonl", "condition_results_sha256"),
    ("runtime_preflight.json", "runtime_preflight_sha256"),
    ("cell_report.json", "cell_report_sha256"),
)
CELL_EVIDENCE_FILES = tuple(filename for filename, _ in CELL_EVIDENCE_DIGEST_KEYS)


def _load_cell(path: Path, config: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((path / "MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("kind") != RUN_KIND:
        raise ValueError(f"not a completed G0b cell root: {path}")
    for filename, key in CELL_EVIDENCE_DIGEST_KEYS:
        if sha256_file(path / filename) != manifest.get(key):
            raise ValueError(f"G0b cell evidence hash mismatch for {filename}: {path}")
    if manifest.get("config_sha256") != sha256_file(config):
        raise ValueError("G0b cell is bound to a different config")
    validate_bound_preflight(config=config, runtime_preflight=path / "runtime_preflight.json")
    roles = G0BCell(**json.loads((path / "ROLES.json").read_text(encoding="utf-8")))
    if asdict(roles) != manifest.get("roles"):
        raise ValueError("G0b cell role manifest mismatch")
    contract = load_contract(config)
    if asdict(roles) not in contract["role_cells"]:
        raise ValueError("G0b cell roles are absent from frozen config")
    expected = contract["models"]["serving_families"][roles.serving_model]
    if (manifest.get("model_id"), manifest.get("model_revision")) != (expected["id"], expected["revision"]):
        raise ValueError("G0b serving identity mismatches its role cell")
    questions = load_questions(path / "frozen_inputs.jsonl")
    raw = list(read_jsonl(path / "raw_completions.jsonl"))
    recomputed = _score_all(questions, str(manifest["cell_name"]), raw, int(manifest["completions_per_cell"]))
    recorded = [ResultRow(**row) for row in read_jsonl(path / "condition_results.jsonl")]
    if canonical_json([asdict(row) for row in recomputed]) != canonical_json([asdict(row) for row in recorded]):
        raise ValueError("G0b condition results do not reproduce from raw completions")
    return manifest, [asdict(row) for row in recorded]


def assemble(*, roots: Sequence[str | Path], output: str | Path, config: str | Path) -> dict[str, Any]:
    if len(roots) != 4:
        raise ValueError("G0b requires exactly four fully crossed cell roots")
    destination = Path(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite G0b aggregate evidence")
    contract = load_contract(config)
    loaded = [_load_cell(Path(root), config) for root in roots]
    manifests = [entry[0] for entry in loaded]
    cells = tuple(G0BCell(**dict(manifest["roles"])) for manifest in manifests)
    expected_pairs = tuple((str(cell["rewriter_model"]), str(cell["shadow_answer_model"])) for cell in contract["role_cells"])
    validate_role_plan(cells, serving_models=contract["models"]["serving_families"].keys(), external_pairs=expected_pairs)
    names = [str(manifest["cell_name"]) for manifest in manifests]
    if len(set(names)) != 4:
        raise ValueError("duplicate G0b cell root")
    shared = ("config_sha256", "question_count", "completions_per_cell")
    for key in shared:
        if len({canonical_json(manifest.get(key)) for manifest in manifests}) != 1:
            raise ValueError(f"G0b cell roots disagree on frozen {key}")
    thresholds = Thresholds(**dict(contract["thresholds"]))
    rows = [ResultRow(**row) for _, records in loaded for row in records]
    report = evaluate_gate(rows, thresholds).to_dict()
    destination.mkdir(parents=True)
    frozen_config = destination / "frozen_config.yaml"
    frozen_config.write_bytes(Path(config).read_bytes())
    all_rows = [asdict(row) for row in rows]
    write_jsonl(destination / "condition_results.jsonl", all_rows)
    write_json(destination / "gate_report.json", report)
    sources: dict[str, dict[str, Any]] = {}
    for root, manifest in zip(roots, manifests, strict=True):
        name = str(manifest["cell_name"])
        source, target = Path(root), destination / "cells" / name
        target.mkdir(parents=True)
        for filename in (*CELL_EVIDENCE_FILES, "MANIFEST.json"):
            shutil.copyfile(source / filename, target / filename)
        sources[name] = {
            "source_manifest_sha256": sha256_file(source / "MANIFEST.json"),
            "roles": manifest["roles"],
            "model_id": manifest["model_id"],
            "model_revision": manifest["model_revision"],
            "input_sha256": sha256_file(target / "frozen_inputs.jsonl"),
            "raw_completions_sha256": sha256_file(target / "raw_completions.jsonl"),
            "condition_results_sha256": sha256_file(target / "condition_results.jsonl"),
        }
    manifest = {
        "kind": AGGREGATE_KIND,
        "config_sha256": sha256_file(frozen_config),
        "question_count": manifests[0]["question_count"],
        "completions_per_cell": manifests[0]["completions_per_cell"],
        "thresholds": asdict(thresholds),
        "condition_results_sha256": sha256_file(destination / "condition_results.jsonl"),
        "gate_report_sha256": sha256_file(destination / "gate_report.json"),
        "sources": sources,
    }
    write_json(destination / "MANIFEST.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assemble the four fully crossed semantic-ancestry G0b cells")
    parser.add_argument("--root", action="append", required=True, dest="roots")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    print(canonical_json(assemble(**vars(args))))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
