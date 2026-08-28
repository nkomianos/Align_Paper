"""Assemble two independently generated model-family roots into a G0 decision."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from under_extinction.io import canonical_json, read_jsonl, sha256_file, write_json, write_jsonl

from .gate import ResultRow, Thresholds, evaluate_gate
from .verify import RUN_KIND


def _load_family_root(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((path / "MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("kind") != RUN_KIND:
        raise ValueError(f"not a semantic-ancestry family root: {path}")
    for filename, key in (
        ("frozen_inputs.jsonl", "input_sha256"),
        ("condition_results.jsonl", "condition_results_sha256"),
        ("raw_completions.jsonl", "raw_completions_sha256"),
        ("runtime_preflight.json", "runtime_preflight_sha256"),
    ):
        if sha256_file(path / filename) != manifest.get(key):
            raise ValueError(f"family evidence hash mismatch for {filename}: {path}")
    if not isinstance(manifest.get("config_sha256"), str):
        raise ValueError(f"family root lacks its frozen config hash: {path}")
    rows_path = path / "condition_results.jsonl"
    return manifest, list(read_jsonl(rows_path))


def assemble(roots: Sequence[str | Path], output: str | Path) -> dict[str, Any]:
    """Merge exactly two complete families, preserving all source-root digests."""

    if len(roots) != 2:
        raise ValueError("G0 requires exactly two independently generated family roots")
    destination = Path(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite a semantic-ancestry G0 aggregate root")
    loaded = [_load_family_root(Path(root)) for root in roots]
    manifests = [item[0] for item in loaded]
    families = [str(manifest.get("model_family", "")) for manifest in manifests]
    if not all(families) or len(set(families)) != 2:
        raise ValueError("family roots must name two different non-empty model families")
    shared = ("input_sha256", "question_count", "completions_per_cell", "thresholds", "config_sha256")
    for key in shared:
        if len({canonical_json(manifest.get(key)) for manifest in manifests}) != 1:
            raise ValueError(f"family roots disagree on frozen {key}")
    threshold_values = manifests[0]["thresholds"]
    thresholds = Thresholds(**threshold_values)
    raw_rows = [row for _, records in loaded for row in records]
    result_rows = [ResultRow(**row) for row in raw_rows]
    report = evaluate_gate(result_rows, thresholds).to_dict()
    destination.mkdir(parents=True)
    source_input = Path(roots[0]) / "frozen_inputs.jsonl"
    input_target = destination / "frozen_inputs.jsonl"
    input_target.write_bytes(source_input.read_bytes())
    write_jsonl(destination / "condition_results.jsonl", raw_rows)
    write_json(destination / "gate_report.json", report)
    source_evidence: dict[str, dict[str, str]] = {}
    for family, root, manifest in zip(families, roots, manifests, strict=True):
        root_path = Path(root)
        raw_target = destination / f"raw_completions_{family}.jsonl"
        raw_target.write_bytes((root_path / "raw_completions.jsonl").read_bytes())
        preflight_target = destination / f"runtime_preflight_{family}.json"
        preflight_target.write_bytes((root_path / "runtime_preflight.json").read_bytes())
        source_evidence[family] = {
            "source_manifest_sha256": sha256_file(root_path / "MANIFEST.json"),
            "raw_completions_sha256": sha256_file(raw_target),
            "runtime_preflight_sha256": sha256_file(preflight_target),
            "source_model_id": str(manifest.get("model_id", "")),
            "source_model_revision": str(manifest.get("model_revision", "")),
        }
    manifest = {
        "kind": RUN_KIND,
        "question_count": manifests[0]["question_count"],
        "model_families_required": 2,
        "completions_per_cell": manifests[0]["completions_per_cell"],
        "thresholds": threshold_values,
        "config_sha256": manifests[0]["config_sha256"],
        "input_sha256": sha256_file(input_target),
        "condition_results_sha256": sha256_file(destination / "condition_results.jsonl"),
        "gate_report_sha256": sha256_file(destination / "gate_report.json"),
        "source_family_evidence": source_evidence,
    }
    write_json(destination / "MANIFEST.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assemble two semantic-ancestry RAG G0 family roots")
    parser.add_argument("--root", action="append", required=True, dest="roots")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    print(canonical_json(assemble(args.roots, args.output)))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
