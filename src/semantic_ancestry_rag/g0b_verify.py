"""Offline integrity verification for a completed semantic-ancestry G0b gate."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from under_extinction.io import canonical_json, read_jsonl, sha256_file, write_json

from .g0b import G0BCell, validate_role_plan
from .g0b_assemble import AGGREGATE_KIND, _load_cell
from .g0b_preflight import load_contract
from .gate import ResultRow, Thresholds, evaluate_gate


KIND = "semantic_ancestry_rag_g0b_verification"


def verify_run(root: str | Path, destination: str | Path | None = None) -> dict[str, Any]:
    path = Path(root)
    manifest_path, config = path / "MANIFEST.json", path / "frozen_config.yaml"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != AGGREGATE_KIND:
        raise ValueError("wrong semantic-ancestry G0b aggregate manifest")
    for key, source in (("config_sha256", config), ("condition_results_sha256", path / "condition_results.jsonl"), ("gate_report_sha256", path / "gate_report.json")):
        if manifest.get(key) != sha256_file(source):
            raise ValueError(f"G0b aggregate manifest mismatch for {key}")
    contract = load_contract(config)
    sources = manifest.get("sources")
    if not isinstance(sources, dict) or len(sources) != 4:
        raise ValueError("G0b aggregate requires four source cells")
    loaded = []
    for name, evidence in sorted(sources.items()):
        if not isinstance(evidence, dict):
            raise ValueError("malformed G0b aggregate source evidence")
        cell_root = path / "cells" / name
        if sha256_file(cell_root / "MANIFEST.json") != evidence.get("source_manifest_sha256"):
            raise ValueError(f"copied G0b source manifest mismatch for {name}")
        cell_manifest, cell_rows = _load_cell(cell_root, config)
        if cell_manifest.get("cell_name") != name:
            raise ValueError("G0b source cell name mismatch")
        loaded.append((cell_manifest, cell_rows))
    cells = tuple(G0BCell(**dict(manifest["roles"])) for manifest, _ in loaded)
    pairs = tuple((str(cell["rewriter_model"]), str(cell["shadow_answer_model"])) for cell in contract["role_cells"])
    validate_role_plan(cells, serving_models=contract["models"]["serving_families"].keys(), external_pairs=pairs)
    recomputed_rows = [row for _, rows in loaded for row in rows]
    recorded_rows = list(read_jsonl(path / "condition_results.jsonl"))
    if canonical_json(recomputed_rows) != canonical_json(recorded_rows):
        raise ValueError("G0b aggregate results do not match its source cells")
    report = evaluate_gate((ResultRow(**row) for row in recomputed_rows), Thresholds(**dict(manifest["thresholds"]))).to_dict()
    recorded_report = json.loads((path / "gate_report.json").read_text(encoding="utf-8"))
    if canonical_json(report) != canonical_json(recorded_report):
        raise ValueError("G0b gate report is not reproducible")
    result = {
        "kind": KIND,
        "run_root": str(path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "recomputed_match": True,
        "decision": report["decision"],
        "pass_gate": report["pass_gate"],
    }
    if destination is not None:
        target = Path(destination)
        if target.exists():
            raise FileExistsError("refusing to overwrite G0b verification")
        write_json(target, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify a completed semantic-ancestry RAG G0b aggregate")
    parser.add_argument("--root", required=True)
    parser.add_argument("--destination")
    args = parser.parse_args(argv)
    print(canonical_json(verify_run(args.root, args.destination)))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
