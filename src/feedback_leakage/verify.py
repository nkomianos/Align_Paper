"""Checksum-bound offline verification of two feedback-leakage model roots."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from under_extinction.io import read_jsonl, sha256_file, write_json

from .analysis import GateThresholds, evaluate_gate, score_family
from .corpus import FeedbackCase, load_cases


def _validate_answer_key(cases: Sequence[FeedbackCase], path: str | Path) -> dict[str, str]:
    rows = list(read_jsonl(path))
    keyed = {str(row["case_id"]): row for row in rows}
    if len(keyed) != len(rows) or set(keyed) != {case.case_id for case in cases}:
        raise ValueError("answer key IDs do not exactly match the frozen corpus")
    fields = (
        "verification_tool",
        "consequential_tool",
        "verification_arguments",
        "target_arguments",
        "proof_field",
    )
    for case in cases:
        public = asdict(case)
        if any(keyed[case.case_id].get(field) != public[field] for field in fields):
            raise ValueError(f"answer key mismatch for {case.case_id}")
        receipt = keyed[case.case_id].get("issued_receipt")
        if not isinstance(receipt, str) or not receipt:
            raise ValueError(f"answer key receipt is missing for {case.case_id}")
    return {case_id: str(row["issued_receipt"]) for case_id, row in keyed.items()}


def verify(
    *,
    config: str | Path,
    answer_key: str | Path,
    roots: Sequence[str | Path],
    destination: str | Path,
) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    thresholds = GateThresholds(**cfg["thresholds"])
    families: dict[str, Any] = {}
    canonical_inputs_sha: str | None = None
    canonical_cases: tuple[FeedbackCase, ...] | None = None
    receipts: dict[str, str] | None = None
    for item in roots:
        root = Path(item)
        manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
        if manifest.get("kind") != "feedback_leakage_g0_family":
            raise ValueError(f"not a feedback-leakage family root: {root}")
        inputs = root / "frozen_inputs.jsonl"
        raw = root / "raw_completions.jsonl"
        if sha256_file(inputs) != manifest["inputs_sha256"] or sha256_file(raw) != manifest["raw_sha256"]:
            raise ValueError(f"evidence checksum mismatch: {root}")
        if canonical_inputs_sha is None:
            canonical_inputs_sha = manifest["inputs_sha256"]
            canonical_cases = load_cases(inputs)
            receipts = _validate_answer_key(canonical_cases, answer_key)
        elif manifest["inputs_sha256"] != canonical_inputs_sha:
            raise ValueError("model families were not run on byte-identical inputs")
        family = next(
            (
                name
                for name, model in cfg["models"].items()
                if model["id"] == manifest["model_id"] and model["revision"] == manifest["revision"]
            ),
            None,
        )
        if family is None or family in families or canonical_cases is None or receipts is None:
            raise ValueError("family root is absent or duplicated in the frozen config")
        if manifest.get("runtime_key_sha256") != sha256_file(answer_key):
            raise ValueError("family root was not generated with the committed runtime key")
        families[family] = score_family(canonical_cases, read_jsonl(raw), receipts, thresholds=thresholds)
    report = evaluate_gate(families, thresholds=thresholds)
    report["config_sha256"] = sha256_file(config)
    report["answer_key_sha256"] = sha256_file(answer_key)
    report["inputs_sha256"] = canonical_inputs_sha
    write_json(destination, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--answer-key", required=True)
    parser.add_argument("--root", action="append", required=True, dest="roots")
    parser.add_argument("--destination", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(verify(**vars(args)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
