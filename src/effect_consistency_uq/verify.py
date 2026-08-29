"""Offline assembly and immutable decision for two family roots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import yaml

from under_extinction.io import read_jsonl, sha256_file, write_json

from .analysis import GateThresholds, evaluate_gate, score_family
from .corpus import load_cases


def _answer_key(path: str | Path) -> dict[str, str]:
    rows = list(read_jsonl(path))
    result = {str(row["task_id"]): str(row["oracle_effect"]) for row in rows}
    if len(result) != len(rows):
        raise ValueError("answer key contains duplicate task IDs")
    return result


def verify(*, config: str | Path, answer_key: str | Path, roots: Sequence[str | Path], destination: str | Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    thresholds = GateThresholds(**cfg["thresholds"])
    families: dict[str, Any] = {}
    key = _answer_key(answer_key)
    for item in roots:
        root = Path(item)
        manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
        if manifest.get("kind") != "effect_consistency_uq_family":
            raise ValueError(f"not an effect-consistency family root: {root}")
        inputs, raw = root / "frozen_inputs.jsonl", root / "raw_completions.jsonl"
        if sha256_file(inputs) != manifest["inputs_sha256"] or sha256_file(raw) != manifest["raw_sha256"]:
            raise ValueError(f"evidence checksum mismatch: {root}")
        family = next((name for name, record in cfg["models"].items() if record["id"] == manifest["model_id"] and record["revision"] == manifest["revision"]), None)
        if family is None or family in families:
            raise ValueError("family root is absent or duplicated in the frozen config")
        cases = load_cases(inputs)
        if set(key) != {case.task_id for case in cases}:
            raise ValueError("answer key is not exactly matched to frozen inputs")
        families[family] = score_family(cases, read_jsonl(raw), key, thresholds=thresholds)
    report = evaluate_gate(families, thresholds=thresholds)
    report["config_sha256"] = sha256_file(config)
    report["answer_key_sha256"] = sha256_file(answer_key)
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
