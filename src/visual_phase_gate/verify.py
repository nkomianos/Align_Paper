from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import yaml

from under_extinction.io import read_jsonl, sha256_file, write_json

from .analysis import PhaseThresholds, evaluate_gate, score_family
from .corpus import load_cases


def verify(*, config: str | Path, roots: Sequence[str | Path], destination: str | Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    thresholds = PhaseThresholds(**cfg["thresholds"])
    families: dict[str, Any] = {}
    for item in roots:
        root = Path(item)
        manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
        if manifest.get("kind") != "visual_patch_phase_family":
            raise ValueError("not a visual patch-phase evidence root")
        inputs, raw = root / "frozen_inputs.jsonl", root / "raw_completions.jsonl"
        if sha256_file(inputs) != manifest["inputs_sha256"] or sha256_file(raw) != manifest["raw_sha256"]:
            raise ValueError("visual-phase evidence checksum mismatch")
        family = next((name for name, record in cfg["models"].items() if record["id"] == manifest["model_id"] and record["revision"] == manifest["revision"]), None)
        if family is None or family in families:
            raise ValueError("unknown or duplicate frozen VLM family")
        families[family] = score_family(load_cases(inputs), read_jsonl(raw), expected_periods=cfg["models"][family]["expected_periods"], thresholds=thresholds)
    report = evaluate_gate(families, thresholds=thresholds)
    report["config_sha256"] = sha256_file(config)
    write_json(destination, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", action="append", required=True, dest="roots")
    parser.add_argument("--destination", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(verify(**vars(args)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
