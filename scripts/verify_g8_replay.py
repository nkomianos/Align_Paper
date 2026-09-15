#!/usr/bin/env python3
"""Verify sealed G8 short-run manifests and bitwise source/replay identity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from g8_common import binary_sha, write_json

RUNTIME_KEYS = {"wall_seconds", "gpu_wall_seconds", "gpu", "training"}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def strip_runtime(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: strip_runtime(item) for key, item in value.items() if key not in RUNTIME_KEYS}
    if isinstance(value, list):
        return [strip_runtime(item) for item in value]
    return value


def manifest_errors(root: Path) -> list[str]:
    manifest = load(root / "MANIFEST.json")
    return [name for name, wanted in manifest.items()
            if not (root / name).is_file() or binary_sha(root / name) != wanted]


def compare_run(source: Path, replay: Path) -> dict[str, Any]:
    source_report, replay_report = load(source / "REPORT.json"), load(replay / "REPORT.json")
    report_exact = strip_runtime(source_report) == strip_runtime(replay_report)
    source_manifest, replay_manifest = load(source / "MANIFEST.json"), load(replay / "MANIFEST.json")
    common = sorted(set(source_manifest) & set(replay_manifest))
    raw_exact = all(source_manifest[name] == replay_manifest[name] for name in common
                    if name.endswith("raw_predictions.jsonl"))
    return {"source_manifest_errors": manifest_errors(source),
            "replay_manifest_errors": manifest_errors(replay),
            "scientific_report_exact": report_exact, "raw_predictions_exact": raw_exact}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); cfg = load(args.config)
    checks = []
    calibration = load(args.root / "CALIBRATION_DECISION.json")
    for summary in calibration["cells"]:
        cell = summary["cell"]["id"]
        for arm in cfg["arms"]:
            for seed in cfg["seeds"]:
                checks.append({"stage": "calibration", "cell": cell, "arm": arm, "seed": seed,
                               **compare_run(args.root / "source" / "calibration" / cell / arm / f"seed_{seed}",
                                             args.root / "replay" / "calibration" / cell / arm / f"seed_{seed}")})
    if calibration["decision"] == "ADVANCE":
        for arm in cfg["arms"]:
            for seed in cfg["seeds"]:
                checks.append({"stage": "posttraining", "arm": arm, "seed": seed,
                               **compare_run(args.root / "source" / "posttraining" / arm / f"seed_{seed}",
                                             args.root / "replay" / "posttraining" / arm / f"seed_{seed}")})
    passed = all(not row["source_manifest_errors"] and not row["replay_manifest_errors"] and
                 row["scientific_report_exact"] and row["raw_predictions_exact"] for row in checks)
    result = {"status": "PASS" if passed else "FAIL", "short_posttraining_bitwise_exact": passed,
              "inherited_pretraining_rule": "decision reproducibility only; G7 source checkpoints are fixed inputs",
              "checks": checks}
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

