#!/usr/bin/env python3
"""Full deterministic replay verifier for G5."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("config", "preregistration", "receipt", "s1_root",
                 "s1_verification", "model_cache", "source", "report"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    return parser.parse_args()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def scientific_numbers(row: dict[str, Any]) -> dict[str, float]:
    values = {}
    for group in ("causal", "outcomes", "parameter_differences"):
        for key, value in row[group].items():
            values[f"{group}.{key}"] = float(value)
    return values


def main() -> None:
    args = parse()
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    if receipt.get("verifier_sha256") != sha(Path(__file__)):
        raise RuntimeError("frozen verifier hash mismatch")
    manifest = json.loads((args.source / "MANIFEST.json").read_text(encoding="utf-8"))
    for relative, expected in manifest.items():
        path = args.source / relative
        if not path.is_file() or file_sha(path) != expected:
            raise RuntimeError(f"source manifest mismatch: {relative}")
    runner = Path(__file__).with_name("run_memory_graft_security_g5.py")
    source_decision = json.loads((args.source / "DECISION.json").read_text())
    with tempfile.TemporaryDirectory(prefix="g5_replay_") as temporary:
        replay_root = Path(temporary) / "run"
        subprocess.run([
            sys.executable, str(runner), "--config", str(args.config),
            "--preregistration", str(args.preregistration), "--receipt", str(args.receipt),
            "--s1-root", str(args.s1_root), "--s1-verification", str(args.s1_verification),
            "--model-cache", str(args.model_cache), "--output", str(replay_root),
        ], check=True)
        replay_decision = json.loads((replay_root / "DECISION.json").read_text())
        decisions = {}
        for arm, metrics in source_decision["outcomes"].items():
            for metric, record in metrics.items():
                key = f"{arm}.{metric}"
                decisions[key] = record["decision"] == replay_decision["outcomes"][arm][metric]["decision"]
        disagreements = {}
        for original in sorted(args.source.glob("decisive/*/seed_*/*predictions.jsonl")):
            relative = original.relative_to(args.source)
            source_rows = [json.loads(line) for line in original.read_text().splitlines()]
            replay_rows = [json.loads(line) for line in (replay_root / relative).read_text().splitlines()]
            if len(source_rows) != len(replay_rows):
                raise RuntimeError(f"prediction row-count mismatch: {relative}")
            disagreements[str(relative).replace("\\", "/")] = sum(
                left["prediction_id"] != right["prediction_id"]
                for left, right in zip(source_rows, replay_rows)
            )
        source_rows = json.loads((args.source / "DECISIVE.json").read_text())
        replay_rows = json.loads((replay_root / "DECISIVE.json").read_text())
        if [(row["arm"], row["seed"]) for row in source_rows] != [(row["arm"], row["seed"]) for row in replay_rows]:
            raise RuntimeError("decisive row identity mismatch")
        maximum_difference = 0.0
        for left, right in zip(source_rows, replay_rows):
            left_numbers = scientific_numbers(left)
            right_numbers = scientific_numbers(right)
            if left_numbers.keys() != right_numbers.keys():
                raise RuntimeError("scientific metric key mismatch")
            maximum_difference = max(maximum_difference, *(abs(left_numbers[key] - right_numbers[key]) for key in left_numbers))
        passed = all(decisions.values()) and not any(disagreements.values()) and maximum_difference <= 1e-12
        report = {
            "kind": "memory_graft_security_g5_full_replay",
            "passed": passed,
            "decision_reproduction": decisions,
            "prediction_id_disagreements": disagreements,
            "maximum_scientific_metric_absolute_difference": maximum_difference,
            "source_manifest_files": len(manifest),
        }
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        if not passed:
            raise RuntimeError("G5 full replay did not certify")


if __name__ == "__main__":
    main()
