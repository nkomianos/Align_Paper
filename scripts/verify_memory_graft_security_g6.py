#!/usr/bin/env python3
"""Full deterministic replay verifier for G6."""
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
    for name in (
        "config", "preregistration", "receipt", "s1_root", "s1_verification",
        "g3_source", "g3_verification", "g31_source", "g31_verification",
        "g4_source", "g4_verification", "model_cache", "source", "report",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    return parser.parse_args()


def canonical_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def scientific_values(row: dict[str, Any], temporal: bool) -> dict[str, float]:
    groups = ("metrics",) if temporal else ("causal", "quality_delta", "pre_poison_quality")
    values = {}
    for group in groups:
        for key, value in row[group].items():
            if isinstance(value, (int, float)):
                values[f"{group}.{key}"] = float(value)
    if not temporal:
        values["full_evaluation.clean_nll"] = float(row["full_evaluation"]["clean_nll"]["intact"])
        for key in ("benign_marker_accuracy", "near_trigger_payload_rate", "untriggered_payload_rate"):
            values[f"full_evaluation.{key}"] = float(row["full_evaluation"][key])
    return values


def main() -> None:
    args = parse()
    receipt = json.loads(args.receipt.read_text())
    if receipt.get("verifier_sha256") != canonical_sha(Path(__file__)):
        raise RuntimeError("frozen verifier hash mismatch")
    manifest = json.loads((args.source / "MANIFEST.json").read_text())
    for relative, expected in manifest.items():
        path = args.source / relative
        if not path.is_file() or file_sha(path) != expected:
            raise RuntimeError(f"source manifest mismatch: {relative}")
    runner = Path(__file__).with_name("run_memory_graft_security_g6.py")
    with tempfile.TemporaryDirectory(prefix="g6_replay_") as temporary:
        replay = Path(temporary) / "run"
        command = [sys.executable, str(runner)]
        for name in (
            "config", "preregistration", "receipt", "s1_root", "s1_verification",
            "g3_source", "g3_verification", "g31_source", "g31_verification",
            "g4_source", "g4_verification", "model_cache",
        ):
            command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
        command.extend(["--output", str(replay)])
        subprocess.run(command, check=True)
        source_decision = json.loads((args.source / "DECISION.json").read_text())
        replay_decision = json.loads((replay / "DECISION.json").read_text())
        for decision in (source_decision, replay_decision):
            decision.pop("runner_wall_seconds", None)
        decision_exact = source_decision == replay_decision
        disagreements = {}
        for original in sorted(args.source.rglob("*predictions.jsonl")):
            relative = original.relative_to(args.source)
            source_rows = [json.loads(line) for line in original.read_text().splitlines()]
            replay_rows = [json.loads(line) for line in (replay / relative).read_text().splitlines()]
            if len(source_rows) != len(replay_rows):
                raise RuntimeError(f"prediction row-count mismatch: {relative}")
            disagreements[relative.as_posix()] = sum(left != right for left, right in zip(source_rows, replay_rows))
        maximum_difference = 0.0
        for filename, temporal in (("NEW_ROWS.json", False), ("NEW_TEMPORAL.json", True)):
            left_rows = json.loads((args.source / filename).read_text())
            right_rows = json.loads((replay / filename).read_text())
            identity = (lambda row: (row["model"], row["seed"], row.get("profile")))
            if [identity(row) for row in left_rows] != [identity(row) for row in right_rows]:
                raise RuntimeError(f"row identity mismatch: {filename}")
            for left, right in zip(left_rows, right_rows):
                a = scientific_values(left, temporal)
                b = scientific_values(right, temporal)
                if a.keys() != b.keys():
                    raise RuntimeError(f"scientific key mismatch: {filename}")
                maximum_difference = max(maximum_difference, *(abs(a[key] - b[key]) for key in a))
        passed = decision_exact and not any(disagreements.values()) and maximum_difference <= 1e-12
        report = {
            "kind": "memory_graft_security_g6_full_replay",
            "passed": passed,
            "decision_exact": decision_exact,
            "prediction_row_disagreements": disagreements,
            "maximum_scientific_metric_absolute_difference": maximum_difference,
            "source_manifest_files": len(manifest),
            "source_manifest_sha256": file_sha(args.source / "MANIFEST.json"),
        }
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
        if not passed:
            raise RuntimeError("G6 full replay did not certify")


if __name__ == "__main__":
    main()
