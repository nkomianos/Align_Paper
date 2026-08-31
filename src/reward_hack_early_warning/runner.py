"""Durable local runner for the reward-hacking early-warning screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from under_extinction.io import read_jsonl, sha256_file, write_json, write_jsonl

from .analysis import ScreenThresholds, analyze_rollouts


def run(
    *,
    inputs: Sequence[str | Path],
    output: str | Path,
    thresholds: ScreenThresholds = ScreenThresholds(),
) -> dict[str, Any]:
    """Analyze one or more JSONL files into a checksum-bound evidence root."""

    root = Path(output)
    if root.exists():
        raise FileExistsError("refusing to overwrite an early-warning evidence root")
    if not inputs:
        raise ValueError("at least one input JSONL is required")
    records: list[dict[str, Any]] = []
    for path in inputs:
        records.extend(read_jsonl(path))
    report = analyze_rollouts(records, thresholds=thresholds)
    root.mkdir(parents=True)
    frozen = root / "frozen_rollouts.jsonl"
    report_path = root / "REPORT.json"
    write_jsonl(frozen, records)
    write_json(report_path, report)
    manifest = {
        "kind": "reward_hack_early_warning_developmental_evidence",
        "record_count": len(records),
        "frozen_rollouts_sha256": sha256_file(frozen),
        "report_sha256": sha256_file(report_path),
        "decision": report["decision"],
    }
    write_json(root / "MANIFEST.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, dest="inputs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=ScreenThresholds.bootstrap_replicates)
    parser.add_argument("--permutation-replicates", type=int, default=ScreenThresholds.permutation_replicates)
    parser.add_argument("--random-seed", type=int, default=ScreenThresholds.random_seed)
    args = parser.parse_args(argv)
    thresholds = ScreenThresholds(
        bootstrap_replicates=args.bootstrap_replicates,
        permutation_replicates=args.permutation_replicates,
        random_seed=args.random_seed,
    )
    manifest = run(inputs=args.inputs, output=args.output, thresholds=thresholds)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
