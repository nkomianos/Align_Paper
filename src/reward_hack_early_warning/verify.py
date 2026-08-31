"""Checksum and deterministic-reanalysis verifier for early-warning evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from under_extinction.io import canonical_json, read_jsonl, sha256_file, write_json

from .analysis import ScreenThresholds, analyze_rollouts


def verify(
    *,
    root: str | Path,
    destination: str | Path | None = None,
) -> dict[str, Any]:
    evidence = Path(root)
    manifest = json.loads((evidence / "MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("kind") != "reward_hack_early_warning_developmental_evidence":
        raise ValueError("not a reward-hack early-warning evidence root")
    frozen = evidence / "frozen_rollouts.jsonl"
    report_path = evidence / "REPORT.json"
    if sha256_file(frozen) != manifest.get("frozen_rollouts_sha256"):
        raise ValueError("frozen rollout checksum mismatch")
    if sha256_file(report_path) != manifest.get("report_sha256"):
        raise ValueError("report checksum mismatch")
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    thresholds = ScreenThresholds(**saved["config"])
    recomputed = analyze_rollouts(read_jsonl(frozen), thresholds=thresholds)
    if canonical_json(saved) != canonical_json(recomputed):
        raise ValueError("deterministic reanalysis does not match the saved report")
    if manifest.get("decision") != saved.get("decision"):
        raise ValueError("manifest decision does not match the verified report")
    result = {
        "verified": True,
        "decision": saved["decision"],
        "report_sha256": manifest["report_sha256"],
        "frozen_rollouts_sha256": manifest["frozen_rollouts_sha256"],
    }
    if destination is not None:
        write_json(destination, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--destination")
    args = parser.parse_args(argv)
    print(json.dumps(verify(**vars(args)), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
