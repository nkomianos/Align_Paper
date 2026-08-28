"""Offline integrity verifier for completed SENTRY G0 artifact roots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from under_extinction.io import canonical_json, read_jsonl, sha256_file, write_json

from .gate import Scenario, evaluate_gate
from .runner import RUN_KIND


def verify_run(root: str | Path, destination: str | Path | None = None) -> dict[str, Any]:
    """Recompute the report from raw scenario rows and verify immutable hashes.

    The private answer key is intentionally not needed: integrity of the reported
    outcome is checked from the preserved rows and precommitted input digest.
    """

    path = Path(root)
    manifest_path, rows_path, report_path = path / "MANIFEST.json", path / "scenario_results.jsonl", path / "gate_report.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != RUN_KIND:
        raise ValueError("wrong SENTRY run manifest")
    for key, source in (("scenario_results_sha256", rows_path), ("gate_report_sha256", report_path), ("public_sources_sha256", path / "public_sources.sha256")):
        if manifest.get(key) != sha256_file(source):
            raise ValueError(f"SENTRY manifest mismatch for {key}")
    rows = list(read_jsonl(rows_path))
    recomputed = evaluate_gate([
        Scenario(**{key: row[key] for key in Scenario.__dataclass_fields__}) for row in rows
    ]).to_dict()
    reported = json.loads(report_path.read_text(encoding="utf-8"))
    if canonical_json(recomputed) != canonical_json(reported):
        raise ValueError("SENTRY report is not reproducible from the raw scenario rows")
    result = {"kind": "sentry_g0_retrieval_verification", "run_root": str(path.resolve()), "manifest_sha256": sha256_file(manifest_path), "scenario_results_sha256": sha256_file(rows_path), "gate_report_sha256": sha256_file(report_path), "recomputed_match": True, "decision": recomputed["decision"], "pass_gate": recomputed["pass_gate"]}
    if destination is not None:
        target = Path(destination)
        if target.exists():
            raise FileExistsError("refusing to overwrite SENTRY verification")
        write_json(target, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify a retrieved SENTRY G0 run")
    parser.add_argument("--root", required=True)
    parser.add_argument("--destination")
    args = parser.parse_args(argv)
    print(canonical_json(verify_run(args.root, args.destination)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
