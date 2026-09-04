"""Read-only verifier for the Hindsight G1 decision-rule power audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from interaction_sprint.hindsight_neural_anchor import build_records
from interaction_sprint.hindsight_neural_policy_g1 import build_disjoint_policy_panels
from interaction_sprint.hindsight_neural_policy_power import (
    EFFECTIVE_GAINS,
    SFT_GAIN_MULTIPLIERS,
    build_power_cells,
    summarize_power_cells,
    target_means,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).parents[1]
    manifest = json.loads((args.root / "MANIFEST.json").read_text(encoding="utf-8"))
    for name, expected in manifest.items():
        if not (args.root / name).is_file() or sha256(args.root / name) != expected:
            raise SystemExit(f"manifest mismatch: {name}")
    expected_spec = {
        "version": "policy-g1-rule-power-v1",
        "effective_gains": list(EFFECTIVE_GAINS),
        "sft_gain_multipliers": list(SFT_GAIN_MULTIPLIERS),
        "alternative_requirement": "at least 6 of 9",
        "null_requirement": "0 of 9",
        "paper_green_light": False,
    }
    if json.loads((args.root / "spec.json").read_text(encoding="utf-8")) != expected_spec:
        raise SystemExit("spec mismatch")
    rows, _, _ = build_records()
    targets = target_means(rows, build_disjoint_policy_panels(rows))
    alternatives, nulls = build_power_cells(targets)
    if json.loads((args.root / "targets.json").read_text(encoding="utf-8")) != targets:
        raise SystemExit("target mismatch")
    if json.loads((args.root / "cells.json").read_text(encoding="utf-8")) != {
        "alternatives": alternatives, "nulls": nulls,
    }:
        raise SystemExit("cell mismatch")
    summary = summarize_power_cells(alternatives, nulls)
    expected_result = {**summary, "paper_green_light": False}
    if json.loads((args.root / "RESULT.json").read_text(encoding="utf-8")) != expected_result:
        raise SystemExit("result mismatch")
    source_paths = [
        repository / "src" / "interaction_sprint" / "hindsight_neural_anchor.py",
        repository / "src" / "interaction_sprint" / "hindsight_neural_policy_g1.py",
        repository / "src" / "interaction_sprint" / "hindsight_neural_policy_power.py",
        repository / "scripts" / "run_hindsight_neural_policy_power_audit.py",
        Path(__file__),
    ]
    expected_sources = {
        str(path.relative_to(repository)).replace("\\", "/"): sha256(path)
        for path in source_paths
    }
    if json.loads((args.root / "source_hashes.json").read_text(encoding="utf-8")) != expected_sources:
        raise SystemExit("source hash mismatch")
    print(json.dumps({"verified": True, **summary}, indent=2))


if __name__ == "__main__":
    main()
