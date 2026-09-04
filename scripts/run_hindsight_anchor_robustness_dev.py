"""Run the frozen delayed-anchor selection/contamination DEV audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from interaction_sprint.hindsight_anchor_robustness import run_audit


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).parents[1]
    report = run_audit()
    (args.root / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8",
    )
    sources = [
        repository / "src" / "interaction_sprint" / "hindsight_anchor_robustness.py",
        Path(__file__),
        repository / "scripts" / "verify_hindsight_anchor_robustness_dev.py",
    ]
    (args.root / "source_hashes.json").write_text(json.dumps({
        str(path.relative_to(repository)).replace("\\", "/"): sha256(path) for path in sources
    }, indent=2), encoding="utf-8")
    manifest = {
        path.name: sha256(path) for path in args.root.iterdir()
        if path.is_file() and path.name != "MANIFEST.json"
    }
    (args.root / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    print(json.dumps({
        "decision": report["decision"],
        "gates": report["gates"],
        "aggregate": report["aggregate"],
        "numpy": np.__version__,
    }, indent=2))


if __name__ == "__main__":
    main()

