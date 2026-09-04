"""Read-only deterministic verifier for the Hindsight gradient power audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from interaction_sprint.hindsight_gradient_power import run_power_audit


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
    manifest = json.loads((args.root / "MANIFEST.json").read_text(encoding="utf-8"))
    for name, expected in manifest.items():
        if sha256(args.root / name) != expected:
            raise SystemExit(f"manifest mismatch: {name}")
    repository = Path(__file__).parents[1]
    sources = [
        repository / "src" / "interaction_sprint" / "hindsight_gradient_power.py",
        repository / "src" / "interaction_sprint" / "hindsight_neural_gradient.py",
        repository / "src" / "interaction_sprint" / "hindsight_neural_anchor.py",
        repository / "scripts" / "run_hindsight_gradient_power_audit.py",
        Path(__file__),
    ]
    expected_sources = {
        str(path.relative_to(repository)).replace("\\", "/"): sha256(path)
        for path in sources
    }
    saved_sources = json.loads((args.root / "source_hashes.json").read_text(encoding="utf-8"))
    if saved_sources != expected_sources:
        raise SystemExit("source hash mismatch")
    saved = json.loads((args.root / "RESULT.json").read_text(encoding="utf-8"))
    replayed = run_power_audit()
    if saved != replayed:
        raise SystemExit("deterministic replay mismatch")
    print(json.dumps({"verified": True, "cells": len(saved["cells"])}, indent=2))


if __name__ == "__main__":
    main()
