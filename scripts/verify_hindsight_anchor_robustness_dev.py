"""Read-only replay verifier for the delayed-anchor robustness audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from interaction_sprint.hindsight_anchor_robustness import run_audit


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.root / "MANIFEST.json").read_text(encoding="utf-8"))
    for name, expected in manifest.items():
        path = args.root / name
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"manifest mismatch: {name}")
    repository = Path(__file__).parents[1]
    sources = [
        repository / "src" / "interaction_sprint" / "hindsight_anchor_robustness.py",
        repository / "scripts" / "run_hindsight_anchor_robustness_dev.py",
        Path(__file__),
    ]
    expected_sources = {
        str(path.relative_to(repository)).replace("\\", "/"): sha256(path) for path in sources
    }
    observed_sources = json.loads((args.root / "source_hashes.json").read_text(encoding="utf-8"))
    if observed_sources != expected_sources:
        raise SystemExit("source hash mismatch")
    observed = json.loads((args.root / "report.json").read_text(encoding="utf-8"))
    expected = json.loads(json.dumps(run_audit(), allow_nan=False))
    if observed != expected:
        raise SystemExit("deterministic replay mismatch")
    print(json.dumps({
        "verified": True,
        "decision": observed["decision"],
        "gates": observed["gates"],
    }, indent=2))


if __name__ == "__main__":
    main()

