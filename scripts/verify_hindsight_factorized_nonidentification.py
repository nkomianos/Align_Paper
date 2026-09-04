"""Read-only verifier for the exact factorized non-identification result."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from interaction_sprint.hindsight_factorized_nonidentification import frozen_result


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
        path = args.root / name
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"manifest mismatch: {name}")
    if json.loads((args.root / "RESULT.json").read_text(encoding="utf-8")) != frozen_result():
        raise SystemExit("exact result replay mismatch")
    repository = Path(__file__).parents[1]
    sources = [
        repository / "src" / "interaction_sprint" / "hindsight_factorized_nonidentification.py",
        repository / "scripts" / "run_hindsight_factorized_nonidentification.py",
        Path(__file__),
    ]
    expected_sources = {
        str(path.relative_to(repository)).replace("\\", "/"): sha256(path)
        for path in sources
    }
    if json.loads((args.root / "source_hashes.json").read_text(encoding="utf-8")) != expected_sources:
        raise SystemExit("source hash mismatch")
    result = frozen_result()
    print(json.dumps({
        "verified": True,
        "decision": result["decision"],
        "manifest_sha256": sha256(args.root / "MANIFEST.json"),
        "immediate_log_total_variation": result["immediate_log_total_variation"],
        "delayed_probe_total_variation": result["delayed_probe_total_variation"],
        "best_persistent_action": result["best_persistent_action"],
        "paper_green_light": False,
    }, indent=2))


if __name__ == "__main__":
    main()
