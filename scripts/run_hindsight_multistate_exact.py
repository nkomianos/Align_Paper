"""Run and seal the exact finite-state Hindsight identification report."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from interaction_sprint.hindsight_multistate_identification import example_report


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
    args.root.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).parents[1]

    def write(name: str, value: object) -> None:
        (args.root / name).write_text(
            json.dumps(value, indent=2, allow_nan=False), encoding="utf-8",
        )

    write("spec.json", {
        "version": "finite-state-delayed-probe-v1",
        "claim": "full-row-rank delayed emission identifies transition channels by right pseudoinverse",
        "paper_green_light": False,
    })
    write("RESULT.json", {**example_report(), "paper_green_light": False})
    sources = [
        repository / "src" / "interaction_sprint" / "hindsight_multistate_identification.py",
        Path(__file__),
        repository / "scripts" / "verify_hindsight_multistate_exact.py",
    ]
    write("source_hashes.json", {
        str(path.relative_to(repository)).replace("\\", "/"): sha256(path) for path in sources
    })
    files = [path for path in args.root.iterdir() if path.is_file() and path.name != "MANIFEST.json"]
    write("MANIFEST.json", {path.name: sha256(path) for path in sorted(files)})
    print(json.dumps(example_report(), indent=2))


if __name__ == "__main__":
    main()
