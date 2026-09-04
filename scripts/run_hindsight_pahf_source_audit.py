"""Seal a metadata-only audit of the pinned public PAHF release."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from interaction_sprint.pahf_source_audit import audit_pinned_source


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).parents[1]
    result = audit_pinned_source(args.source_root)
    write_json(args.root / "spec.json", {
        "version": "pahf-source-audit-v1",
        "content_policy": "metadata and hashes only; no public scenario text copied",
        "paper_green_light": False,
    })
    write_json(args.root / "RESULT.json", result)
    sources = [
        repository / "src" / "interaction_sprint" / "pahf_source_audit.py",
        Path(__file__),
        repository / "scripts" / "verify_hindsight_pahf_source_audit.py",
    ]
    write_json(args.root / "source_hashes.json", {
        str(path.relative_to(repository)).replace("\\", "/"): sha256(path)
        for path in sources
    })
    files = sorted(path for path in args.root.iterdir() if path.name != "MANIFEST.json")
    write_json(args.root / "MANIFEST.json", {path.name: sha256(path) for path in files})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

