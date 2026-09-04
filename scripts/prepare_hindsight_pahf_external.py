"""Prepare non-overwriting EndoPAHF natural-surface inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from interaction_sprint.hindsight_pahf_pairs import build_pinned_partitions
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
    source_audit = audit_pinned_source(args.source_root)
    partitions = build_pinned_partitions(args.source_root)
    for name in ("learning", "development", "confirmation"):
        write_json(args.root / f"{name}.json", partitions[name])
    write_json(args.root / "spec.json", {
        "version": "endo-pahf-natural-surface-v1",
        "source_commit": source_audit["source"]["commit"],
        "worlds": ["transient_expression", "persistent_transition"],
        "ordinary_logs_identical_by_construction": True,
        "delayed_neutral_probe_separates_worlds": True,
        "paper_green_light": False,
    })
    write_json(args.root / "invariants.json", partitions["invariants"])
    write_json(args.root / "source_receipt.json", {
        "decision": source_audit["decision"],
        "license": source_audit["source"]["license"],
        "license_sha256": source_audit["source"]["license_sha256"],
    })
    repository = Path(__file__).parents[1]
    sources = [
        repository / "src" / "interaction_sprint" / "hindsight_pahf_pairs.py",
        repository / "src" / "interaction_sprint" / "pahf_source_audit.py",
        Path(__file__),
        repository / "scripts" / "verify_hindsight_pahf_external.py",
    ]
    write_json(args.root / "source_hashes.json", {
        str(path.relative_to(repository)).replace("\\", "/"): sha256(path)
        for path in sources
    })
    files = sorted(path for path in args.root.iterdir() if path.name != "MANIFEST.json")
    write_json(args.root / "MANIFEST.json", {path.name: sha256(path) for path in files})
    print(json.dumps(partitions["invariants"], indent=2))


if __name__ == "__main__":
    main()

