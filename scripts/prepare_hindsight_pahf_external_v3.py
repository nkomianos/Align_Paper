"""Prepare full-learning, label-counterbalanced EndoPAHF v3 inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from interaction_sprint.hindsight_pahf_full import (
    EXPECTED_FULL_LEARNING_BASES,
    build_full_learning_balanced_partitions,
)
from interaction_sprint.pahf_source_audit import audit_pinned_source


PARENT_V2_MANIFEST_SHA256 = (
    "915dbc068573c50990c42db8aa48a1ba5158f12c4684cea4d0d10727d151e5c2"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--parent-v2-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.parent_v2_root / "MANIFEST.json") != PARENT_V2_MANIFEST_SHA256:
        raise SystemExit("parent v2 manifest mismatch")
    args.root.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).parents[1]
    source = audit_pinned_source(args.source_root)
    partitions = build_full_learning_balanced_partitions(args.source_root)

    def write(name: str, value: object) -> None:
        (args.root / name).write_text(
            json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
        )

    write("spec.json", {
        "scope": "ENDO_PAHF_FULL_LEARNING_LABEL_COUNTERBALANCED_INPUTS_ONLY",
        "version": 3,
        "source_commit": source["source"]["commit"],
        "parent_v2_manifest_sha256": PARENT_V2_MANIFEST_SHA256,
        "learning_base_records": EXPECTED_FULL_LEARNING_BASES,
        "rotations_per_base": 4,
        "development_confirmation_selection_changed": False,
        "repair_reason": "increase unique public learning coverage after a development-only assay-health audit",
        "confirmation_used_for_model_or_threshold_selection": False,
        "paper_green_light": False,
    })
    for name in (
        "learning", "development", "confirmation", "invariants", "base_invariants"
    ):
        write(f"{name}.json", partitions[name])
    write("source_receipt.json", {
        "decision": source["decision"],
        "license": source["source"]["license"],
        "license_sha256": source["source"]["license_sha256"],
    })
    sources = [
        repository / "src" / "interaction_sprint" / "hindsight_pahf_pairs.py",
        repository / "src" / "interaction_sprint" / "hindsight_pahf_balanced.py",
        repository / "src" / "interaction_sprint" / "hindsight_pahf_full.py",
        repository / "src" / "interaction_sprint" / "pahf_source_audit.py",
        Path(__file__),
        repository / "scripts" / "verify_hindsight_pahf_external_v3.py",
    ]
    write("source_hashes.json", {
        str(path.relative_to(repository)).replace("\\", "/"): sha256(path)
        for path in sources
    })
    files = [path for path in args.root.iterdir() if path.is_file()]
    write("MANIFEST.json", {path.name: sha256(path) for path in sorted(files)})
    print(json.dumps({
        "decision": "ENDO_PAHF_V3_FULL_LEARNING_INPUTS_PREPARED",
        "counts": {
            name: len(partitions[name])
            for name in ("learning", "development", "confirmation")
        },
        "confirmation_used_for_model_or_threshold_selection": False,
        "paper_green_light": False,
    }, indent=2))


if __name__ == "__main__":
    main()
