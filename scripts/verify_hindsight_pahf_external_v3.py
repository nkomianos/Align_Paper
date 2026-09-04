"""Read-only replay verifier for full-learning EndoPAHF v3 inputs."""
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
from prepare_hindsight_pahf_external_v3 import PARENT_V2_MANIFEST_SHA256


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
    manifest = json.loads((args.root / "MANIFEST.json").read_text(encoding="utf-8"))
    for name, expected in manifest.items():
        path = args.root / name
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"manifest mismatch: {name}")
    source = audit_pinned_source(args.source_root)
    expected_spec = {
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
    }
    if json.loads((args.root / "spec.json").read_text(encoding="utf-8")) != expected_spec:
        raise SystemExit("spec mismatch")
    expected = build_full_learning_balanced_partitions(args.source_root)
    for name in (
        "learning", "development", "confirmation", "invariants", "base_invariants"
    ):
        actual = json.loads((args.root / f"{name}.json").read_text(encoding="utf-8"))
        if actual != expected[name]:
            raise SystemExit(f"replay mismatch: {name}")
    counts = {
        name: len(expected[name])
        for name in ("learning", "development", "confirmation")
    }
    if counts != {"learning": 2520, "development": 384, "confirmation": 1024}:
        raise SystemExit(f"unexpected v3 counts: {counts}")
    if expected["base_invariants"]["learning"]["n"] != EXPECTED_FULL_LEARNING_BASES:
        raise SystemExit("full learning pool was not retained")
    expected_receipt = {
        "decision": source["decision"],
        "license": source["source"]["license"],
        "license_sha256": source["source"]["license_sha256"],
    }
    if json.loads((args.root / "source_receipt.json").read_text(encoding="utf-8")) != expected_receipt:
        raise SystemExit("source receipt mismatch")
    repository = Path(__file__).parents[1]
    sources = [
        repository / "src" / "interaction_sprint" / "hindsight_pahf_pairs.py",
        repository / "src" / "interaction_sprint" / "hindsight_pahf_balanced.py",
        repository / "src" / "interaction_sprint" / "hindsight_pahf_full.py",
        repository / "src" / "interaction_sprint" / "pahf_source_audit.py",
        repository / "scripts" / "prepare_hindsight_pahf_external_v3.py",
        Path(__file__),
    ]
    expected_sources = {
        str(path.relative_to(repository)).replace("\\", "/"): sha256(path)
        for path in sources
    }
    if json.loads((args.root / "source_hashes.json").read_text(encoding="utf-8")) != expected_sources:
        raise SystemExit("source hash mismatch")
    print(json.dumps({
        "verified": True,
        "decision": "ENDO_PAHF_V3_FULL_LEARNING_INPUTS_REPLAY_VERIFIED",
        "manifest_sha256": sha256(args.root / "MANIFEST.json"),
        "counts": counts,
        "confirmation_used_for_model_or_threshold_selection": False,
        "paper_green_light": False,
    }, indent=2))


if __name__ == "__main__":
    main()
