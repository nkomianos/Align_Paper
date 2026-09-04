"""Read-only replay verifier for EndoPAHF natural-surface preparation."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).parents[1]
    manifest = json.loads((args.root / "MANIFEST.json").read_text(encoding="utf-8"))
    for name, expected in manifest.items():
        path = args.root / name
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"manifest mismatch: {name}")
    source_audit = audit_pinned_source(args.source_root)
    expected_spec = {
        "version": "endo-pahf-natural-surface-v1",
        "source_commit": source_audit["source"]["commit"],
        "worlds": ["transient_expression", "persistent_transition"],
        "ordinary_logs_identical_by_construction": True,
        "delayed_neutral_probe_separates_worlds": True,
        "paper_green_light": False,
    }
    if json.loads((args.root / "spec.json").read_text(encoding="utf-8")) != expected_spec:
        raise SystemExit("spec mismatch")
    partitions = build_pinned_partitions(args.source_root)
    for name in ("learning", "development", "confirmation"):
        observed = json.loads((args.root / f"{name}.json").read_text(encoding="utf-8"))
        if observed != partitions[name]:
            raise SystemExit(f"partition mismatch: {name}")
    if json.loads((args.root / "invariants.json").read_text(encoding="utf-8")) != partitions["invariants"]:
        raise SystemExit("invariants mismatch")
    expected_receipt = {
        "decision": source_audit["decision"],
        "license": source_audit["source"]["license"],
        "license_sha256": source_audit["source"]["license_sha256"],
    }
    if json.loads((args.root / "source_receipt.json").read_text(encoding="utf-8")) != expected_receipt:
        raise SystemExit("source receipt mismatch")
    sources = [
        repository / "src" / "interaction_sprint" / "hindsight_pahf_pairs.py",
        repository / "src" / "interaction_sprint" / "pahf_source_audit.py",
        repository / "scripts" / "prepare_hindsight_pahf_external.py",
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
        "decision": "ENDO_PAHF_INPUTS_REPLAY_VERIFIED",
        "counts": {
            name: partitions["invariants"][name]["n"]
            for name in ("learning", "development", "confirmation")
        },
        "paper_green_light": False,
    }, indent=2))


if __name__ == "__main__":
    main()

