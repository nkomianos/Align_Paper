"""Read-only replay verifier for the EndoPAHF learnability diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from interaction_sprint.hindsight_pahf_full import build_full_learning_base_partitions
from interaction_sprint.hindsight_pahf_learnability import run_development_learnability_audit
from interaction_sprint.hindsight_pahf_pairs import build_pinned_partitions, load_json_records
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
    audit_pinned_source(args.source_root)
    manifest = json.loads((args.root / "MANIFEST.json").read_text(encoding="utf-8"))
    for name, expected in manifest.items():
        path = args.root / name
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"manifest mismatch: {name}")
    shopping = args.source_root / "data" / "shopping"
    phase1 = load_json_records(shopping / "phase1.json")
    phase2 = load_json_records(shopping / "phase2.json")
    phase3 = load_json_records(shopping / "phase3.json")
    phase4 = load_json_records(shopping / "phase4.json")
    subset = build_pinned_partitions(args.source_root)
    full = build_full_learning_base_partitions(phase1, phase3, phase2, phase4)
    expected_result = run_development_learnability_audit(
        subset["learning"], full["learning"], subset["development"], phase1, phase2
    )
    observed = json.loads((args.root / "RESULT.json").read_text(encoding="utf-8"))
    if observed != expected_result:
        raise SystemExit("result replay mismatch")
    expected_spec = {
        "scope": "RETROSPECTIVE_DEVELOPMENT_ONLY_ASSAY_HEALTH_DIAGNOSTIC",
        "reason": "determine whether the old persistent target is learnable before neural spending",
        "subset_learning_bases": len(subset["learning"]),
        "full_learning_bases": len(full["learning"]),
        "development_bases": len(subset["development"]),
        "confirmation_read": False,
        "thresholds_are_not_confirmatory": True,
        "paper_green_light": False,
    }
    if json.loads((args.root / "spec.json").read_text(encoding="utf-8")) != expected_spec:
        raise SystemExit("spec mismatch")
    repository = Path(__file__).parents[1]
    source_paths = [
        repository / "src" / "interaction_sprint" / "hindsight_pahf_pairs.py",
        repository / "src" / "interaction_sprint" / "hindsight_pahf_balanced.py",
        repository / "src" / "interaction_sprint" / "hindsight_pahf_full.py",
        repository / "src" / "interaction_sprint" / "hindsight_pahf_learnability.py",
        repository / "scripts" / "run_hindsight_pahf_learnability_audit.py",
        Path(__file__),
    ]
    expected_sources = {
        str(path.relative_to(repository)).replace("\\", "/"): sha256(path)
        for path in source_paths
    }
    if json.loads((args.root / "source_hashes.json").read_text(encoding="utf-8")) != expected_sources:
        raise SystemExit("source hash mismatch")
    print(json.dumps({
        "verified": True,
        "decision": observed["decision"],
        "manifest_sha256": sha256(args.root / "MANIFEST.json"),
        "confirmation_opened": False,
        "paper_green_light": False,
    }, indent=2))


if __name__ == "__main__":
    main()
