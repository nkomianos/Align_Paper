"""Run and seal the development-only EndoPAHF learnability diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import sklearn

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
    args.root.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).parents[1]
    audit_pinned_source(args.source_root)
    shopping = args.source_root / "data" / "shopping"
    phase1 = load_json_records(shopping / "phase1.json")
    phase2 = load_json_records(shopping / "phase2.json")
    phase3 = load_json_records(shopping / "phase3.json")
    phase4 = load_json_records(shopping / "phase4.json")
    subset = build_pinned_partitions(args.source_root)
    full = build_full_learning_base_partitions(phase1, phase3, phase2, phase4)
    result = run_development_learnability_audit(
        subset["learning"], full["learning"], subset["development"], phase1, phase2
    )

    def write(name: str, value: object) -> None:
        (args.root / name).write_text(
            json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
        )

    write("spec.json", {
        "scope": "RETROSPECTIVE_DEVELOPMENT_ONLY_ASSAY_HEALTH_DIAGNOSTIC",
        "reason": "determine whether the old persistent target is learnable before neural spending",
        "subset_learning_bases": len(subset["learning"]),
        "full_learning_bases": len(full["learning"]),
        "development_bases": len(subset["development"]),
        "confirmation_read": False,
        "thresholds_are_not_confirmatory": True,
        "paper_green_light": False,
    })
    write("RESULT.json", result)
    write("runtime.json", {"scikit_learn": sklearn.__version__})
    source_paths = [
        repository / "src" / "interaction_sprint" / "hindsight_pahf_pairs.py",
        repository / "src" / "interaction_sprint" / "hindsight_pahf_balanced.py",
        repository / "src" / "interaction_sprint" / "hindsight_pahf_full.py",
        repository / "src" / "interaction_sprint" / "hindsight_pahf_learnability.py",
        Path(__file__),
        repository / "scripts" / "verify_hindsight_pahf_learnability_audit.py",
    ]
    write("source_hashes.json", {
        str(path.relative_to(repository)).replace("\\", "/"): sha256(path)
        for path in source_paths
    })
    files = [path for path in args.root.iterdir() if path.is_file()]
    write("MANIFEST.json", {path.name: sha256(path) for path in sorted(files)})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
