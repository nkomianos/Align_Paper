"""Prepare target-blind EndoPAHF inputs for an official SLIFT comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from interaction_sprint.hindsight_slift_baseline import (
    SLIFT_RELEASE_LAST_UPDATE_UTC,
    SLIFT_RELEASE_PAGE,
    SLIFT_RELEASE_URL,
    SLIFT_RELEASE_TREE_SHA256,
    build_slift_g2b_payloads,
)


INPUT_MANIFEST_SHA256 = "2ae32c119087d97de6f2e5a65959b4c94a7d81362732871ff806b157e000fab2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records(path: Path) -> list[dict[str, object]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"expected a list of objects: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.input_root / "MANIFEST.json") != INPUT_MANIFEST_SHA256:
        raise SystemExit("EndoPAHF v3 input manifest mismatch")
    args.root.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).parents[1]
    payloads, metadata = build_slift_g2b_payloads(
        load_records(args.input_root / "learning.json"),
        load_records(args.input_root / "development.json"),
    )
    # Intentionally do not read confirmation.json in this preparation.
    for name, payload in payloads.items():
        (args.root / name).write_bytes(payload)
    spec = {
        **metadata,
        "scope": "SLIFT_BASELINE_INPUTS_AND_ROLE_SENSITIVITY_ONLY",
        "input_manifest_sha256": INPUT_MANIFEST_SHA256,
        "selection": "EndoPAHF G2 v2 outcome-blind global schedule",
        "official_inferred_role_arm": "learning_expression.jsonl",
        "fixed_role_sensitivity_arms": ["FIX", "SPEC"],
        "fixed_roles_selected_from_outcomes": False,
        "official_slift": {
            "release_page": SLIFT_RELEASE_PAGE,
            "archive_url": SLIFT_RELEASE_URL,
            "content_tree_sha256": SLIFT_RELEASE_TREE_SHA256,
            "archive_container_is_regenerated": True,
            "reported_last_update_utc": SLIFT_RELEASE_LAST_UPDATE_UTC,
            "snapshot_committed_here": False,
            "reason_not_vendored": "the released snapshot contains no license file",
        },
    }
    (args.root / "spec.json").write_text(
        json.dumps(spec, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sources = [
        repository / "src" / "interaction_sprint" / "hindsight_pahf_g2_v2.py",
        repository / "src" / "interaction_sprint" / "hindsight_slift_baseline.py",
        Path(__file__),
        repository / "scripts" / "verify_hindsight_slift_g2b.py",
    ]
    (args.root / "source_hashes.json").write_text(
        json.dumps({
            str(path.relative_to(repository)).replace("\\", "/"): sha256(path)
            for path in sources
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    files = [path for path in args.root.iterdir() if path.is_file()]
    (args.root / "MANIFEST.json").write_text(
        json.dumps({path.name: sha256(path) for path in sorted(files)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
