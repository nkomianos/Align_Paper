"""Read-only replay verifier for the EndoPAHF/SLIFT G2b preparation."""
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
from prepare_hindsight_slift_g2b import INPUT_MANIFEST_SHA256, load_records


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.input_root / "MANIFEST.json") != INPUT_MANIFEST_SHA256:
        raise SystemExit("EndoPAHF v3 input manifest mismatch")
    manifest = json.loads((args.root / "MANIFEST.json").read_text(encoding="utf-8"))
    for name, expected in manifest.items():
        path = args.root / name
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"manifest mismatch: {name}")
    payloads, metadata = build_slift_g2b_payloads(
        load_records(args.input_root / "learning.json"),
        load_records(args.input_root / "development.json"),
    )
    for name, expected in payloads.items():
        if (args.root / name).read_bytes() != expected:
            raise SystemExit(f"replay mismatch: {name}")
    expected_spec = {
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
    if json.loads((args.root / "spec.json").read_text(encoding="utf-8")) != expected_spec:
        raise SystemExit("spec mismatch")
    repository = Path(__file__).parents[1]
    sources = [
        repository / "src" / "interaction_sprint" / "hindsight_pahf_g2_v2.py",
        repository / "src" / "interaction_sprint" / "hindsight_slift_baseline.py",
        repository / "scripts" / "prepare_hindsight_slift_g2b.py",
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
        "decision": "ENDO_PAHF_SLIFT_G2B_INPUTS_REPLAY_VERIFIED",
        "manifest_sha256": sha256(args.root / "MANIFEST.json"),
        **{key: metadata[key] for key in (
            "learning_rows", "learning_unique_bases", "development_rows",
            "expression_transition_raw_sha256_equal", "confirmation_opened",
            "persistent_target_visible_to_algorithm", "paper_green_light",
        )},
    }, indent=2))


if __name__ == "__main__":
    main()
