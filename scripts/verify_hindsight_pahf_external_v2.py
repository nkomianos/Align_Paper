"""Read-only replay verifier for label-counterbalanced EndoPAHF inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from interaction_sprint.hindsight_pahf_balanced import build_balanced_partitions
from prepare_hindsight_pahf_external_v2 import INPUT_MANIFEST_SHA256


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
        raise SystemExit("v1 input manifest mismatch")
    manifest = json.loads((args.root / "MANIFEST.json").read_text(encoding="utf-8"))
    for name, expected in manifest.items():
        path = args.root / name
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"manifest mismatch: {name}")
    expected_spec = {
        "scope": "ENDO_PAHF_LABEL_COUNTERBALANCED_INPUTS_ONLY",
        "version": 2,
        "input_manifest_sha256": INPUT_MANIFEST_SHA256,
        "rotations_per_base": 4,
        "selection_changed": False,
        "confirmation_used_for_model_or_threshold_selection": False,
        "paper_green_light": False,
    }
    if json.loads((args.root / "spec.json").read_text(encoding="utf-8")) != expected_spec:
        raise SystemExit("spec mismatch")
    source_partitions = {
        name: json.loads((args.input_root / f"{name}.json").read_text(encoding="utf-8"))
        for name in ("learning", "development", "confirmation")
    }
    expected = build_balanced_partitions(source_partitions)
    for name in ("learning", "development", "confirmation", "invariants"):
        actual = json.loads((args.root / f"{name}.json").read_text(encoding="utf-8"))
        if actual != expected[name]:
            raise SystemExit(f"replay mismatch: {name}")
    for name in ("learning", "development", "confirmation"):
        report = expected["invariants"][name]
        base = report["base_records"]
        if not (
            report["n"] == 4 * base
            and report["unique_ids"] == report["n"]
            and report["four_rotations_per_base"]
            and set(report["old_target_counts"].values()) == {base}
            and set(report["new_target_counts"].values()) == {base}
            and report["ordinary_log_world_mismatches"] == 0
            and report["expression_transition_probe_differences"] == report["n"]
        ):
            raise SystemExit(f"counterbalance invariant failure: {name}")
    repository = Path(__file__).parents[1]
    sources = [
        repository / "src" / "interaction_sprint" / "hindsight_pahf_balanced.py",
        repository / "scripts" / "prepare_hindsight_pahf_external_v2.py",
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
        "decision": "ENDO_PAHF_V2_COUNTERBALANCED_INPUTS_REPLAY_VERIFIED",
        "manifest_sha256": sha256(args.root / "MANIFEST.json"),
        "counts": {name: len(expected[name]) for name in (
            "learning", "development", "confirmation"
        )},
        "paper_green_light": False,
    }, indent=2))


if __name__ == "__main__":
    main()
