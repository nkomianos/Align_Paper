"""Prepare label-counterbalanced EndoPAHF inputs without overwriting v1."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from interaction_sprint.hindsight_pahf_balanced import build_balanced_partitions


INPUT_MANIFEST_SHA256 = "ee2d023c5d285022a1f7220e680f57fcf4a3aea76d59e5202426b0b9c13e56e0"


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
    args.root.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).parents[1]

    def read(name: str) -> object:
        return json.loads((args.input_root / name).read_text(encoding="utf-8"))

    def write(name: str, value: object) -> None:
        (args.root / name).write_text(
            json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
        )

    partitions = build_balanced_partitions({
        name: read(f"{name}.json")
        for name in ("learning", "development", "confirmation")
    })
    spec = {
        "scope": "ENDO_PAHF_LABEL_COUNTERBALANCED_INPUTS_ONLY",
        "version": 2,
        "input_manifest_sha256": INPUT_MANIFEST_SHA256,
        "rotations_per_base": 4,
        "selection_changed": False,
        "confirmation_used_for_model_or_threshold_selection": False,
        "paper_green_light": False,
    }
    write("spec.json", spec)
    for name in ("learning", "development", "confirmation", "invariants"):
        write(f"{name}.json", partitions[name])
    sources = [
        repository / "src" / "interaction_sprint" / "hindsight_pahf_balanced.py",
        Path(__file__),
        repository / "scripts" / "verify_hindsight_pahf_external_v2.py",
    ]
    write("source_hashes.json", {
        str(path.relative_to(repository)).replace("\\", "/"): sha256(path)
        for path in sources
    })
    files = [path for path in args.root.iterdir() if path.is_file()]
    write("MANIFEST.json", {path.name: sha256(path) for path in sorted(files)})
    print(json.dumps({
        "decision": "ENDO_PAHF_V2_COUNTERBALANCED_INPUTS_PREPARED",
        "counts": {name: len(partitions[name]) for name in (
            "learning", "development", "confirmation"
        )},
        "paper_green_light": False,
    }, indent=2))


if __name__ == "__main__":
    main()
