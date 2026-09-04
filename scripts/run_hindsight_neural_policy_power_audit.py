"""Run and seal the prospectively frozen model-free G1 power audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from interaction_sprint.hindsight_neural_anchor import build_records
from interaction_sprint.hindsight_neural_policy_g1 import build_disjoint_policy_panels
from interaction_sprint.hindsight_neural_policy_power import (
    EFFECTIVE_GAINS,
    SFT_GAIN_MULTIPLIERS,
    build_power_cells,
    summarize_power_cells,
    target_means,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).parents[1]

    def write(name: str, value: object) -> None:
        (args.root / name).write_text(
            json.dumps(value, indent=2, allow_nan=False), encoding="utf-8",
        )

    rows, _, _ = build_records()
    panels = build_disjoint_policy_panels(rows)
    targets = target_means(rows, panels)
    alternatives, nulls = build_power_cells(targets)
    summary = summarize_power_cells(alternatives, nulls)
    write("spec.json", {
        "version": "policy-g1-rule-power-v2",
        "effective_gains": list(EFFECTIVE_GAINS),
        "sft_gain_multipliers": list(SFT_GAIN_MULTIPLIERS),
        "alternative_requirement": "at least 6 of 9",
        "null_requirement": "0 of 9",
        "paper_green_light": False,
    })
    write("targets.json", targets)
    write("cells.json", {"alternatives": alternatives, "nulls": nulls})
    write("RESULT.json", {**summary, "paper_green_light": False})
    source_paths = [
        repository / "src" / "interaction_sprint" / "hindsight_neural_anchor.py",
        repository / "src" / "interaction_sprint" / "hindsight_neural_policy_g1.py",
        repository / "src" / "interaction_sprint" / "hindsight_neural_policy_power.py",
        Path(__file__),
        repository / "scripts" / "verify_hindsight_neural_policy_power_audit.py",
    ]
    write("source_hashes.json", {
        str(path.relative_to(repository)).replace("\\", "/"): sha256(path)
        for path in source_paths
    })
    files = [path for path in args.root.iterdir() if path.is_file() and path.name != "MANIFEST.json"]
    write("MANIFEST.json", {path.name: sha256(path) for path in sorted(files)})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
