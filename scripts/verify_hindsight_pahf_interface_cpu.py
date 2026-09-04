"""Read-only verifier for the EndoPAHF CPU interface rehearsal."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from interaction_sprint.hindsight_pahf_interface import (
    OPTION_LETTERS,
    REHEARSAL_COUNT,
    REHEARSAL_SALT,
    select_rehearsal_rows,
    summarize_scores,
)
from run_hindsight_pahf_interface_cpu import BATCH_SIZE, MODEL_ID, MODEL_REVISION, THREADS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.output_root / "MANIFEST.json").read_text(encoding="utf-8"))
    for name, expected in manifest.items():
        path = args.output_root / name
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"manifest mismatch: {name}")

    expected_spec = {
        "scope": "CPU_REHEARSAL_ONLY_NOT_PAPER_EVIDENCE",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "batch_size": BATCH_SIZE,
        "threads": THREADS,
        "selection_count": REHEARSAL_COUNT,
        "selection_salt": REHEARSAL_SALT,
        "input_manifest_sha256": sha256(args.input_root / "MANIFEST.json"),
        "contexts": ["immediate", "delayed_expression", "delayed_transition"],
        "paper_green_light": False,
    }
    spec = json.loads((args.output_root / "spec.json").read_text(encoding="utf-8"))
    if spec != expected_spec:
        raise SystemExit("spec mismatch")
    development = json.loads((args.input_root / "development.json").read_text(encoding="utf-8"))
    selected = select_rehearsal_rows(development)
    if json.loads((args.output_root / "selected_cases.json").read_text(encoding="utf-8")) != selected:
        raise SystemExit("selected case mismatch")

    repository = Path(__file__).parents[1]
    sources = [
        repository / "src" / "interaction_sprint" / "hindsight_pahf_interface.py",
        repository / "scripts" / "run_hindsight_pahf_interface_cpu.py",
        Path(__file__),
    ]
    expected_sources = {
        str(path.relative_to(repository)).replace("\\", "/"): sha256(path)
        for path in sources
    }
    if json.loads((args.output_root / "source_hashes.json").read_text(encoding="utf-8")) != expected_sources:
        raise SystemExit("source hash mismatch")

    rows = json.loads((args.output_root / "scores.json").read_text(encoding="utf-8"))
    expected_grid = {
        (str(row["id"]), context)
        for row in selected
        for context in ("immediate", "delayed_expression", "delayed_transition")
    }
    if {(str(row["id"]), str(row["context"])) for row in rows} != expected_grid:
        raise SystemExit("score grid mismatch")
    by_id = {str(row["id"]): row for row in selected}
    for row in rows:
        source = by_id[str(row["id"])]
        expected_target = (
            source["old_target"] if row["context"] == "delayed_expression"
            else source["new_target"]
        )
        probabilities = [float(value) for value in row["normalized_choice_probabilities"]]
        target_index = OPTION_LETTERS.index(str(expected_target))
        if (
            row["target"] != expected_target
            or len(probabilities) != 4
            or abs(sum(probabilities) - 1.0) > 1e-5
            or abs(float(row["normalized_target_probability"]) - probabilities[target_index]) > 1e-7
            or bool(row["normalized_correct"]) != (max(range(4), key=probabilities.__getitem__) == target_index)
            or not 0.0 <= float(row["full_vocabulary_choice_mass"]) <= 1.0
        ):
            raise SystemExit(f"invalid score row: {row['id']} {row['context']}")
    expected_result = summarize_scores(rows)
    result = json.loads((args.output_root / "RESULT.json").read_text(encoding="utf-8"))
    for key, value in expected_result.items():
        if result.get(key) != value:
            raise SystemExit(f"result mismatch: {key}")
    if result.get("paper_green_light") is not False:
        raise SystemExit("invalid paper claim")
    print(json.dumps({
        "verified": True,
        "decision": result["decision"],
        "manifest_sha256": sha256(args.output_root / "MANIFEST.json"),
        "contexts": result["contexts"],
    }, indent=2))


if __name__ == "__main__":
    main()
