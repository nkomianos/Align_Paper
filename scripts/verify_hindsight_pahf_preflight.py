"""Read-only verifier for the capable-model EndoPAHF preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from interaction_sprint.hindsight_neural_anchor import MODEL_ID, MODEL_REVISION
from interaction_sprint.hindsight_pahf_interface import OPTION_LETTERS
from interaction_sprint.hindsight_pahf_preflight import (
    BASE_COUNT, CONTEXTS, SELECTION_SALT, select_base_panel, summarize_preflight,
)
from run_hindsight_pahf_preflight import BATCH_SIZE, INPUT_MANIFEST_SHA256


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
        raise SystemExit("input manifest mismatch")
    manifest = json.loads((args.root / "MANIFEST.json").read_text(encoding="utf-8"))
    for name, expected in manifest.items():
        path = args.root / name
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"manifest mismatch: {name}")
    expected_spec = {
        "scope": "CAPABLE_MODEL_INTERFACE_PREFLIGHT_NOT_PAPER_EVIDENCE",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "input_manifest_sha256": INPUT_MANIFEST_SHA256,
        "base_count": BASE_COUNT,
        "variants_per_base": 4,
        "contexts": list(CONTEXTS),
        "selection_salt": SELECTION_SALT,
        "batch_size": BATCH_SIZE,
        "confirmation_opened": False,
        "paper_green_light": False,
    }
    if json.loads((args.root / "spec.json").read_text(encoding="utf-8")) != expected_spec:
        raise SystemExit("spec mismatch")
    development = json.loads((args.input_root / "development.json").read_text(encoding="utf-8"))
    selected = select_base_panel(development)
    if json.loads((args.root / "selected_cases.json").read_text(encoding="utf-8")) != selected:
        raise SystemExit("selected cases mismatch")
    rows = json.loads((args.root / "scores.json").read_text(encoding="utf-8"))
    expected_grid = {
        (str(row["id"]), context)
        for row in selected for context in CONTEXTS
    }
    if {(str(row["id"]), str(row["context"])) for row in rows} != expected_grid:
        raise SystemExit("score grid mismatch")
    by_id = {str(row["id"]): row for row in selected}
    for row in rows:
        source = by_id[str(row["id"])]
        target = source["new_target"] if row["context"] == "immediate" else source["old_target"]
        probabilities = [float(value) for value in row["normalized_choice_probabilities"]]
        target_index = OPTION_LETTERS.index(str(target))
        if (
            row["target"] != target
            or len(probabilities) != 4
            or abs(sum(probabilities) - 1.0) > 1e-5
            or bool(row["normalized_correct"]) != (max(range(4), key=probabilities.__getitem__) == target_index)
            or abs(float(row["normalized_target_probability"]) - probabilities[target_index]) > 1e-7
            or not 0 <= float(row["full_vocabulary_choice_mass"]) <= 1
        ):
            raise SystemExit(f"invalid score row: {row['id']} {row['context']}")
    expected_result = summarize_preflight(rows)
    result = json.loads((args.root / "RESULT.json").read_text(encoding="utf-8"))
    for key, value in expected_result.items():
        if result.get(key) != value:
            raise SystemExit(f"result mismatch: {key}")
    repository = Path(__file__).parents[1]
    sources = [
        repository / "src" / "interaction_sprint" / "hindsight_pahf_interface.py",
        repository / "src" / "interaction_sprint" / "hindsight_pahf_preflight.py",
        repository / "scripts" / "run_hindsight_pahf_preflight.py",
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
        "decision": result["decision"],
        "manifest_sha256": sha256(args.root / "MANIFEST.json"),
        "gates": result["gates"],
        "paper_green_light": False,
    }, indent=2))


if __name__ == "__main__":
    main()
