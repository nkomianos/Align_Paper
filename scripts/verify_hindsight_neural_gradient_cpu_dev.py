"""Read-only verifier for the small-model nested-gradient CPU rehearsal."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from interaction_sprint.hindsight_neural_anchor import (
    HINDSIGHT_BLOCK,
    LORA_ALPHA,
    LORA_RANK,
    OFFICIAL_SDPO_COMMIT,
    OFFICIAL_SDPO_REPOSITORY,
    SEED,
    build_records,
    semantic_letter_index,
)
from interaction_sprint.hindsight_neural_gradient import PANEL_COUNT
from interaction_sprint.hindsight_neural_gradient_v2 import (
    ANCHOR_BUDGETS,
    build_nested_anchor_panels,
    summarize_nested_gradients,
)
from run_hindsight_neural_gradient_cpu_dev import BATCH, MODEL_ID, MODEL_REVISION, THREADS


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
    manifest = json.loads((args.root / "MANIFEST.json").read_text(encoding="utf-8"))
    for name, expected in manifest.items():
        if not (args.root / name).is_file() or sha256(args.root / name) != expected:
            raise SystemExit(f"manifest mismatch: {name}")

    expected_spec = {
        "scope": "CPU_REHEARSAL_ONLY_NOT_NEURAL_G0",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "official_sdpo_repository": OFFICIAL_SDPO_REPOSITORY,
        "official_sdpo_commit": OFFICIAL_SDPO_COMMIT,
        "seed": SEED,
        "batch": BATCH,
        "threads": THREADS,
        "anchor_budgets": list(ANCHOR_BUDGETS),
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "hindsight_block": HINDSIGHT_BLOCK,
        "gradient_storage_dtype": "bfloat16",
        "paper_green_light": False,
    }
    if json.loads((args.root / "spec.json").read_text(encoding="utf-8")) != expected_spec:
        raise SystemExit("spec mismatch")
    repository = Path(__file__).parents[1]
    sources = [
        repository / "src" / "interaction_sprint" / "hindsight_neural_anchor.py",
        repository / "src" / "interaction_sprint" / "hindsight_neural_gradient.py",
        repository / "src" / "interaction_sprint" / "hindsight_neural_gradient_v2.py",
        repository / "src" / "latent_contract" / "sender_update.py",
        repository / "scripts" / "run_hindsight_neural_gradient_cpu_dev.py",
        Path(__file__),
    ]
    expected_sources = {
        str(path.relative_to(repository)).replace("\\", "/"): sha256(path) for path in sources
    }
    if json.loads((args.root / "source_hashes.json").read_text(encoding="utf-8")) != expected_sources:
        raise SystemExit("source hash mismatch")
    rows, evaluation, _ = build_records()
    if json.loads((args.root / "cases.json").read_text(encoding="utf-8")) != {
        "train": rows, "evaluation": evaluation,
    }:
        raise SystemExit("case mismatch")
    if json.loads((args.root / "panels.json").read_text(encoding="utf-8")) != {
        str(budget): panels for budget, panels in build_nested_anchor_panels(rows).items()
    }:
        raise SystemExit("panel mismatch")

    qualification = json.loads((args.root / "qualification.json").read_text(encoding="utf-8"))
    qrows = qualification["rows"]
    expected_pairs = {(row["id"], semantic) for row in evaluation for semantic in (0, 1)}
    if {(row.get("id"), row.get("semantic")) for row in qrows} != expected_pairs:
        raise SystemExit("qualification grid mismatch")
    by_id = {row["id"]: row for row in evaluation}
    for row in qrows:
        target = semantic_letter_index(int(row["semantic"]), int(by_id[row["id"]]["swap"]))
        probability = float(row["normalized_target_probability"])
        expected_correct = probability >= .5 if target == 0 else probability > .5
        if row["target_letter"] != "AB"[target] or row["normalized_correct"] != expected_correct:
            raise SystemExit("qualification row mismatch")
    qmetrics = {
        "n": len(qrows),
        "correct": sum(int(row["normalized_correct"]) for row in qrows),
        "mean_normalized_target_probability": float(np.mean([
            row["normalized_target_probability"] for row in qrows
        ])),
        "min_normalized_target_probability": min(row["normalized_target_probability"] for row in qrows),
        "min_ab_mass": min(row["ab_mass"] for row in qrows),
    }
    qgates = {
        "all_semantic_choices_correct": qmetrics["correct"] == qmetrics["n"],
        "mean_normalized_target_probability_at_least_point_90":
            qmetrics["mean_normalized_target_probability"] >= .90,
        "min_normalized_target_probability_at_least_point_70":
            qmetrics["min_normalized_target_probability"] >= .70,
        "min_full_vocabulary_ab_mass_at_least_point_20": qmetrics["min_ab_mass"] >= .20,
    }
    if qualification["metrics"] != qmetrics or qualification["gates"] != qgates:
        raise SystemExit("qualification summary mismatch")
    qualified = all(qgates.values())
    if qualification["qualified"] != qualified:
        raise SystemExit("qualification decision mismatch")
    result = json.loads((args.root / "RESULT.json").read_text(encoding="utf-8"))
    if result.get("paper_green_light") is not False:
        raise SystemExit("invalid green-light claim")
    if not qualified:
        if result["decision"] != "CPU_REHEARSAL_ONLY_INTERFACE_UNQUALIFIED":
            raise SystemExit("invalid interface stop")
        print(json.dumps({"verified": True, "decision": result["decision"]}, indent=2))
        return

    layout = json.loads((args.root / "parameter_layout.json").read_text(encoding="utf-8"))
    if not (
        len(layout["names"]) == len(layout["shapes"]) == len(layout["numels"])
        and sum(layout["numels"]) == layout["total"] > 0
        and all(int(np.prod(shape)) == numel for shape, numel in zip(layout["shapes"], layout["numels"]))
    ):
        raise SystemExit("parameter layout mismatch")
    initial = torch.load(args.root / "initial_adapter.pt", map_location="cpu", weights_only=True)
    if set(initial) != set(layout["names"]):
        raise SystemExit("initial adapter mismatch")
    expected_order = ["oracle_delayed", "raw_immediate"] + [
        f"budget_{budget:02d}_panel_{panel:02d}_{field}"
        for budget in ANCHOR_BUDGETS for panel in range(PANEL_COUNT)
        for field in ("delayed", "immediate")
    ]
    if result.get("completed_vectors") != expected_order:
        raise SystemExit("completed gradient order mismatch")
    if json.loads((args.root / "gradient_progress.json").read_text(encoding="utf-8"))["completed"] != expected_order:
        raise SystemExit("gradient progress mismatch")
    vectors = {}
    for name in expected_order:
        vector = torch.load(args.root / f"gradient_{name}.pt", map_location="cpu", weights_only=True)
        if vector.dtype != torch.bfloat16 or vector.ndim != 1 or vector.numel() != layout["total"]:
            raise SystemExit(f"gradient layout mismatch: {name}")
        vectors[name] = vector
    summary = summarize_nested_gradients(vectors)
    expected_decision = f"CPU_REHEARSAL_ONLY_{summary['decision']}"
    if result["decision"] != expected_decision:
        raise SystemExit("decision mismatch")
    for key in ("gates", "budgets", "scope"):
        if result.get(key) != summary[key]:
            raise SystemExit(f"summary mismatch: {key}")
    print(json.dumps({
        "verified": True,
        "decision": result["decision"],
        "manifest_sha256": sha256(args.root / "MANIFEST.json"),
        "gates": result["gates"],
    }, indent=2))


if __name__ == "__main__":
    main()
