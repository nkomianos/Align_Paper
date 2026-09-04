"""Read-only structural and gradient-metric verifier for neural gradient G0."""
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
    MODEL_ID,
    MODEL_REVISION,
    OFFICIAL_SDPO_COMMIT,
    OFFICIAL_SDPO_REPOSITORY,
    SEED,
    TRAIN_BATCH,
    build_records,
    semantic_letter_index,
)
from interaction_sprint.hindsight_neural_gradient import (
    ANCHORS_PER_ACTION,
    ANCHORS_PER_PANEL,
    PANEL_COUNT,
    build_anchor_panels,
    summarize_gradients,
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
    manifest = json.loads((args.root / "MANIFEST.json").read_text(encoding="utf-8"))
    if not manifest:
        raise SystemExit("empty manifest")
    for name, expected in manifest.items():
        path = args.root / name
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"manifest mismatch: {name}")

    spec = json.loads((args.root / "spec.json").read_text(encoding="utf-8"))
    expected_spec = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "official_sdpo_repository": OFFICIAL_SDPO_REPOSITORY,
        "official_sdpo_commit": OFFICIAL_SDPO_COMMIT,
        "objective": "initial-adapter full-vocabulary first-token reverse-KL gradient estimation",
        "seed": SEED,
        "batch": TRAIN_BATCH,
        "panel_count": PANEL_COUNT,
        "anchors_per_panel": ANCHORS_PER_PANEL,
        "anchors_per_action": ANCHORS_PER_ACTION,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "hindsight_block": HINDSIGHT_BLOCK,
        "gradient_storage_dtype": "bfloat16",
        "paper_green_light": False,
    }
    if spec != expected_spec:
        raise SystemExit("frozen spec mismatch")

    rows, evaluation, _ = build_records()
    cases = json.loads((args.root / "cases.json").read_text(encoding="utf-8"))
    panels = json.loads((args.root / "panels.json").read_text(encoding="utf-8"))
    if cases != {"train": rows, "evaluation": evaluation} or panels != build_anchor_panels(rows):
        raise SystemExit("frozen cases or panels mismatch")

    repository = Path(__file__).parents[1]
    source_paths = [
        repository / "src" / "interaction_sprint" / "hindsight_neural_anchor.py",
        repository / "src" / "interaction_sprint" / "hindsight_neural_gradient.py",
        repository / "src" / "latent_contract" / "sender_update.py",
        repository / "scripts" / "run_hindsight_neural_gradient_g0.py",
        Path(__file__),
    ]
    expected_sources = {
        str(path.relative_to(repository)).replace("\\", "/"): sha256(path)
        for path in source_paths
    }
    if json.loads((args.root / "source_hashes.json").read_text(encoding="utf-8")) != expected_sources:
        raise SystemExit("source hash mismatch against verifier revision")

    qualification = json.loads((args.root / "qualification.json").read_text(encoding="utf-8"))
    qrows = qualification["rows"]
    expected_pairs = {(row["id"], semantic) for row in evaluation for semantic in (0, 1)}
    if {(row.get("id"), row.get("semantic")) for row in qrows} != expected_pairs:
        raise SystemExit("qualification cases mismatch")
    eval_by_id = {row["id"]: row for row in evaluation}
    for row in qrows:
        target = semantic_letter_index(int(row["semantic"]), int(eval_by_id[row["id"]]["swap"]))
        probability = float(row["normalized_target_probability"])
        expected_correct = probability >= .5 if target == 0 else probability > .5
        if row["target_letter"] != "AB"[target] or row["normalized_correct"] != expected_correct:
            raise SystemExit(f"qualification target/correctness mismatch: {row['id']}")
        if not 0 <= probability <= 1 or not 0 <= float(row["ab_mass"]) <= 1 + 1e-6:
            raise SystemExit(f"invalid qualification probability: {row['id']}")
    qmetrics = {
        "n": len(qrows),
        "correct": sum(int(row["normalized_correct"]) for row in qrows),
        "mean_normalized_target_probability": float(np.mean([
            row["normalized_target_probability"] for row in qrows
        ])),
        "min_normalized_target_probability": min(row["normalized_target_probability"] for row in qrows),
        "min_ab_mass": min(row["ab_mass"] for row in qrows),
        "mean_target_log_probability_shift": float(np.mean([
            row["target_log_probability_shift"] for row in qrows
        ])),
    }
    qgates = {
        "all_semantic_choices_correct": qmetrics["correct"] == qmetrics["n"],
        "mean_normalized_target_probability_at_least_point_90":
            qmetrics["mean_normalized_target_probability"] >= .90,
        "min_normalized_target_probability_at_least_point_70":
            qmetrics["min_normalized_target_probability"] >= .70,
        "min_full_vocabulary_ab_mass_at_least_point_20": qmetrics["min_ab_mass"] >= .20,
    }
    if qmetrics != qualification["metrics"] or qgates != qualification["gates"]:
        raise SystemExit("qualification arithmetic mismatch")
    if qualification["qualified"] != all(qgates.values()):
        raise SystemExit("qualification decision mismatch")

    result = json.loads((args.root / "RESULT.json").read_text(encoding="utf-8"))
    if result.get("paper_green_light") is not False:
        raise SystemExit("invalid paper-green-light claim")
    if result["decision"] == "STOP_MODEL_INTERFACE_UNQUALIFIED":
        if qualification["qualified"]:
            raise SystemExit("invalid interface stop")
        print(json.dumps({"verified": True, "decision": result["decision"]}, indent=2))
        return
    if not qualification["qualified"]:
        raise SystemExit("gradient result despite failed qualification")

    layout = json.loads((args.root / "parameter_layout.json").read_text(encoding="utf-8"))
    if not (
        len(layout["names"]) == len(layout["shapes"]) == len(layout["numels"])
        and len(set(layout["names"])) == len(layout["names"])
        and sum(layout["numels"]) == layout["total"] > 0
        and all(int(np.prod(shape)) == numel for shape, numel in zip(layout["shapes"], layout["numels"]))
        and len(layout["answer_token_ids"]) == 2
    ):
        raise SystemExit("invalid parameter layout")
    initial = torch.load(args.root / "initial_adapter.pt", map_location="cpu", weights_only=True)
    if set(initial) != set(layout["names"]):
        raise SystemExit("initial-adapter keys differ from parameter layout")
    shape_by_name = dict(zip(layout["names"], layout["shapes"]))
    for name, tensor in initial.items():
        if list(tensor.shape) != shape_by_name[name] or not bool(torch.isfinite(tensor).all()):
            raise SystemExit(f"invalid initial adapter tensor: {name}")
        if name.endswith(".b") and bool(torch.count_nonzero(tensor)):
            raise SystemExit(f"nonzero initial LoRA B tensor: {name}")
    expected_names = {"oracle_delayed", "raw_immediate"}
    expected_names |= {f"panel_{index:02d}_delayed" for index in range(PANEL_COUNT)}
    expected_names |= {f"panel_{index:02d}_immediate" for index in range(PANEL_COUNT)}
    if result.get("completed_vectors") != [
        "oracle_delayed", "raw_immediate",
        *[name for index in range(PANEL_COUNT)
          for name in (f"panel_{index:02d}_delayed", f"panel_{index:02d}_immediate")],
    ]:
        raise SystemExit("completed-vector order mismatch")
    progress = json.loads((args.root / "gradient_progress.json").read_text(encoding="utf-8"))
    if progress.get("completed") != result["completed_vectors"]:
        raise SystemExit("gradient progress mismatch")
    vectors = {}
    for name in expected_names:
        path = args.root / f"gradient_{name}.pt"
        if not path.is_file():
            raise SystemExit(f"missing gradient: {name}")
        vector = torch.load(path, map_location="cpu", weights_only=True)
        if vector.dtype != torch.bfloat16 or vector.ndim != 1 or vector.numel() != layout["total"]:
            raise SystemExit(f"gradient layout mismatch: {name}")
        vectors[name] = vector
    summary = summarize_gradients(vectors)
    for key in ("decision", "gates", "aggregate", "panels", "scope"):
        if result.get(key) != summary[key]:
            raise SystemExit(f"gradient summary mismatch: {key}")
    print(json.dumps({
        "verified": True,
        "decision": result["decision"],
        "gates": result["gates"],
        "aggregate": result["aggregate"],
    }, indent=2))


if __name__ == "__main__":
    main()
