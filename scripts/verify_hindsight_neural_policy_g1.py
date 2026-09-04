"""Read-only verifier for the prospectively frozen Hindsight neural policy G1."""
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
from interaction_sprint.hindsight_neural_policy_g1 import (
    POLICY_ANCHORS_PER_ACTION,
    POLICY_ANCHORS_PER_PANEL,
    POLICY_BATCH,
    POLICY_LEARNING_RATE,
    POLICY_PANEL_COUNT,
    POLICY_SEED,
    POLICY_STEPS,
    build_disjoint_policy_panels,
    build_policy_schedules,
    expected_policy_arm_names,
    summarize_evaluation_rows,
    summarize_policy_controls,
    summarize_policy_endpoints,
)
from run_hindsight_neural_policy_g1 import POLICY_RULE_POWER_AUDIT_MANIFEST_SHA256


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
    repository = Path(__file__).parents[1]

    manifest = json.loads((args.root / "MANIFEST.json").read_text(encoding="utf-8"))
    if not manifest:
        raise SystemExit("empty manifest")
    for name, expected in manifest.items():
        path = args.root / name
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"manifest mismatch: {name}")

    prerequisite = json.loads((args.root / "gradient_prerequisite.json").read_text(encoding="utf-8"))
    if (
        set(prerequisite) != {
            "gradient_root_at_launch", "manifest_sha256", "result_sha256",
            "decision", "committed_verifier_receipt",
        }
        or prerequisite["decision"] != "NEURAL_GRADIENT_G0_V2_QUALIFIED"
        or prerequisite["committed_verifier_receipt"].get("verified") is not True
        or prerequisite["committed_verifier_receipt"].get("decision")
        != "NEURAL_GRADIENT_G0_V2_QUALIFIED"
        or len(prerequisite["manifest_sha256"]) != 64
        or len(prerequisite["result_sha256"]) != 64
    ):
        raise SystemExit("invalid gradient prerequisite receipt")

    spec = json.loads((args.root / "spec.json").read_text(encoding="utf-8"))
    expected_spec = {
        "version": "policy-learning-g1-v4-separate-baselines",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "official_sdpo_repository": OFFICIAL_SDPO_REPOSITORY,
        "official_sdpo_commit": OFFICIAL_SDPO_COMMIT,
        "objective": "full-vocabulary reverse KL(student(.|x) || stopgrad teacher(.|x,o)) at first answer token",
        "data_seed": SEED,
        "policy_seed": POLICY_SEED,
        "steps_per_arm": POLICY_STEPS,
        "batch": POLICY_BATCH,
        "panel_count": POLICY_PANEL_COUNT,
        "anchors_per_panel": POLICY_ANCHORS_PER_PANEL,
        "anchors_per_action": POLICY_ANCHORS_PER_ACTION,
        "learning_rate": POLICY_LEARNING_RATE,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "hindsight_block": HINDSIGHT_BLOCK,
        "trained_arms": 2 + 3 * POLICY_PANEL_COUNT,
        "required_gradient_manifest_sha256": prerequisite["manifest_sha256"],
        "policy_rule_power_audit_manifest_sha256": POLICY_RULE_POWER_AUDIT_MANIFEST_SHA256,
        "paper_green_light": False,
    }
    if spec != expected_spec:
        raise SystemExit("frozen spec mismatch")

    rows, evaluation, _ = build_records()
    panels = build_disjoint_policy_panels(rows)
    schedules = build_policy_schedules(rows, panels)
    if json.loads((args.root / "cases.json").read_text(encoding="utf-8")) != {
        "train": rows, "evaluation": evaluation,
    }:
        raise SystemExit("frozen cases mismatch")
    if json.loads((args.root / "panels.json").read_text(encoding="utf-8")) != panels:
        raise SystemExit("frozen panels mismatch")
    if json.loads((args.root / "schedules.json").read_text(encoding="utf-8")) != schedules:
        raise SystemExit("frozen schedules mismatch")

    source_paths = [
        repository / "src" / "interaction_sprint" / "hindsight_neural_anchor.py",
        repository / "src" / "interaction_sprint" / "hindsight_neural_gradient.py",
        repository / "src" / "interaction_sprint" / "hindsight_neural_gradient_v2.py",
        repository / "src" / "interaction_sprint" / "hindsight_neural_policy_g1.py",
        repository / "src" / "latent_contract" / "sender_update.py",
        repository / "scripts" / "verify_hindsight_neural_gradient_g0_v2.py",
        repository / "scripts" / "run_hindsight_neural_policy_g1.py",
        Path(__file__),
        repository / "scripts" / "run_hindsight_neural_policy_g1_remote.sh",
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
    if result.get("paper_green_light") is not False or result.get("gradient_prerequisite") != prerequisite:
        raise SystemExit("invalid scope or prerequisite claim")
    if result["decision"] == "STOP_MODEL_INTERFACE_UNQUALIFIED":
        if qualification["qualified"]:
            raise SystemExit("invalid interface stop")
        print(json.dumps({"verified": True, "decision": result["decision"]}, indent=2))
        return
    if not qualification["qualified"]:
        raise SystemExit("policy result despite failed qualification")

    setup = json.loads((args.root / "runtime_setup.json").read_text(encoding="utf-8"))
    if (
        not setup.get("lora_modules")
        or int(setup.get("trainable_parameters", 0)) <= 0
        or int(setup.get("total_parameters", 0)) <= int(setup["trainable_parameters"])
        or len(setup.get("answer_token_ids", [])) != 2
    ):
        raise SystemExit("invalid runtime setup")

    all_names = expected_policy_arm_names()
    if result["decision"] == "STOP_POLICY_ACQUISITION_UNQUALIFIED":
        expected_names = all_names[:3]
    else:
        expected_names = all_names
    expected_completed = expected_names[1:]
    if result.get("completed_arms") != expected_completed:
        raise SystemExit("completed-arm order mismatch")
    progress = json.loads((args.root / "training_progress.json").read_text(encoding="utf-8"))
    if progress != {"completed": expected_completed}:
        raise SystemExit("training progress mismatch")

    initial = torch.load(args.root / "initial_adapter.pt", map_location="cpu", weights_only=True)
    if not initial or any(not bool(torch.isfinite(tensor).all()) for tensor in initial.values()):
        raise SystemExit("invalid initial adapter")
    metrics: dict[str, dict[str, float]] = {}
    for name in expected_names:
        eval_rows = json.loads((args.root / f"{name}_eval.json").read_text(encoding="utf-8"))
        metrics[name] = summarize_evaluation_rows(eval_rows)
        if name == "baseline":
            continue
        step_rows = json.loads((args.root / f"{name}_steps.json").read_text(encoding="utf-8"))
        if len(step_rows) != POLICY_STEPS or [row["step"] for row in step_rows] != list(range(1, POLICY_STEPS + 1)):
            raise SystemExit(f"invalid step log: {name}")
        expected_schedule = schedules["global"]
        expected_anchor_ids = []
        if name not in {"raw_immediate", "oracle_delayed"}:
            panel_index = int(name.split("_")[1])
            expected_anchor_ids = schedules["panels"][str(panel_index)]["anchor_ids"]
        if [row["population_batch_ids"] for row in step_rows] != expected_schedule:
            raise SystemExit(f"step schedule mismatch: {name}")
        if any(row["anchor_ids"] != expected_anchor_ids for row in step_rows):
            raise SystemExit(f"anchor schedule mismatch: {name}")
        adapter = torch.load(args.root / f"{name}_adapter.pt", map_location="cpu", weights_only=True)
        if set(adapter) != set(initial):
            raise SystemExit(f"adapter keys mismatch: {name}")
        for key, tensor in adapter.items():
            if tensor.shape != initial[key].shape or not bool(torch.isfinite(tensor).all()):
                raise SystemExit(f"invalid adapter tensor: {name}/{key}")
        optimizer = torch.load(args.root / f"{name}_optimizer.pt", map_location="cpu", weights_only=True)
        if not isinstance(optimizer, dict) or "state" not in optimizer or "param_groups" not in optimizer:
            raise SystemExit(f"invalid optimizer state: {name}")

    if metrics != result.get("endpoint_metrics"):
        raise SystemExit("endpoint metric mismatch")
    if result["decision"] == "STOP_POLICY_ACQUISITION_UNQUALIFIED":
        summary = summarize_policy_controls(metrics)
        summary["scope"] = (
            "Raw/full-oracle policy acquisition control stopped before sparse-panel "
            "training; this is an assay stop, not a correction result."
        )
        summary_keys = ("decision", "gates", "aggregate", "scope")
    else:
        summary = summarize_policy_endpoints(metrics)
        summary_keys = ("decision", "gates", "aggregate", "panels", "scope")
    for key in summary_keys:
        if result.get(key) != summary[key]:
            raise SystemExit(f"policy summary mismatch: {key}")
    print(json.dumps({
        "verified": True,
        "decision": result["decision"],
        "gates": result["gates"],
        "aggregate": result["aggregate"],
    }, indent=2))


if __name__ == "__main__":
    main()
