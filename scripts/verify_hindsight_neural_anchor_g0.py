"""Read-only structural verifier for neural delayed-anchor SDPO evidence.

This verifies saved hashes, frozen data/schedule, metric arithmetic and declared
gates.  It does not replay neural forward passes or optimizer updates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from interaction_sprint.hindsight_neural_anchor import (
    ANCHORS_PER_BATCH,
    HINDSIGHT_BLOCK,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_RANK,
    MODEL_ID,
    MODEL_REVISION,
    OFFICIAL_SDPO_COMMIT,
    OFFICIAL_SDPO_REPOSITORY,
    SEED,
    TRAIN_BATCH,
    TRAIN_STEPS,
    build_records,
    record_invariants,
    semantic_letter_index,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluation_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    metrics = {}
    for semantic in (0, 1):
        probabilities = np.asarray([row["semantic_probabilities"][semantic] for row in rows], dtype=float)
        metrics[str(semantic)] = {
            "accuracy": float(np.mean([row["predicted_semantic"] == semantic for row in rows])),
            "mean_probability": float(np.mean(probabilities)),
            "mean_nll": float(np.mean(-np.log(np.maximum(probabilities, 1e-12)))),
            "min_ab_mass": min(row["ab_mass"] for row in rows),
        }
    return metrics


def validate_evaluation_rows(
    rows: list[dict[str, object]], expected: list[dict[str, object]], name: str,
) -> None:
    if len(rows) != len(expected):
        raise SystemExit(f"evaluation row count mismatch: {name}")
    expected_by_id = {row["id"]: row for row in expected}
    if set(row.get("id") for row in rows) != set(expected_by_id):
        raise SystemExit(f"evaluation ids mismatch: {name}")
    for row in rows:
        reference = expected_by_id[row["id"]]
        if row.get("swap") != reference["swap"]:
            raise SystemExit(f"evaluation swap mismatch: {name}/{row['id']}")
        probabilities = np.asarray(row.get("semantic_probabilities"), dtype=float)
        if probabilities.shape != (2,) or not np.isfinite(probabilities).all():
            raise SystemExit(f"invalid probabilities: {name}/{row['id']}")
        if abs(float(probabilities.sum()) - 1.) > 1e-6 or (probabilities < 0).any():
            raise SystemExit(f"unnormalized probabilities: {name}/{row['id']}")
        if row.get("predicted_semantic") != int(np.argmax(probabilities)):
            raise SystemExit(f"prediction mismatch: {name}/{row['id']}")
        mass = float(row.get("ab_mass"))
        if not np.isfinite(mass) or not 0. <= mass <= 1. + 1e-6:
            raise SystemExit(f"invalid A/B mass: {name}/{row['id']}")


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
        "objective": "full-vocabulary reverse KL(student(.|x) || stopgrad teacher(.|x,o)) at first answer token",
        "seed": SEED,
        "steps": TRAIN_STEPS,
        "batch": TRAIN_BATCH,
        "anchors_per_batch": ANCHORS_PER_BATCH,
        "learning_rate": LEARNING_RATE,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "hindsight_block": HINDSIGHT_BLOCK,
        "paper_green_light": False,
    }
    for key, expected in expected_spec.items():
        if spec.get(key) != expected:
            raise SystemExit(f"spec mismatch: {key}")

    rows, evaluation, schedule = build_records()
    observed_cases = json.loads((args.root / "cases.json").read_text(encoding="utf-8"))
    observed_schedule = json.loads((args.root / "schedule.json").read_text(encoding="utf-8"))
    if observed_cases != {"train": rows, "evaluation": evaluation} or observed_schedule != schedule:
        raise SystemExit("frozen data or schedule mismatch")
    observed_invariants = json.loads((args.root / "invariants.json").read_text(encoding="utf-8"))
    # JSON stringifies integer dictionary keys in the two marginal counters.
    expected_invariants = json.loads(json.dumps(record_invariants(rows)))
    if observed_invariants != expected_invariants:
        raise SystemExit("data invariant mismatch")

    repository = Path(__file__).parents[1]
    dependencies = [
        repository / "src" / "interaction_sprint" / "hindsight_neural_anchor.py",
        repository / "src" / "latent_contract" / "sender_update.py",
    ]
    expected_dependency_hashes = {
        str(path.relative_to(repository)).replace("\\", "/"): sha256(path) for path in dependencies
    }
    observed_dependency_hashes = json.loads((args.root / "dependency_hashes.json").read_text(encoding="utf-8"))
    if observed_dependency_hashes != expected_dependency_hashes:
        raise SystemExit("dependency hash mismatch against verifier revision")
    if sha256(args.root / "runner_source.py") != sha256(repository / "scripts" / "run_hindsight_neural_anchor_g0.py"):
        raise SystemExit("runner source mismatch against verifier revision")

    qualification = json.loads((args.root / "qualification.json").read_text(encoding="utf-8"))
    qrows = qualification["rows"]
    expected_pairs = {(row["id"], semantic) for row in evaluation for semantic in (0, 1)}
    if {(row.get("id"), row.get("semantic")) for row in qrows} != expected_pairs:
        raise SystemExit("qualification cases mismatch")
    eval_by_id = {row["id"]: row for row in evaluation}
    for row in qrows:
        target_index = semantic_letter_index(int(row["semantic"]), int(eval_by_id[row["id"]]["swap"]))
        probability = float(row["normalized_target_probability"])
        if row.get("target_letter") != "AB"[target_index]:
            raise SystemExit(f"qualification target mismatch: {row['id']}")
        expected_correct = probability >= .5 if target_index == 0 else probability > .5
        if row.get("normalized_correct") != expected_correct:
            raise SystemExit(f"qualification correctness mismatch: {row['id']}")
        if not 0. <= probability <= 1. or not 0. <= float(row["ab_mass"]) <= 1. + 1e-6:
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

    baseline_rows = json.loads((args.root / "baseline_eval.json").read_text(encoding="utf-8"))
    validate_evaluation_rows(baseline_rows, evaluation, "baseline")
    baseline = evaluation_metrics(baseline_rows)
    saved_baseline = json.loads((args.root / "baseline_metrics.json").read_text(encoding="utf-8"))
    if baseline != saved_baseline:
        raise SystemExit("saved baseline metric mismatch")
    if baseline != result["baseline"]:
        raise SystemExit("baseline metric mismatch")

    arms = result["arms"]
    for name, metrics in arms.items():
        rows_path = args.root / f"{name}_eval.json"
        steps_path = args.root / f"{name}_steps.json"
        adapter_path = args.root / f"{name}_adapter.pt"
        optimizer_path = args.root / f"{name}_optimizer.pt"
        if not all(path.is_file() for path in (rows_path, steps_path, adapter_path, optimizer_path)):
            raise SystemExit(f"missing completed-arm artifact: {name}")
        arm_evaluation_rows = json.loads(rows_path.read_text(encoding="utf-8"))
        validate_evaluation_rows(arm_evaluation_rows, evaluation, name)
        if evaluation_metrics(arm_evaluation_rows) != metrics:
            raise SystemExit(f"evaluation arithmetic mismatch: {name}")
        steps = json.loads(steps_path.read_text(encoding="utf-8"))
        if len(steps) != TRAIN_STEPS:
            raise SystemExit(f"step count mismatch: {name}")
        for expected_index, (step, batch) in enumerate(zip(steps, schedule), 1):
            if step["step"] != expected_index or step["batch_ids"] != batch:
                raise SystemExit(f"schedule mismatch: {name}/{expected_index}")
            if "anchor_residual_mean" in step:
                expected_loss = step["immediate_mean"] + step["anchor_residual_mean"]
                if abs(step["loss"] - expected_loss) > 2e-6:
                    raise SystemExit(f"augmented loss mismatch: {name}/{expected_index}")
            if "transition_augmented_identity_error" in step and step["transition_augmented_identity_error"] > 1e-6:
                raise SystemExit(f"transition identity mismatch: {name}/{expected_index}")
            if not np.isfinite(float(step["loss"])) or not np.isfinite(float(step["gradient_norm"])):
                raise SystemExit(f"nonfinite training diagnostic: {name}/{expected_index}")

    if result["decision"] == "STOP_TRUTHFUL_SDPO_ACQUISITION_UNQUALIFIED":
        if set(arms) != {"truthful_raw"}:
            raise SystemExit("invalid acquisition-stop arm set")
        truthful = arms["truthful_raw"]["1"]
        if truthful["accuracy"] >= .80 and truthful["mean_probability"] - baseline["1"]["mean_probability"] >= .10:
            raise SystemExit("acquisition stop despite passing positive control")
        print(json.dumps({"verified": True, "decision": result["decision"]}, indent=2))
        return

    required = {
        "truthful_raw", "raw_shared", "expression_anchor_sdpo",
        "expression_anchor_sft", "expression_augmented",
        "transition_anchor_sdpo", "transition_anchor_sft",
    }
    if set(arms) != required:
        raise SystemExit("completed comparison arm set mismatch")
    if result.get("completed_arms") != [
        "truthful_raw", "raw_shared", "expression_anchor_sdpo",
        "expression_anchor_sft", "expression_augmented",
        "transition_anchor_sdpo", "transition_anchor_sft",
    ]:
        raise SystemExit("completed-arm order mismatch")
    truthful = arms["truthful_raw"]["1"]
    raw_zero = arms["raw_shared"]["0"]
    expression_augmented = arms["expression_augmented"]["1"]
    expression_baselines = [arms["expression_anchor_sdpo"]["1"], arms["expression_anchor_sft"]["1"]]
    positive_control = (
        truthful["accuracy"] >= .80
        and truthful["mean_probability"] - baseline["1"]["mean_probability"] >= .10
    )
    gates = {
        "teacher_interface_qualified": all(qgates.values()),
        "truthful_sdpo_positive_control": positive_control,
        "raw_selects_transition_action": raw_zero["accuracy"] >= .80 and raw_zero["mean_probability"] >= .65,
        "expression_augmented_selects_expression_action": expression_augmented["accuracy"] >= .80,
        "expression_augmented_beats_equal_anchor_baselines": (
            expression_augmented["mean_probability"] >= max(item["mean_probability"] for item in expression_baselines) + .05
            and expression_augmented["accuracy"] >= max(item["accuracy"] for item in expression_baselines)
        ),
        "transition_augmented_raw_identity": result["max_transition_augmented_raw_identity_error"] <= 1e-6,
    }
    expected_decision = "NEURAL_ANCHOR_G0_QUALIFIED" if all(gates.values()) else "NEURAL_ANCHOR_G0_NOT_QUALIFIED"
    if result["gates"] != gates or result["decision"] != expected_decision:
        raise SystemExit("final gate arithmetic mismatch")
    print(json.dumps({"verified": True, "decision": result["decision"], "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
