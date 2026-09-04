"""Read-only verifier for the natural-initialization Hindsight diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path

from interaction_sprint.hindsight_bifurcation_lora import STEPS, prepare_contexts, schedule
from interaction_sprint.hindsight_natural_feedback_lora import ARMS, natural_metrics


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.resolve().is_relative_to(args.root.resolve()):
        raise ValueError("Receipt must be fresh and outside the evidence root")

    manifest = load(args.root / "MANIFEST.json")
    expected = {
        "contexts.json", "fixed_marginals.json", "initial_adapter.pt", "initial_probabilities.json",
        "RESULT.json", "runner_source.py", "runtime.json", "schedule.json", "spec.json",
    }
    for arm in ARMS:
        expected.update({f"{arm}_adapter.pt", f"{arm}_checkpoints.json", f"{arm}_optimizer.pt", f"{arm}_steps.json"})
    if set(manifest) != expected:
        raise ValueError("Unexpected manifest members")
    for name, digest in manifest.items():
        if hashlib.sha256((args.root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Hash mismatch: {name}")
    source = Path("src/interaction_sprint/hindsight_natural_feedback_lora.py")
    if source.read_bytes() != (args.root / "runner_source.py").read_bytes():
        raise ValueError("Runner source mismatch")

    contexts = load(args.root / "contexts.json")
    if contexts != prepare_contexts(args.evidence):
        raise ValueError("Prerequisite context mismatch")
    development_ids = [item["base_id"] for item in contexts if item["split"] == "development"]
    if load(args.root / "schedule.json") != schedule(development_ids):
        raise ValueError("Schedule mismatch")
    spec = load(args.root / "spec.json")
    if spec["prerequisite_manifest_sha256"] != hashlib.sha256((args.evidence / "MANIFEST.json").read_bytes()).hexdigest():
        raise ValueError("Prerequisite binding mismatch")

    initial = load(args.root / "initial_probabilities.json")
    checkpoints = {}
    step_count = 0
    for arm in ARMS:
        steps = load(args.root / f"{arm}_steps.json")
        if len(steps) != STEPS or [row["step"] for row in steps] != list(range(1, STEPS + 1)):
            raise ValueError(f"Step coverage mismatch: {arm}")
        step_count += len(steps)
        values = load(args.root / f"{arm}_checkpoints.json")
        if set(values) != {"0", "32", "64"} or any(len(values[key]) != 32 for key in values):
            raise ValueError(f"Checkpoint coverage mismatch: {arm}")
        if values["0"] != initial:
            raise ValueError(f"Initial policy mismatch: {arm}")
        checkpoints[arm] = values

    metrics = natural_metrics(contexts, initial, checkpoints["dynamic_natural"][str(STEPS)], checkpoints["fixed_natural"][str(STEPS)])
    result = load(args.root / "RESULT.json")
    if metrics != result["metrics"]:
        raise ValueError("Metric recomputation mismatch")
    passed = metrics["upper_median_signed_dynamic_advantage"] >= 0.25 and metrics["positive_advantage_fraction"] >= 0.75 and metrics["dynamic_basin_consistency"] >= 0.75
    expected_status = "NATURAL_LORA_FEEDBACK_SCREEN_POSITIVE" if passed else "NATURAL_LORA_FEEDBACK_SCREEN_NEGATIVE"
    if result["status"] != expected_status or result["counts"] != {"forward_batches": 129, "backwards": 128, "updates": 128}:
        raise ValueError("Decision or count mismatch")

    receipt = dict(
        verified=True,
        manifest_sha256=hashlib.sha256((args.root / "MANIFEST.json").read_bytes()).hexdigest(),
        files_verified=len(manifest),
        step_records_verified=step_count,
        status=result["status"],
        metrics={key: value for key, value in metrics.items() if key != "records"},
        neural_forwards_replayed=False,
        state_deserialization_checked=False,
        verifier_scope="Hashes, source/prerequisite/schedule identity, exact initial matching, checkpoint/step coverage, metric and decision recomputation",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
