"""Read-only verifier for the shared-LoRA Hindsight bifurcation follow-up."""
import argparse
import hashlib
import json
from pathlib import Path

from interaction_sprint.hindsight_bifurcation_lora import ARMS, STEPS, prepare_contexts, schedule


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
        "contexts.json", "fixed_marginals.json", "initial_adapter.pt", "offsets.json",
        "RESULT.json", "runner_source.py", "runtime.json", "schedule.json", "spec.json",
    }
    for arm in ARMS:
        expected.update({f"{arm}_adapter.pt", f"{arm}_checkpoints.json", f"{arm}_optimizer.pt", f"{arm}_steps.json"})
    if set(manifest) != expected:
        raise ValueError("Unexpected manifest members")
    for name, digest in manifest.items():
        if hashlib.sha256((args.root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Hash mismatch: {name}")
    source = Path("src/interaction_sprint/hindsight_bifurcation_lora.py")
    if source.read_bytes() != (args.root / "runner_source.py").read_bytes():
        raise ValueError("Runner source mismatch")

    contexts = load(args.root / "contexts.json")
    if contexts != prepare_contexts(args.evidence):
        raise ValueError("Prerequisite context reconstruction mismatch")
    development_ids = [item["base_id"] for item in contexts if item["split"] == "development"]
    if load(args.root / "schedule.json") != schedule(development_ids):
        raise ValueError("Schedule mismatch")
    spec = load(args.root / "spec.json")
    prerequisite_hash = hashlib.sha256((args.evidence / "MANIFEST.json").read_bytes()).hexdigest()
    if spec["prerequisite_manifest_sha256"] != prerequisite_hash:
        raise ValueError("Prerequisite binding mismatch")

    checkpoints = {}
    total_steps = 0
    for arm in ARMS:
        steps = load(args.root / f"{arm}_steps.json")
        if len(steps) != STEPS or [row["step"] for row in steps] != list(range(1, STEPS + 1)):
            raise ValueError(f"Step coverage mismatch: {arm}")
        total_steps += len(steps)
        values = load(args.root / f"{arm}_checkpoints.json")
        if set(values) != {"0", "32", "64"} or any(len(values[key]) != 32 for key in values):
            raise ValueError(f"Checkpoint coverage mismatch: {arm}")
        checkpoints[arm] = values

    for minus, plus in (("dynamic_minus", "fixed_minus"), ("dynamic_plus", "fixed_plus")):
        a = {row["base_id"]: row for row in checkpoints[minus]["0"]}
        b = {row["base_id"]: row for row in checkpoints[plus]["0"]}
        if a != b:
            raise ValueError("Matched initial policies differ")

    final = {arm: {row["base_id"]: row for row in checkpoints[arm][str(STEPS)]} for arm in ARMS}
    ids = [item["base_id"] for item in contexts if item["split"] == "confirmation" and item["bistable"]]
    dynamic = [final["dynamic_plus"][key]["probability_one"] - final["dynamic_minus"][key]["probability_one"] for key in ids]
    fixed = [final["fixed_plus"][key]["probability_one"] - final["fixed_minus"][key]["probability_one"] for key in ids]
    metrics = dict(
        confirmation_bistable=len(ids),
        dynamic_median_separation=sorted(dynamic)[len(dynamic) // 2],
        fixed_median_separation=sorted(fixed)[len(fixed) // 2],
        fixed_maximum_absolute_separation=max(abs(value) for value in fixed),
        dynamic_positive_fraction=sum(value > 0 for value in dynamic) / len(dynamic),
        dynamic_differences=dynamic,
        fixed_differences=fixed,
    )
    result = load(args.root / "RESULT.json")
    if metrics != result["metrics"]:
        raise ValueError("Metric recomputation mismatch")
    passed = metrics["dynamic_median_separation"] >= 0.25 and metrics["fixed_maximum_absolute_separation"] <= 0.10 and metrics["dynamic_positive_fraction"] >= 0.75
    expected_status = "LORA_BIFURCATION_SCREEN_POSITIVE" if passed else "LORA_BIFURCATION_SCREEN_NEGATIVE"
    if result["status"] != expected_status or result["counts"] != {"forward_batches": 257, "backwards": 256, "updates": 256}:
        raise ValueError("Decision or call count mismatch")

    receipt = dict(
        verified=True,
        manifest_sha256=hashlib.sha256((args.root / "MANIFEST.json").read_bytes()).hexdigest(),
        files_verified=len(manifest),
        step_records_verified=total_steps,
        status=result["status"],
        metrics=metrics,
        adapter_deserialization_checked=False,
        neural_forwards_replayed=False,
        verifier_scope="Hashes, source/prerequisite/schedule identity, checkpoint coverage, matched initialization, metric and decision recomputation",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in receipt.items() if key != "metrics"}))


if __name__ == "__main__":
    main()
