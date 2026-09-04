"""Read-only verifier for the Hindsight bifurcation screen."""
import argparse
import hashlib
import json
from pathlib import Path

from interaction_sprint.hindsight_bifurcation_g0 import analyze, dataset, distribution


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.resolve().is_relative_to(args.root.resolve()):
        raise ValueError("Receipt must be a fresh path outside the evidence root")

    manifest = load(args.root / "MANIFEST.json")
    expected_files = {"cases.json", "RESULT.json", "rows.json", "runner_source.py", "runtime.json", "spec.json"}
    if set(manifest) != expected_files:
        raise ValueError("Unexpected manifest members")
    for name, expected in manifest.items():
        actual = hashlib.sha256((args.root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Hash mismatch: {name}")

    source = Path("src/interaction_sprint/hindsight_bifurcation_g0.py")
    if (args.root / "runner_source.py").read_bytes() != source.read_bytes():
        raise ValueError("Runner source differs from committed working source")
    cases = load(args.root / "cases.json")
    if cases != dataset():
        raise ValueError("Case set differs from prospective source")

    rows = load(args.root / "rows.json")
    if len(rows) != 96 or len([row for row in rows if row["kind"] == "hindsight"]) != 64:
        raise ValueError("Incomplete row coverage")
    for row in rows:
        scores = []
        for token_values in row["token_logprobabilities"]:
            scores.append(sum(token_values))
        if any(abs(a - b) > 1e-12 for a, b in zip(scores, row["logprobabilities"])):
            raise ValueError("Token/sequence likelihood mismatch")
        expected = distribution(scores)
        if any(abs(a - b) > 1e-12 for a, b in zip(expected, row["probabilities"])):
            raise ValueError("Probability normalization mismatch")
        predicted = 0 if expected[0] >= expected[1] else 1
        if predicted != row["prediction"]:
            raise ValueError("Prediction mismatch")

    recorded = load(args.root / "RESULT.json")
    recomputed = analyze(rows)
    for key in ("status", "teacher", "splits", "contexts", "criteria", "parameter_updates", "human_preference_transition_tested", "paper_green_light"):
        if recorded[key] != recomputed[key]:
            raise ValueError(f"Recomputed result mismatch: {key}")
    if recorded["scored_prompts"] != 96 or recorded["sequence_forwards"] != 192:
        raise ValueError("Recorded call counts mismatch")

    receipt = dict(
        verified=True,
        manifest_sha256=hashlib.sha256((args.root / "MANIFEST.json").read_bytes()).hexdigest(),
        files_verified=len(manifest),
        rows_verified=len(rows),
        sequence_forwards_recorded=recorded["sequence_forwards"],
        status=recorded["status"],
        confirmation_bistable=recorded["splits"]["confirmation"]["bistable"],
        neural_forwards_replayed=False,
        verifier_scope="Hashes, source/case identity, score arithmetic, coverage, fixed-point and decision recomputation",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
