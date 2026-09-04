"""Read-only verifier for the classical PUPPET human-feedback DEV evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from interaction_sprint.hindsight_human_feedback import (
    ARMS,
    cluster_bootstrap_gain,
    metrics,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.root / "MANIFEST.json").read_text(encoding="utf-8"))
    if set(manifest) != {"report.json", "predictions.jsonl"}:
        raise ValueError("unexpected manifest members")
    for name, expected in manifest.items():
        if sha256(args.root / name) != expected:
            raise ValueError(f"checksum mismatch: {name}")
    report = json.loads((args.root / "report.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (args.root / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 72 or len({row["record_sha256"] for row in rows}) != 72:
        raise ValueError("expected 72 unique DEV records")
    groups = [row["query_sha256"] for row in rows]
    if len(set(groups)) != 7:
        raise ValueError("expected seven DEV query groups")
    if set(row["condition"] for row in rows) - {"C3", "C4", "C6"}:
        raise ValueError("unexpected condition")
    target = np.array([row["belief_delta"] for row in rows], dtype=float)
    predictions = {
        depth: {arm: np.array([row["predictions"][str(depth)][arm] for row in rows], dtype=float)
                for arm in ARMS}
        for depth in (3, 6)
    }
    reproduced_metrics = {
        str(depth): {arm: metrics(target, predictions[depth][arm]) for arm in ARMS}
        for depth in (3, 6)
    }
    reproduced_comparisons = {
        "early_user_over_query": cluster_bootstrap_gain(
            target, predictions[3]["query_only"], predictions[3]["user_only"], groups),
        "late_user_over_query": cluster_bootstrap_gain(
            target, predictions[6]["query_only"], predictions[6]["user_only"], groups),
        "early_full_over_assistant": cluster_bootstrap_gain(
            target, predictions[3]["assistant_only"], predictions[3]["full"], groups),
        "late_full_over_assistant": cluster_bootstrap_gain(
            target, predictions[6]["assistant_only"], predictions[6]["full"], groups),
    }
    if reproduced_metrics != report["metrics"] or reproduced_comparisons != report["comparisons"]:
        raise ValueError("reported statistics do not reproduce")
    gates = {
        "late_full_spearman_at_least_point_30": reproduced_metrics["6"]["full"]["spearman"] >= 0.30,
        "late_user_adds_query_cluster_robust_signal":
            reproduced_comparisons["late_user_over_query"]["cluster_bootstrap_ci95_low"] > 0,
        "late_user_relative_mse_gain_at_least_5pct":
            reproduced_comparisons["late_user_over_query"]["relative_mse_gain"] >= 0.05,
        "late_full_adds_beyond_assistant_cluster_robust_signal":
            reproduced_comparisons["late_full_over_assistant"]["cluster_bootstrap_ci95_low"] > 0,
        "late_full_relative_mse_gain_at_least_5pct":
            reproduced_comparisons["late_full_over_assistant"]["relative_mse_gain"] >= 0.05,
    }
    decision = "DEV_SIGNAL_QUALIFIED_CONFIRMATION_STILL_LOCKED" if all(gates.values()) else "DEV_SIGNAL_NOT_QUALIFIED"
    if gates != report["gates"] or decision != report["decision"]:
        raise ValueError("gate decision does not reproduce")
    print(json.dumps({"verified": True, "decision": decision,
                      "manifest_sha256": sha256(args.root / "MANIFEST.json")}, indent=2))


if __name__ == "__main__":
    main()

