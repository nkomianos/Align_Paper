"""Run the frozen DEV-only PUPPET text-state audit without printing raw text."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from interaction_sprint.hindsight_human_feedback import (
    ARMS,
    DATA_SHA256,
    RIDGE_ALPHA,
    cluster_bootstrap_gain,
    leave_one_query_out,
    metrics,
    records_from_rows,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    raw = args.data.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        raise ValueError("dataset checksum mismatch")
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    records, splits = records_from_rows(rows)
    dev = [record for record in records if record.query_sha256 in splits["dev"]]
    if len(dev) != 72 or len({record.query_sha256 for record in dev}) != 7:
        raise ValueError("frozen DEV cohort changed")
    target = np.array([record.belief_delta for record in dev], dtype=float)
    groups = [record.query_sha256 for record in dev]
    predictions = {}
    result_metrics = {}
    for depth in (3, 6):
        predictions[depth] = {}
        result_metrics[depth] = {}
        for arm in ARMS:
            prediction = leave_one_query_out(dev, arm, depth)
            predictions[depth][arm] = prediction
            result_metrics[depth][arm] = metrics(target, prediction)
    comparisons = {
        "early_user_over_query": cluster_bootstrap_gain(
            target, predictions[3]["query_only"], predictions[3]["user_only"], groups),
        "late_user_over_query": cluster_bootstrap_gain(
            target, predictions[6]["query_only"], predictions[6]["user_only"], groups),
        "early_full_over_assistant": cluster_bootstrap_gain(
            target, predictions[3]["assistant_only"], predictions[3]["full"], groups),
        "late_full_over_assistant": cluster_bootstrap_gain(
            target, predictions[6]["assistant_only"], predictions[6]["full"], groups),
    }
    late_full = result_metrics[6]["full"]
    gates = {
        "late_full_spearman_at_least_point_30": late_full["spearman"] >= 0.30,
        "late_user_adds_query_cluster_robust_signal":
            comparisons["late_user_over_query"]["cluster_bootstrap_ci95_low"] > 0,
        "late_user_relative_mse_gain_at_least_5pct":
            comparisons["late_user_over_query"]["relative_mse_gain"] >= 0.05,
        "late_full_adds_beyond_assistant_cluster_robust_signal":
            comparisons["late_full_over_assistant"]["cluster_bootstrap_ci95_low"] > 0,
        "late_full_relative_mse_gain_at_least_5pct":
            comparisons["late_full_over_assistant"]["relative_mse_gain"] >= 0.05,
    }
    decision = "DEV_SIGNAL_QUALIFIED_CONFIRMATION_STILL_LOCKED" if all(gates.values()) else "DEV_SIGNAL_NOT_QUALIFIED"
    report = {
        "decision": decision,
        "scope": "Developmental predictive audit only; not causal mediation, welfare, preference learning, or a paper gate.",
        "dataset_sha256": DATA_SHA256,
        "cohort": {"rows": len(dev), "query_groups": len(set(groups)), "conditions": sorted(set(r.condition for r in dev))},
        "model": {"kind": "fixed TF-IDF word+character Ridge", "alpha": RIDGE_ALPHA,
                  "validation": "leave-one-query-out on the frozen seven-query DEV subset"},
        "metrics": result_metrics,
        "comparisons": comparisons,
        "gates": gates,
        "interpretation_limits": [
            "Text prediction of a later survey value does not identify a message-level causal effect.",
            "The six-user-turn cohort conditions on conversation length.",
            "A source-model style or query artifact may be predictive without representing belief.",
            "Confirmation query outcomes remain unopened by this script.",
        ],
    }
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    prediction_path = args.output / "predictions.jsonl"
    with prediction_path.open("x", encoding="utf-8") as handle:
        for index, record in enumerate(dev):
            item = {
                "record_sha256": record.record_sha256,
                "query_sha256": record.query_sha256,
                "condition": record.condition,
                "pre_belief": record.pre_belief,
                "belief_delta": record.belief_delta,
                "predictions": {str(depth): {arm: float(predictions[depth][arm][index]) for arm in ARMS}
                                for depth in (3, 6)},
            }
            handle.write(json.dumps(item, sort_keys=True) + "\n")
    manifest = {path.name: sha256(path) for path in (report_path, prediction_path)}
    (args.output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"decision": decision, "gates": gates, "metrics": result_metrics,
                      "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()

