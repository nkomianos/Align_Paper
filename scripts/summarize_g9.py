#!/usr/bin/env python3
"""Apply the frozen G9 calibration and routing decisions."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

T_CRITICAL_N3 = 4.302652729696142


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def interval(values: Sequence[float]) -> dict[str, Any]:
    x = np.asarray(values, dtype=float)
    se = float(x.std(ddof=1) / math.sqrt(len(x)))
    mean = float(x.mean())
    return {"values": x.tolist(), "mean": mean, "standard_error": se,
            "lower": mean - T_CRITICAL_N3 * se, "upper": mean + T_CRITICAL_N3 * se}


def calibration(cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    exact_gate = float(cfg["thresholds"]["calibration_exact_gain"])
    mrr_gate = float(cfg["thresholds"]["secondary_mrr_gain"])
    summaries = []
    selected = None
    for rate in cfg["calibration"]["learning_rates"]:
        exact_replay = True
        source_rows = []
        for arm in cfg["arms"]:
            for seed in cfg["seeds"]:
                source = load(root / "source" / "calibration" / rate["id"] / arm / f"seed_{seed}" / "REPORT.json")
                replay = load(root / "replay" / "calibration" / rate["id"] / arm / f"seed_{seed}" / "REPORT.json")
                source_rows.append(source)
                for key in ("before", "after", "exact_match_gain", "mrr_gain",
                            "mean_log_probability_gain", "final_state_sha256", "placement_rows"):
                    exact_replay = exact_replay and source[key] == replay[key]
        exact_pass = all(row["exact_match_gain"] >= exact_gate for row in source_rows)
        mrr_pass = all(row["mrr_gain"] >= mrr_gate for row in source_rows)
        current = {"rate": rate,
                   "source_exact_match_gains": [row["exact_match_gain"] for row in source_rows],
                   "source_mrr_gains": [row["mrr_gain"] for row in source_rows],
                   "source_log_probability_gains": [row["mean_log_probability_gain"] for row in source_rows],
                   "all_six_exact_pass": exact_pass, "all_six_secondary_mrr_pass": mrr_pass,
                   "short_run_bitwise_replay": exact_replay}
        summaries.append(current)
        if selected is None and exact_pass and exact_replay:
            selected = rate
    if selected is not None:
        decision, failure_type = "ADVANCE", None
    elif any(row["all_six_secondary_mrr_pass"] for row in summaries):
        decision, failure_type = "CLOSE_CALIBRATION_FAILED", "METRIC_SENSITIVITY_RANK_MOVED_ARGMAX_DID_NOT"
    else:
        decision, failure_type = "CLOSE_CALIBRATION_FAILED", "CAPABILITY_PAYLOAD_RANK_DID_NOT_MOVE_ENOUGH"
    return {"status": "COMPLETE", "decision": decision, "failure_classification": failure_type,
            "selected_rate": selected, "rates": summaries,
            "selection_rule": "lowest registered rate where all six checkpoints gain >=0.15 exact match and the short run replays bitwise",
            "secondary_rule": "if exact calibration fails, all-six mean-reciprocal-rank gain >=0.15 classifies metric sensitivity; otherwise capability failure"}


def routing(cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for arm in cfg["arms"]:
        summaries[arm] = {}
        reports = [load(root / "source" / "posttraining" / arm / f"seed_{seed}" / "REPORT.json")
                   for seed in cfg["seeds"]]
        for fine in cfg["posttraining"]["fine_tuning_arms"]:
            rows = [next(cell for cell in report["cells"] if cell["fine_tuning_arm"] == fine)
                    for report in reports]
            keys = ("installed_attack_excess", "outside_component_sufficiency",
                    "component_sufficiency", "outside_minus_component_sufficiency")
            summaries[arm][fine] = {key: interval([row["measures"][key] for row in rows]) for key in keys}
            summaries[arm][fine]["continuous_intact_mrr"] = interval(
                [row["continuous_measures"]["intact"]["mean_reciprocal_rank"] for row in rows])
            if arm == "conditional_memory":
                for key in ("whole_table_necessity", "whole_table_sufficiency", "target_row_necessity",
                            "target_row_sufficiency", "target_row_zero_specificity"):
                    summaries[arm][fine][key] = interval([row["measures"][key] for row in rows])
    threshold = float(cfg["thresholds"]["minimum_meaningful_effect"])
    installation = all(summaries[arm]["ordinary"]["installed_attack_excess"]["lower"] > threshold
                       for arm in cfg["arms"])
    effect = summaries["conditional_memory"]["ordinary"]["outside_minus_component_sufficiency"]
    if not installation:
        decision = "INVALID_POSTTRAINING_ROUTING_ASSAY_INSTALLATION_FAILURE"
    elif effect["lower"] > threshold:
        decision = "BACKBONE_ROUTING_SURVIVES_JOINT_PRETRAINING"
    elif effect["upper"] < -threshold:
        decision = "MEMORY_ROUTING_BOUNDARY_CONDITION"
    else:
        decision = "MIXED_OR_INDETERMINATE_ROUTING"
    locality = summaries["conditional_memory"]["ordinary"]
    locality_pass = all(locality[key]["lower"] > threshold for key in
                        ("target_row_necessity", "target_row_sufficiency", "target_row_zero_specificity"))
    return {"status": "COMPLETE", "posttraining": summaries,
            "decision": {"installation_valid": installation, "routing": decision,
                         "nominal_row_boundary": "SUPPORTED" if locality_pass else "NOT_SUPPORTED"},
            "inherited_pretraining": "G7 source checkpoints; non-bitwise pretraining provenance declared"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("calibration", "routing"), required=True)
    args = parser.parse_args(); cfg = load(args.config)
    result = calibration(cfg, args.root) if args.mode == "calibration" else routing(cfg, args.root)
    write = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(write, encoding="utf-8")
    print(write)


if __name__ == "__main__":
    main()
