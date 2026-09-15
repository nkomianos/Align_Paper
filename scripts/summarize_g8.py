#!/usr/bin/env python3
"""Apply the frozen G8 calibration-selection and routing decision rules."""
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
    threshold = float(cfg["thresholds"]["minimum_meaningful_effect"])
    summaries = []
    selected = None
    for cell in cfg["calibration"]["ladder"]:
        rows = []
        complete = True
        for execution in ("source", "replay"):
            for arm in cfg["arms"]:
                for seed in cfg["seeds"]:
                    path = root / execution / "calibration" / cell["id"] / arm / f"seed_{seed}" / "REPORT.json"
                    if not path.exists():
                        complete = False
                        continue
                    rows.append(load(path))
        if not complete:
            continue
        exact = True
        for arm in cfg["arms"]:
            for seed in cfg["seeds"]:
                source = load(root / "source" / "calibration" / cell["id"] / arm / f"seed_{seed}" / "REPORT.json")
                replay = load(root / "replay" / "calibration" / cell["id"] / arm / f"seed_{seed}" / "REPORT.json")
                for key in ("before_accuracy", "after_accuracy", "unconditional_accuracy_excess",
                            "before_predictions", "after_predictions", "final_state_sha256", "placement_rows"):
                    exact = exact and source[key] == replay[key]
        source_rows = [load(root / "source" / "calibration" / cell["id"] / arm / f"seed_{seed}" / "REPORT.json")
                       for arm in cfg["arms"] for seed in cfg["seeds"]]
        passes = [row["unconditional_accuracy_excess"] >= threshold for row in source_rows]
        current = {"cell": cell, "source_excess": [row["unconditional_accuracy_excess"] for row in source_rows],
                   "all_six_pass": all(passes), "short_run_bitwise_replay": exact}
        summaries.append(current)
        if selected is None and current["all_six_pass"] and exact:
            selected = cell
    return {"status": "COMPLETE", "decision": "ADVANCE" if selected else "CLOSE_CALIBRATION_FAILED",
            "selected_cell": selected, "cells": summaries,
            "selection_rule": "first registered cell where all six inherited checkpoints reach >=0.15 unconditional excess and source/replay are bitwise exact"}


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
    args = parser.parse_args()
    cfg = load(args.config)
    result = calibration(cfg, args.root) if args.mode == "calibration" else routing(cfg, args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
