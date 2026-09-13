#!/usr/bin/env python3
"""Aggregate frozen G7 source results without post-hoc selection."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


T_CRITICAL_N3 = 4.302652729696142


def interval(values: Sequence[float]) -> dict[str, Any]:
    x = np.asarray(values, dtype=float)
    mean = float(x.mean()); se = float(x.std(ddof=1) / math.sqrt(len(x)))
    return {"values": x.tolist(), "mean": mean, "standard_error": se,
            "lower": mean - T_CRITICAL_N3 * se, "upper": mean + T_CRITICAL_N3 * se}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    cfg = load(args.config); seeds = [int(x) for x in cfg["seeds"]]
    pretrain, cells = {}, {}
    for arm in cfg["arms"]:
        pretrain[arm] = [load(args.root / "pretrain" / arm / f"seed_{seed}" / "REPORT.json")
                         for seed in seeds]
        cells[arm] = {}
        for fine in cfg["posttraining"]["fine_tuning_arms"]:
            cells[arm][fine] = [
                next(cell for cell in load(args.root / "posttrain" / arm / f"seed_{seed}" / "REPORT.json")["cells"]
                     if cell["fine_tuning_arm"] == fine) for seed in seeds
            ]
    pretraining = {}
    for arm, reports in pretrain.items():
        pretraining[arm] = {
            "intact_nll": interval([r["evaluation"]["intact"]["nll"] for r in reports]),
            "bypass_minus_intact_nll": interval([r["evaluation"]["bypass_minus_intact_nll"] for r in reports]),
            "training_tokens_per_second": interval([r["training_tokens_per_second"] for r in reports]),
        }
    paired_nll = [
        pretrain["conditional_memory"][i]["evaluation"]["intact"]["nll"] -
        pretrain["dense_control"][i]["evaluation"]["intact"]["nll"] for i in range(len(seeds))
    ]
    pretraining["conditional_minus_dense_nll"] = interval(paired_nll)
    summaries = {}
    for arm in cfg["arms"]:
        summaries[arm] = {}
        for fine, rows in cells[arm].items():
            keys = ("installed_attack_excess", "outside_component_sufficiency",
                    "component_sufficiency", "outside_minus_component_sufficiency")
            summaries[arm][fine] = {key: interval([r["measures"][key] for r in rows]) for key in keys}
            summaries[arm][fine]["installation_seed_passes"] = [
                r["measures"]["installed_attack_excess"] >= cfg["thresholds"]["installation_eligibility"]
                for r in rows
            ]
            if arm == "conditional_memory":
                for key in ("whole_table_necessity", "whole_table_sufficiency", "target_row_necessity",
                            "target_row_sufficiency", "target_row_zero_drop", "target_row_zero_specificity",
                            "component_necessity", "outside_component_necessity"):
                    summaries[arm][fine][key] = interval([r["measures"][key] for r in rows])
    threshold = float(cfg["thresholds"]["minimum_meaningful_effect"])
    apparatus = all(summaries[arm]["ordinary"]["installation_seed_passes"]
                    for arm in cfg["arms"])
    bias = summaries["conditional_memory"]["ordinary"]["outside_minus_component_sufficiency"]
    if not apparatus:
        routing = "INVALID_FOR_CROSS_ARM_ROUTING_INTERPRETATION_APPARATUS_FAILURE"
    elif bias["lower"] > threshold:
        routing = "BACKBONE_ROUTING_SURVIVES_JOINT_PRETRAINING"
    elif bias["upper"] < -threshold:
        routing = "MEMORY_ROUTING_BOUNDARY_CONDITION"
    else:
        routing = "INDETERMINATE_OR_MIXED_ROUTING"
    conditional = summaries["conditional_memory"]["ordinary"]
    locality = "PER_ITEM_BOUNDARY_SUPPORTED" if all(
        conditional[key]["lower"] > threshold for key in
        ("target_row_necessity", "target_row_sufficiency", "target_row_zero_specificity")
    ) else "PER_ITEM_BOUNDARY_NOT_SUPPORTED"
    result = {"status": "COMPLETE", "seeds": seeds, "pretraining": pretraining,
              "posttraining": summaries, "decision": {"apparatus_valid": apparatus,
              "routing": routing, "nominal_row_locality": locality},
              "interpretation_scope": cfg["scope"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
