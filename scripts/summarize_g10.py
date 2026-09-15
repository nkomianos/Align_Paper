#!/usr/bin/env python3
"""Apply frozen G10 semantic calibration and confirmatory decisions."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def load(path: Path) -> Any: return json.loads(path.read_text(encoding="utf-8"))


def interval(values: Sequence[float]) -> dict[str, Any]:
    x=np.asarray(values,dtype=float);mean=float(x.mean());se=float(x.std(ddof=1)/math.sqrt(len(x)))
    half=2.7764451051977987*se
    return {"values":x.tolist(),"mean":mean,"standard_error":se,"lower":mean-half,"upper":mean+half}


def calibration(cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    source=load(root/"source"/"calibration"/"REPORT.json");replay=load(root/"replay"/"calibration"/"REPORT.json")
    exact=all(source[key]==replay[key] for key in ("before","after","sequence_exact_gain","token_mrr_gain",
                                                    "mean_token_log_probability_gain","placement_rows","final_state_sha256"))
    passed=source["sequence_exact_gain"]>=float(cfg["thresholds"]["calibration_sequence_exact_gain"])
    return {"status":"COMPLETE","decision":"ADVANCE" if passed and exact else "CLOSE_CALIBRATION_FAILED",
            "sequence_exact_pass":passed,"short_run_bitwise_replay":exact,
            "source_sequence_exact_gain":source["sequence_exact_gain"],"source_token_mrr_gain":source["token_mrr_gain"],
            "source_log_probability_gain":source["mean_token_log_probability_gain"]}


def confirmatory(cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    reports=[load(root/"source"/"posttraining"/f"seed_{seed}"/"REPORT.json") for seed in cfg["posttraining"]["seeds"]]
    ordinary=[next(x for x in row["cells"] if x["arm"]=="ordinary")["measures"] for row in reports]
    frozen=[next(x for x in row["cells"] if x["arm"]=="frozen_graft")["measures"] for row in reports]
    keys=("installed_exact_excess","outside_graft_sufficiency","graft_sufficiency",
          "outside_minus_graft_sufficiency","whole_table_necessity","whole_table_sufficiency",
          "target_union_necessity","target_union_sufficiency","target_union_zero_drop",
          "mean_per_user_deletion_selectivity","mean_cross_user_drop")
    estimates={key:interval([row[key] for row in ordinary]) for key in keys}
    estimates["frozen_graft_installed_exact_excess"]=interval([row["installed_exact_excess"] for row in frozen])
    threshold=float(cfg["thresholds"]["minimum_meaningful_effect"])
    installation=estimates["installed_exact_excess"]["lower"]>threshold
    route=estimates["outside_minus_graft_sufficiency"]
    if not installation: routing="INVALID_INSTALLATION_FAILURE"
    elif route["lower"]>threshold: routing="BACKBONE_ROUTING"
    elif route["upper"] < -threshold: routing="GRAFT_ROUTING"
    else: routing="MIXED_OR_INDETERMINATE"
    deletion="SUPPORTED" if installation and estimates["mean_per_user_deletion_selectivity"]["lower"]>threshold else "NOT_SUPPORTED"
    return {"status":"COMPLETE","estimates":estimates,
            "decision":{"installation_valid":installation,"routing":routing,"per_user_deletion_boundary":deletion},
            "unit":"five independent clean-adaptation/training seeds; users and prompts are within-seed measurements"}


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,required=True);p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--mode",choices=("calibration","confirmatory"),required=True)
    a=p.parse_args();cfg=load(a.config);result=calibration(cfg,a.root) if a.mode=="calibration" else confirmatory(cfg,a.root)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":main()
