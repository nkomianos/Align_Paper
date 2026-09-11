#!/usr/bin/env python3
"""Independent no-GPU audit of S1 selection, replication inventory, and decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


T_CRITICAL_DF4 = 2.7764451051977987


def interval(values: list[float]) -> dict[str, float]:
    if len(values) != 5:
        raise AssertionError(f"expected five seed values, found {len(values)}")
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / math.sqrt(5))
    return {"mean": mean, "standard_error": se,
            "lower": mean - T_CRITICAL_DF4 * se, "upper": mean + T_CRITICAL_DF4 * se}


def assert_close(left: Any, right: Any, path: str) -> None:
    if isinstance(right, dict):
        if set(left) != set(right):
            raise AssertionError(f"keys differ at {path}")
        for key in right:
            assert_close(left[key], right[key], f"{path}.{key}")
    elif isinstance(right, float):
        if not math.isclose(float(left), right, rel_tol=1e-10, abs_tol=1e-12):
            raise AssertionError(f"value differs at {path}: {left} != {right}")
    elif left != right:
        raise AssertionError(f"value differs at {path}: {left} != {right}")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    development_path = args.root / "DEVELOPMENT_SUMMARY.json"
    decisive_path = args.root / "DECISIVE_SUMMARY.json"
    decision_path = args.root / "DECISION.json"
    development = json.loads(development_path.read_text())
    decisive = json.loads(decisive_path.read_text())
    decision = json.loads(decision_path.read_text())
    threshold = float(config["threshold_derivation"]["minimum_intact_asr_excess_for_ablation_eligibility"])
    minimum = float(config["estimand"]["minimum_meaningful_localization_specificity"])
    expected_dev_seed = int(config["staging"]["development_seed"])
    expected_replication = sorted(int(seed) for seed in config["staging"]["replication_seeds"])
    expected_counts = sorted(int(count) for count in config["poison_training"]["poison_counts"])
    expected_arms = sorted(config["poison_training"]["table_arms"])

    selections: dict[str, int | None] = {}
    per_model: dict[str, Any] = {}
    for model in (spec["alias"] for spec in config["models"]):
        dev = [cell for cell in development if cell["model"] == model]
        observed_dev = sorted((int(cell["seed"]), int(cell["poison_count"]), cell["table_arm"]) for cell in dev)
        expected_dev = sorted((expected_dev_seed, count, arm) for count in expected_counts for arm in expected_arms)
        if observed_dev != expected_dev:
            raise AssertionError(f"development inventory differs for {model}")
        eligible = sorted(
            int(cell["poison_count"]) for cell in dev
            if cell["table_arm"] == "trainable" and float(cell["installed_attack_excess"]) >= threshold
        )
        selected = eligible[0] if eligible else None
        selections[model] = selected
        cells = [cell for cell in decisive if cell["model"] == model]
        if selected is None:
            if cells:
                raise AssertionError(f"unexpected decisive cells for excluded model {model}")
            per_model[model] = {"selected_poison_count": None, "status": "NO_ELIGIBLE_POISON_COUNT"}
            continue
        observed = sorted((int(cell["seed"]), int(cell["poison_count"]), cell["table_arm"]) for cell in cells)
        expected = sorted((seed, selected, arm) for seed in expected_replication for arm in expected_arms)
        if observed != expected:
            raise AssertionError(f"decisive inventory differs for {model}")
        by_key = {(int(cell["seed"]), cell["table_arm"]): cell for cell in cells}
        localization = [
            float(by_key[(seed, "trainable")]["evaluation"]["localization_specificity"])
            for seed in expected_replication
        ]
        preference = [
            float(by_key[(seed, "trainable")]["evaluation"]["localization_specificity"])
            - float(by_key[(seed, "frozen")]["evaluation"]["localization_specificity"])
            for seed in expected_replication
        ]
        localization_interval = interval(localization)
        preference_interval = interval(preference)
        passed = localization_interval["lower"] > minimum and preference_interval["lower"] > 0
        per_model[model] = {
            "selected_poison_count": selected,
            "localization_specificity": localization_interval,
            "table_preference": preference_interval,
            "passed": passed,
            "status": "PASS" if passed else "FAIL",
        }
    passes = sum(bool(result.get("passed")) for result in per_model.values())
    if passes == len(config["models"]):
        status = "CROSS_SCALE_POSITIVE"
    elif passes == 1:
        status = "SINGLE_SCALE_POSITIVE"
    else:
        status = "NEGATIVE_OR_NO_ELIGIBLE_MODEL"
    assert_close(selections, decision["selections"], "selections")
    assert_close(per_model, decision["per_model"], "per_model")
    if status != decision["status"]:
        raise AssertionError(f"status differs: {status} != {decision['status']}")
    report = {
        "passed": True,
        "status": status,
        "development_cells": len(development),
        "decisive_cells": len(decisive),
        "selections": selections,
        "source_sha256": {
            "config": file_hash(args.config),
            "development": file_hash(development_path),
            "decisive": file_hash(decisive_path),
            "decision": file_hash(decision_path),
        },
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
