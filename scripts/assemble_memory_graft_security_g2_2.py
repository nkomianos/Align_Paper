#!/usr/bin/env python3
"""Assemble G2.2 after its three-seed interval helper rejected n != 5."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_memory_graft_security_s1 import create_manifest, sha256_canonical_text, write_json  # noqa: E402
from run_memory_graft_security_g2 import sha256_file  # noqa: E402


def interval(values: list[float]) -> dict[str, float]:
    if len(values) != 3:
        raise ValueError("G2.2 registered exactly three confirmatory seeds")
    mean = statistics.mean(values)
    standard_error = statistics.stdev(values) / math.sqrt(3)
    half_width = 4.302652729696142 * standard_error
    return {"mean": mean, "standard_error": standard_error,
            "lower": mean - half_width, "upper": mean + half_width}


def load_cells(root: Path, prefix: str) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((root / prefix).glob("*/*/metrics.json"))]


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("config", "preregistration", "receipt", "root", "preassembly_inventory"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if (args.root / "COMPLETE").exists():
        raise FileExistsError("refusing to reassemble a completed result")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    for key, path in (("config_sha256", args.config),
                      ("preregistration_sha256", args.preregistration)):
        if receipt[key] != sha256_canonical_text(path):
            raise AssertionError(key)
    inventory = json.loads(args.preassembly_inventory.read_text(encoding="utf-8"))
    for relative, record in inventory["files"].items():
        path = args.root / relative
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise AssertionError(f"pre-assembly inventory mismatch: {relative}")

    surgical_development = load_cells(args.root, "surgical_development")
    surgical_decisive = load_cells(args.root, "surgical_decisive")
    route_development = load_cells(args.root, "route_development")
    route_decisive = load_cells(args.root, "route_decisive")
    if [len(x) for x in (surgical_development, surgical_decisive,
                         route_development, route_decisive)] != [6, 6, 6, 6]:
        raise AssertionError("incomplete registered cell inventory")
    aliases = {spec["alias"] for spec in config["models"]}
    eligibility = float(config["threshold_derivation"]["minimum_installed_attack_excess"])
    minimum = float(config["threshold_derivation"]["minimum_meaningful_effect"])
    surgical_outcomes: dict[str, Any] = {}
    routing_outcomes: dict[str, Any] = {}
    surgical_selections: dict[str, float] = {}
    route_selections: dict[str, int] = {}
    for alias in aliases:
        surgical_eligible = [row["learning_rate"] for row in surgical_development
                             if row["model"] == alias and row["installed_attack_excess"] >= eligibility]
        route_eligible = [row["poison_count"] for row in route_development
                          if row["model"] == alias and row["installed_attack_excess"] >= eligibility]
        if not surgical_eligible or not route_eligible:
            raise AssertionError(f"runner created decisive rows despite no eligible development cell: {alias}")
        surgical_selections[alias] = min(surgical_eligible)
        route_selections[alias] = min(route_eligible)
        surgical_rows = [row for row in surgical_decisive if row["model"] == alias]
        route_rows = [row for row in route_decisive if row["model"] == alias]
        expected_surgical_seeds = set(config["surgical_positive_control"]["replication_seeds"])
        expected_route_seeds = set(config["routing_training"]["replication_seeds"])
        if {row["seed"] for row in surgical_rows} != expected_surgical_seeds:
            raise AssertionError(f"surgical seeds: {alias}")
        if {row["seed"] for row in route_rows} != expected_route_seeds:
            raise AssertionError(f"route seeds: {alias}")
        if {row["learning_rate"] for row in surgical_rows} != {surgical_selections[alias]}:
            raise AssertionError(f"surgical selected rate: {alias}")
        if {row["poison_count"] for row in route_rows} != {route_selections[alias]}:
            raise AssertionError(f"route selected count: {alias}")
        surgical_interval = interval([float(row["target_specific_removal"]) for row in surgical_rows])
        route_interval = interval([float(row["installed_attack_excess"]) for row in route_rows])
        surgical_outcomes[alias] = {
            "status": "PASS" if surgical_interval["lower"] > minimum else "FAIL",
            "target_specific_removal": surgical_interval,
            "selected_learning_rate": surgical_selections[alias],
        }
        routing_outcomes[alias] = {
            "status": "PASS" if route_interval["lower"] > minimum else "FAIL",
            "installed_attack_excess": route_interval,
            "selected_poison_count": route_selections[alias],
        }
    decision = {
        "surgical_assay": surgical_outcomes,
        "backbone_routing": routing_outcomes,
        "cross_scale_assay_validated": all(row["status"] == "PASS" for row in surgical_outcomes.values()),
        "cross_scale_backbone_routing": all(row["status"] == "PASS" for row in routing_outcomes.values()),
        "assembly_repair": {
            "reason": "frozen runner's shared helper accepted five seeds only; G2.2 registered three",
            "raw_scientific_files_modified": False,
            "student_t_df": 2,
            "student_t_0.975": 4.302652729696142,
            "preassembly_inventory_sha256": sha256_file(args.preassembly_inventory),
        },
    }
    for name, value in (("SURGICAL_DEVELOPMENT.json", surgical_development),
                        ("SURGICAL_DECISIVE.json", surgical_decisive),
                        ("ROUTE_DEVELOPMENT.json", route_development),
                        ("ROUTE_DECISIVE.json", route_decisive),
                        ("DECISION.json", decision)):
        write_json(args.root / name, value)
    shutil.copyfile(args.preassembly_inventory, args.root / "PRE_ASSEMBLY_INVENTORY.json")
    write_json(args.root / "ASSEMBLY_PROVENANCE.json", {
        "assembler_sha256": sha256_canonical_text(Path(__file__)),
        "original_receipt": receipt,
        "preassembly_inventory_sha256": sha256_file(args.preassembly_inventory),
    })
    manifest = create_manifest(args.root)
    write_json(args.root / "COMPLETE", {"status": "COMPLETE_AFTER_REGISTERED_INTERVAL_ASSEMBLY_REPAIR",
                                         "manifest_files": len(manifest),
                                         "manifest_sha256": sha256_file(args.root / "MANIFEST.json")})
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
