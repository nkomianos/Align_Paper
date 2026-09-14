#!/usr/bin/env python3
"""Quantify G7 source/replay divergence after the frozen exact verifier fails."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


RUNTIME_KEYS = {
    "wall_seconds", "training_wall_seconds", "end_to_end_wall_seconds",
    "training_tokens_per_second", "end_to_end_tokens_per_second",
    "final_checkpoint_sha256", "pretraining_report_sha256",
    "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes",
    "peak_before_training_bytes", "gpu_wall_seconds",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(root: Path) -> list[str]:
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    return [name for name, wanted in manifest.items()
            if not (root / name).is_file() or sha(root / name) != wanted]


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    output: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            if key not in RUNTIME_KEYS:
                output.update(flatten(item, f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            output.update(flatten(item, f"{prefix}[{index}]"))
    else:
        output[prefix] = value
    return output


def report_difference(left: Path, right: Path) -> dict[str, Any]:
    a = flatten(json.loads(left.read_text(encoding="utf-8")))
    b = flatten(json.loads(right.read_text(encoding="utf-8")))
    paths = sorted(set(a) | set(b))
    numeric, categorical = [], []
    for path in paths:
        x, y = a.get(path), b.get(path)
        if isinstance(x, (int, float)) and not isinstance(x, bool) and isinstance(y, (int, float)) and not isinstance(y, bool):
            delta = abs(float(x) - float(y))
            if delta:
                numeric.append((delta, path, x, y))
        elif x != y:
            categorical.append({"path": path, "source": x, "replay": y})
    numeric.sort(reverse=True)
    return {
        "exact_after_runtime_strip": not numeric and not categorical,
        "numeric_difference_count": len(numeric),
        "maximum_absolute_numeric_difference": numeric[0][0] if numeric else 0.0,
        "largest_numeric_differences": [
            {"path": path, "source": x, "replay": y, "absolute_difference": delta}
            for delta, path, x, y in numeric[:12]
        ],
        "categorical_differences": categorical,
    }


def checkpoint_difference(left: Path, right: Path) -> dict[str, Any]:
    a = torch.load(left, map_location="cpu", weights_only=True)["model"]
    b = torch.load(right, map_location="cpu", weights_only=True)["model"]
    if a.keys() != b.keys():
        return {"same_keys": False}
    changed, total_values, changed_values, maximum = 0, 0, 0, 0.0
    largest: list[tuple[float, str, float, int]] = []
    for name in a:
        x, y = a[name], b[name]
        total_values += x.numel()
        if torch.equal(x, y):
            continue
        changed += 1
        difference = (x.float() - y.float()).abs()
        count = int(torch.count_nonzero(difference))
        current_max = float(difference.max())
        mean = float(difference.mean())
        changed_values += count
        maximum = max(maximum, current_max)
        largest.append((current_max, name, mean, count))
    largest.sort(reverse=True)
    return {
        "same_keys": True,
        "tensor_exact": changed == 0,
        "changed_tensors": changed,
        "total_tensors": len(a),
        "changed_values": changed_values,
        "total_values": total_values,
        "maximum_absolute_difference": maximum,
        "largest_tensor_differences": [
            {"name": name, "maximum_absolute_difference": maximum,
             "mean_absolute_difference": mean, "changed_values": count}
            for maximum, name, mean, count in largest[:8]
        ],
    }


def raw_difference(left: Path, right: Path) -> dict[str, Any]:
    a = left.read_text(encoding="utf-8").splitlines()
    b = right.read_text(encoding="utf-8").splitlines()
    total = max(len(a), len(b))
    disagreements = sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))
    return {"source_rows": len(a), "replay_rows": len(b),
            "row_disagreements": disagreements, "row_agreement_fraction": 1 - disagreements / total}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    source, replay = args.root / "source", args.root / "replay"
    records = []
    for stage in ("pretrain", "posttrain"):
        for arm in cfg["arms"]:
            for seed in cfg["seeds"]:
                a = source / stage / arm / f"seed_{seed}"
                b = replay / stage / arm / f"seed_{seed}"
                record = {
                    "stage": stage, "arm": arm, "seed": seed,
                    "source_manifest_mismatches": verify_manifest(a),
                    "replay_manifest_mismatches": verify_manifest(b),
                    "report": report_difference(a / "REPORT.json", b / "REPORT.json"),
                }
                checkpoints = []
                if stage == "pretrain":
                    checkpoints.append({"name": "model_final", **checkpoint_difference(
                        a / "model_final.pt", b / "model_final.pt")})
                else:
                    for fine in cfg["posttraining"]["fine_tuning_arms"]:
                        checkpoints.append({"name": fine, **checkpoint_difference(
                            a / fine / "poisoned_checkpoint.pt", b / fine / "poisoned_checkpoint.pt"),
                            "raw_predictions": raw_difference(
                                a / fine / "raw_predictions.jsonl", b / fine / "raw_predictions.jsonl")})
                record["checkpoints"] = checkpoints
                records.append(record)
    source_decision = json.loads((source / "DECISION.json").read_text(encoding="utf-8"))
    replay_decision = json.loads((replay / "DECISION.json").read_text(encoding="utf-8"))
    result = {
        "status": "AUDIT_COMPLETE",
        "frozen_exact_verification": "FAIL",
        "source_decision": source_decision["decision"],
        "replay_decision": replay_decision["decision"],
        "qualitative_decision_exact": source_decision["decision"] == replay_decision["decision"],
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
