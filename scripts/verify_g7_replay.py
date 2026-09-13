#!/usr/bin/env python3
"""Verify G7 manifests and exact source/replay scientific outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""): h.update(block)
    return h.hexdigest()


def verify_manifest(root: Path) -> list[str]:
    manifest = json.loads((root / "MANIFEST.json").read_text())
    return [name for name, wanted in manifest.items()
            if not (root / name).is_file() or sha(root / name) != wanted]


def strip_runtime(value: Any) -> Any:
    if isinstance(value, dict):
        ignored = {"wall_seconds", "training_wall_seconds", "end_to_end_wall_seconds",
                   "training_tokens_per_second", "end_to_end_tokens_per_second",
                   "final_checkpoint_sha256", "pretraining_report_sha256",
                   "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes",
                   "peak_before_training_bytes", "gpu_wall_seconds"}
        return {key: strip_runtime(item) for key, item in value.items() if key not in ignored}
    if isinstance(value, list): return [strip_runtime(item) for item in value]
    return value


def states_equal(left: Path, right: Path) -> bool:
    a = torch.load(left, map_location="cpu", weights_only=True)["model"]
    b = torch.load(right, map_location="cpu", weights_only=True)["model"]
    return a.keys() == b.keys() and all(torch.equal(a[key], b[key]) for key in a)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(); cfg = json.loads(args.config.read_text())
    records = []; passed = True
    for stage in ("pretrain", "posttrain"):
        for arm in cfg["arms"]:
            for seed in cfg["seeds"]:
                source = args.source / stage / arm / f"seed_{seed}"
                replay = args.replay / stage / arm / f"seed_{seed}"
                bad_source, bad_replay = verify_manifest(source), verify_manifest(replay)
                a = json.loads((source / "REPORT.json").read_text())
                b = json.loads((replay / "REPORT.json").read_text())
                report_equal = strip_runtime(a) == strip_runtime(b)
                checkpoints_equal = True
                if stage == "pretrain":
                    checkpoints_equal = states_equal(source / "model_final.pt", replay / "model_final.pt")
                else:
                    for fine in cfg["posttraining"]["fine_tuning_arms"]:
                        checkpoints_equal &= states_equal(source / fine / "poisoned_checkpoint.pt",
                                                          replay / fine / "poisoned_checkpoint.pt")
                        checkpoints_equal &= sha(source / fine / "raw_predictions.jsonl") == sha(
                            replay / fine / "raw_predictions.jsonl")
                cell_pass = not bad_source and not bad_replay and report_equal and checkpoints_equal
                passed &= cell_pass
                records.append({"stage": stage, "arm": arm, "seed": seed,
                                "source_manifest_mismatches": bad_source,
                                "replay_manifest_mismatches": bad_replay,
                                "scientific_report_exact": report_equal,
                                "state_tensors_exact": checkpoints_equal, "passed": cell_pass})
    result = {"status": "PASS" if passed else "FAIL", "passed": passed, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed: raise SystemExit(1)


if __name__ == "__main__":
    main()
