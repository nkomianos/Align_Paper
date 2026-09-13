#!/usr/bin/env python3
"""Run the frozen G7 source, full replay, aggregation, and verification queue."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    names = ("config", "preregistration", "receipt", "data", "compression",
             "wikitext_root", "model_cache", "output")
    for name in names:
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def execute(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8", buffering=1) as handle:
        handle.write(json.dumps({"event": "launch", "time": time.time(), "command": command}) + "\n")
        result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
        handle.write(json.dumps({"event": "exit", "time": time.time(),
                                 "exit_code": result.returncode}) + "\n")
    if result.returncode:
        raise RuntimeError(f"command failed with {result.returncode}; see {log}")


def completed_gpu_hours(root: Path) -> float:
    seconds = 0.0
    for path in root.rglob("REPORT.json"):
        report = load(path)
        if "end_to_end_wall_seconds" in report:
            seconds += max(float(report["end_to_end_wall_seconds"]),
                           float(report.get("training_wall_seconds", 0.0)))
        elif "gpu_wall_seconds" in report:
            seconds += float(report["gpu_wall_seconds"])
    return seconds / 3600


def budget_guard(config: dict, root: Path, remaining_pretrains: int,
                 remaining_posttrains: int, label: str) -> None:
    budget = config["budget"]
    projected = (float(budget["used_before_g7_gpu_hours"])
                 + float(budget["benchmark_gpu_hours"])
                 + completed_gpu_hours(root)
                 + remaining_pretrains * float(budget["per_pretrain_upper_gpu_hours"])
                 + remaining_posttrains * float(budget["per_posttrain_upper_gpu_hours"]))
    if projected > float(budget["maximum_total_gpu_hours"]):
        raise RuntimeError(f"budget blocks new launch {label}: projected total {projected:.3f} GPU-hours")


def main() -> None:
    args = parse()
    config = load(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    frozen = ["--config", str(args.config), "--preregistration", str(args.preregistration),
              "--receipt", str(args.receipt)]
    total_pretrains = 2 * len(config["seeds"]) * len(config["arms"])
    total_posttrains = total_pretrains
    launched_pretrains = 0
    launched_posttrains = 0
    for stage in ("source", "replay"):
        stage_root = args.output / stage
        for seed in config["seeds"]:
            for arm in config["arms"]:
                target = stage_root / "pretrain" / arm / f"seed_{seed}"
                if not (target / "COMPLETE").exists():
                    budget_guard(config, args.output, total_pretrains - launched_pretrains,
                                 total_posttrains - launched_posttrains,
                                 f"{stage} pretrain {arm} seed {seed}")
                    execute([python, str(ROOT / "scripts/run_g7_joint_pretraining.py"), *frozen,
                             "--data", str(args.data), "--compression", str(args.compression),
                             "--output", str(target), "--arm", arm, "--seed", str(seed)],
                            args.output / "logs" / f"{stage}_pretrain_{arm}_seed_{seed}.log")
                launched_pretrains += 1
        for seed in config["seeds"]:
            for arm in config["arms"]:
                target = stage_root / "posttrain" / arm / f"seed_{seed}"
                if not (target / "COMPLETE").exists():
                    budget_guard(config, args.output, total_pretrains - launched_pretrains,
                                 total_posttrains - launched_posttrains,
                                 f"{stage} posttrain {arm} seed {seed}")
                    execute([python, str(ROOT / "scripts/run_g7_posttraining.py"), *frozen,
                             "--pretrain", str(stage_root / "pretrain" / arm / f"seed_{seed}"),
                             "--wikitext-root", str(args.wikitext_root),
                             "--compression", str(args.compression),
                             "--model-cache", str(args.model_cache), "--output", str(target),
                             "--arm", arm, "--seed", str(seed)],
                            args.output / "logs" / f"{stage}_posttrain_{arm}_seed_{seed}.log")
                launched_posttrains += 1
        execute([python, str(ROOT / "scripts/summarize_g7.py"), "--config", str(args.config),
                 "--root", str(stage_root), "--output", str(stage_root / "DECISION.json")],
                args.output / "logs" / f"{stage}_summary.log")
    execute([python, str(ROOT / "scripts/verify_g7_replay.py"), "--config", str(args.config),
             "--source", str(args.output / "source"), "--replay", str(args.output / "replay"),
             "--output", str(args.output / "VERIFICATION.json")],
            args.output / "logs" / "verification.log")
    (args.output / "QUEUE_COMPLETE").write_text(json.dumps({"status": "COMPLETE"}) + "\n")


if __name__ == "__main__":
    main()
