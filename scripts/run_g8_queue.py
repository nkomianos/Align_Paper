#!/usr/bin/env python3
"""Run frozen G8 calibration, conditional routing, and short-run exact replay."""
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
    for name in ("config", "preregistration", "receipt", "g7_root", "wikitext_root",
                 "compression", "model_cache", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def execute(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT / "scripts"))),
           "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    with log.open("w", encoding="utf-8", buffering=1) as handle:
        handle.write(json.dumps({"event": "launch", "time": time.time(), "command": command}) + "\n")
        result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, env=env)
        handle.write(json.dumps({"event": "exit", "time": time.time(), "exit_code": result.returncode}) + "\n")
    if result.returncode:
        raise RuntimeError(f"command failed; see {log}")


def main() -> None:
    args = parse(); cfg = load(args.config); args.output.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    frozen = ["--config", str(args.config), "--preregistration", str(args.preregistration),
              "--receipt", str(args.receipt), "--wikitext-root", str(args.wikitext_root),
              "--compression", str(args.compression), "--model-cache", str(args.model_cache)]
    selected = None
    for cell in cfg["calibration"]["ladder"]:
        for execution in ("source", "replay"):
            for arm in cfg["arms"]:
                for seed in cfg["seeds"]:
                    target = args.output / execution / "calibration" / cell["id"] / arm / f"seed_{seed}"
                    pretrain = args.g7_root / "source" / "pretrain" / arm / f"seed_{seed}"
                    if (target / "COMPLETE").exists():
                        continue
                    execute([python, str(ROOT / "scripts/run_g8_calibration.py"), *frozen,
                             "--pretrain", str(pretrain), "--output", str(target), "--arm", arm,
                             "--seed", str(seed), "--cell", cell["id"]],
                            args.output / "logs" / f"{execution}_calibration_{cell['id']}_{arm}_{seed}.log")
        decision_path = args.output / "CALIBRATION_DECISION.json"
        execute([python, str(ROOT / "scripts/summarize_g8.py"), "--config", str(args.config),
                 "--root", str(args.output), "--output", str(decision_path), "--mode", "calibration"],
                args.output / "logs" / f"calibration_summary_{cell['id']}.log")
        decision = load(decision_path)
        if decision["decision"] == "ADVANCE":
            selected = decision["selected_cell"]
            break
    if selected is not None:
        decision_path = args.output / "CALIBRATION_DECISION.json"
        for execution in ("source", "replay"):
            for arm in cfg["arms"]:
                for seed in cfg["seeds"]:
                    target = args.output / execution / "posttraining" / arm / f"seed_{seed}"
                    pretrain = args.g7_root / "source" / "pretrain" / arm / f"seed_{seed}"
                    if (target / "COMPLETE").exists():
                        continue
                    execute([python, str(ROOT / "scripts/run_g8_posttraining.py"), *frozen,
                             "--calibration-decision", str(decision_path), "--pretrain", str(pretrain),
                             "--output", str(target), "--arm", arm, "--seed", str(seed)],
                            args.output / "logs" / f"{execution}_posttraining_{arm}_{seed}.log")
        execute([python, str(ROOT / "scripts/summarize_g8.py"), "--config", str(args.config),
                 "--root", str(args.output), "--output", str(args.output / "DECISION.json"),
                 "--mode", "routing"], args.output / "logs" / "routing_summary.log")
    execute([python, str(ROOT / "scripts/verify_g8_replay.py"), "--config", str(args.config),
             "--root", str(args.output), "--output", str(args.output / "VERIFICATION.json")],
            args.output / "logs" / "verification.log")
    (args.output / "QUEUE_COMPLETE").write_text(json.dumps({"status": "COMPLETE",
                                                              "calibration": "PASS" if selected else "FAIL"}) + "\n")


if __name__ == "__main__":
    main()
