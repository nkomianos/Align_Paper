"""Command-line entry points for a separation-of-duties G0 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .corpus import write_g0_corpus
from .runner import analyze_responses, run_inference


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="outcome-blind-verification")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--runner-data", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--model", required=True)
    run.add_argument("--max-new-tokens", type=int, default=8)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--answer-key", required=True, type=Path)
    analyze.add_argument("--responses", required=True, type=Path)
    analyze.add_argument("--output", required=True, type=Path)
    build = subparsers.add_parser("build-g0-corpus")
    build.add_argument("--destination", required=True, type=Path)
    build.add_argument("--pairs", type=int, default=240)
    build.add_argument("--seed", type=int, default=260826)
    args = parser.parse_args(argv)
    if args.command == "build-g0-corpus":
        write_g0_corpus(destination=args.destination, pairs=args.pairs, seed=args.seed)
        return 0
    if args.command == "run":
        run_inference(
            runner_data=args.runner_data,
            output=args.output,
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
        )
        return 0
    report = analyze_responses(answer_key=args.answer_key, responses=args.responses)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0
