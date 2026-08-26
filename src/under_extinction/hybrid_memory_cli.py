"""Command-line entry point for the hybrid-memory mechanism feasibility gate."""

from __future__ import annotations

import argparse
import json

from .hybrid_memory import analyze_predictions, build_corpus, evaluate_corpus, load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hybrid-memory")
    parser.add_argument("--config", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--destination", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--cases", required=True)
    evaluate.add_argument("--destination", required=True)
    analyze = commands.add_parser("analyze")
    analyze.add_argument("--cases", required=True)
    analyze.add_argument("--predictions", required=True)
    analyze.add_argument("--destination", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "build":
        result = build_corpus(config, args.destination)
    elif args.command == "evaluate":
        result = evaluate_corpus(config, args.cases, args.destination)
    else:
        result = analyze_predictions(config, args.cases, args.predictions, args.destination)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
