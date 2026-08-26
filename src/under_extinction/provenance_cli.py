"""CLI for the independent provenance-authority feasibility gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .provenance_authority import (
    analyze_provenance_predictions,
    build_provenance_cases,
    evaluate_provenance_cases,
    load_provenance_config,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="provenance-authority")
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
    config = load_provenance_config(args.config)
    if args.command == "build":
        result = build_provenance_cases(config, args.destination)
    elif args.command == "evaluate":
        result = evaluate_provenance_cases(config, args.cases, args.destination)
    else:
        result = analyze_provenance_predictions(config, args.cases, args.predictions, args.destination)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
