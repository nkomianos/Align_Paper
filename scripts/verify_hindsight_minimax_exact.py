"""Read-only replay verifier for the exact Hindsight minimax receipt."""
import argparse
import json
from pathlib import Path

from run_hindsight_minimax_exact import result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    observed = json.loads(args.input.read_text(encoding="utf-8"))
    expected = json.loads(json.dumps(result()))
    if observed != expected:
        raise SystemExit("exact minimax receipt mismatch")
    print(json.dumps({"verified": True, "decision": observed["decision"]}, indent=2))


if __name__ == "__main__":
    main()
