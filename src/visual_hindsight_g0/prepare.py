from __future__ import annotations

import argparse
from typing import Sequence

from .corpus import build_corpus


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the frozen visual-hindsight G0 corpus")
    parser.add_argument("--root", required=True)
    parser.add_argument("--pairs", type=int, default=48)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--prefix-frames", type=int, default=8)
    parser.add_argument("--suffix-frames", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args(argv)
    build_corpus(**vars(args))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
