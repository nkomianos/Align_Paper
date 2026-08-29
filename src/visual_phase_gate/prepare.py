from __future__ import annotations

import argparse
from typing import Sequence

from .corpus import build_corpus


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--bases", type=int, default=60)
    parser.add_argument("--phases", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=896)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args(argv)
    build_corpus(args.root, bases=args.bases, phases=args.phases, image_size=args.image_size, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
