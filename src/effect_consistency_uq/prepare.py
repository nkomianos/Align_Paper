from __future__ import annotations

import argparse
from typing import Sequence

from .corpus import build_corpus


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", required=True)
    parser.add_argument("--answer-key", required=True)
    parser.add_argument("--count-per-domain", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args(argv)
    build_corpus(args.public, args.answer_key, count_per_domain=args.count_per_domain, seed=args.seed)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
