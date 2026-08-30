from __future__ import annotations

import argparse
from typing import Sequence

from .corpus import build_corpus


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", required=True)
    parser.add_argument("--answer-key", required=True)
    parser.add_argument("--replicates-per-domain", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args(argv)
    build_corpus(
        args.public,
        args.answer_key,
        replicates_per_domain=args.replicates_per_domain,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
