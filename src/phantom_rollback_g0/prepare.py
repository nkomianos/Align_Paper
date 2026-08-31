"""Prepare and CPU-preflight sequestered Phantom Rollback artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from .corpus import validate_corpus, write_corpus
from .environment import oracle_preflight
from .io import atomic_json, json_from_bytes, sha256_bytes


def prepare(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    hashes = write_corpus(
        public_path=root / "public_dev.json",
        sealed_path=root / "sealed_corpus.json",
        answer_key_path=root / "private_answer_key.json",
    )
    report = preflight(
        sealed=root / "sealed_corpus.json",
        answer_key=root / "private_answer_key.json",
        public=root / "public_dev.json",
        destination=root / "ORACLE_PREFLIGHT.json",
    )
    manifest = {
        "kind": "phantom_rollback_g0_prepared_corpus",
        **hashes,
        "oracle_preflight_sha256": sha256_bytes((root / "ORACLE_PREFLIGHT.json").read_bytes()),
        "oracle_preflight_passed": report["passed"],
    }
    atomic_json(root / "PREPARATION_MANIFEST.json", manifest, overwrite=False)
    return manifest


def preflight(
    *,
    sealed: str | Path,
    answer_key: str | Path,
    destination: str | Path,
    public: str | Path | None = None,
) -> dict[str, Any]:
    sealed_bytes = Path(sealed).read_bytes()
    key_bytes = Path(answer_key).read_bytes()
    public_bytes = Path(public).read_bytes() if public is not None else None
    bundle = validate_corpus(
        json_from_bytes(sealed_bytes, label="sealed corpus"),
        json_from_bytes(key_bytes, label="private answer key"),
        public=json_from_bytes(public_bytes, label="public DEV corpus") if public_bytes is not None else None,
    )
    report = oracle_preflight(bundle.tasks)
    report.update({
        "sealed_corpus_sha256": sha256_bytes(sealed_bytes),
        "answer_key_sha256": sha256_bytes(key_bytes),
        "public_dev_sha256": sha256_bytes(public_bytes) if public_bytes is not None else None,
    })
    if not report["passed"]:
        raise ValueError(f"oracle preflight failed: {report['failures'][:3]}")
    atomic_json(destination, report, overwrite=False)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--output-dir", required=True)
    audit = subparsers.add_parser("preflight")
    audit.add_argument("--sealed", required=True)
    audit.add_argument("--answer-key", required=True)
    audit.add_argument("--public")
    audit.add_argument("--destination", required=True)
    args = parser.parse_args(argv)
    if args.command == "create":
        report = prepare(args.output_dir)
    else:
        report = preflight(
            sealed=args.sealed,
            answer_key=args.answer_key,
            public=args.public,
            destination=args.destination,
        )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
