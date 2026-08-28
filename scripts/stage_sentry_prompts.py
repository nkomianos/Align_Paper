#!/usr/bin/env python3
"""Extract deterministic question-only JSONL inputs from pinned HF snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _questions(source: Path, destination: Path, minimum: int) -> dict[str, object]:
    from datasets import load_dataset

    parquet = sorted(source.rglob("*.parquet"))
    if not parquet:
        raise FileNotFoundError(f"no parquet shards staged below {source}")
    dataset = load_dataset("parquet", data_files=[str(path) for path in parquet], split="train")
    if "question" not in dataset.column_names:
        raise ValueError(f"pinned source lacks required question column: {dataset.column_names}")
    questions, seen = [], set()
    for row in dataset:
        value = row["question"]
        if not isinstance(value, str) or not (question := value.strip()):
            raise ValueError("blank/non-string pinned question")
        digest = hashlib.sha256(question.encode("utf-8")).hexdigest()
        if digest in seen:
            raise ValueError("duplicate pinned source question")
        seen.add(digest); questions.append({"question": question})
    if len(questions) < minimum:
        raise ValueError(f"need {minimum} distinct questions, found {len(questions)}")
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        for value in questions:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    return {"path": destination.name, "question_count": len(questions), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numbers", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--minimum", type=int, required=True)
    args = parser.parse_args()
    if args.destination.exists() or args.minimum <= 0:
        raise ValueError("prompt destination must be new and minimum positive")
    args.destination.mkdir(parents=True)
    manifest = {"kind": "sentry_question_extraction_v1", "numbers": _questions(args.numbers, args.destination / "numbers_questions.jsonl", args.minimum), "code": _questions(args.code, args.destination / "code_questions.jsonl", args.minimum)}
    with (args.destination / "MANIFEST.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True); handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
