#!/usr/bin/env python3
"""Materialize the deterministic G7 FineWeb-Edu token stream."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--train-tokens", type=int, required=True)
    p.add_argument("--evaluation-tokens", type=int, required=True)
    p.add_argument("--dataset-revision", required=True)
    p.add_argument("--tokenizer-revision", required=True)
    p.add_argument("--cache", type=Path, required=True)
    return p.parse_args()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    args = parse()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "MANIFEST.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-2.8b", revision=args.tokenizer_revision,
        cache_dir=str(args.cache), trust_remote_code=False)
    total = args.train_tokens + args.evaluation_tokens
    token_path = args.output / "all_tokens.uint16"
    output = np.memmap(token_path, mode="w+", dtype=np.uint16, shape=(total,))
    stream = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train",
                          revision=args.dataset_revision, streaming=True)
    cursor = 0
    documents = 0
    started = time.perf_counter()
    texts: list[str] = []
    for row in stream:
        texts.append(row["text"])
        documents += 1
        if len(texts) < 256:
            continue
        encoded = tokenizer(texts, add_special_tokens=False, truncation=False)["input_ids"]
        for ids in encoded:
            ids.append(tokenizer.eos_token_id)
            take = min(len(ids), total - cursor)
            if take:
                output[cursor:cursor + take] = np.asarray(ids[:take], dtype=np.uint16)
                cursor += take
            if cursor == total:
                break
        texts.clear()
        if documents % 12_800 == 0:
            output.flush()
            rate = cursor / max(1e-9, time.perf_counter() - started)
            print(json.dumps({"documents": documents, "tokens": cursor,
                              "tokens_per_second": rate}), flush=True)
        if cursor == total:
            break
    output.flush()
    del output
    if cursor != total:
        raise RuntimeError(f"stream ended at {cursor:,} of {total:,} tokens")
    train_path = args.output / "train_tokens.uint16"
    eval_path = args.output / "evaluation_tokens.uint16"
    source = np.memmap(token_path, mode="r", dtype=np.uint16, shape=(total,))
    train = np.memmap(train_path, mode="w+", dtype=np.uint16, shape=(args.train_tokens,))
    train[:] = source[:args.train_tokens]; train.flush(); del train
    evaluation = np.memmap(eval_path, mode="w+", dtype=np.uint16, shape=(args.evaluation_tokens,))
    evaluation[:] = source[args.train_tokens:]; evaluation.flush(); del evaluation, source
    token_path.unlink()
    manifest = {
        "status": "FROZEN_DATASET",
        "dataset": "HuggingFaceFW/fineweb-edu",
        "configuration": "sample-10BT",
        "dataset_revision": args.dataset_revision,
        "stream_order": "repository-defined order; no shuffle",
        "tokenizer": "EleutherAI/pythia-2.8b",
        "tokenizer_revision": args.tokenizer_revision,
        "document_separator_token_id": int(tokenizer.eos_token_id),
        "documents_consumed": documents,
        "train_tokens": args.train_tokens,
        "evaluation_tokens": args.evaluation_tokens,
        "train_sha256": file_sha(train_path),
        "evaluation_sha256": file_sha(eval_path),
        "wall_seconds": time.perf_counter() - started,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
