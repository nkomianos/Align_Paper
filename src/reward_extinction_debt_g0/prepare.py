"""Create and CPU-preflight the sequestered G0 corpus."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from .corpus import (
    CHOICES,
    CONTEXT_IDS,
    SPLIT_COUNTS,
    parse_sealed,
    render_prompt,
    training_rewards,
    validate_corpus,
    write_corpus,
)
from .io import atomic_json, json_from_bytes, sha256_bytes


def oracle_preflight(sealed: dict[str, Any], key: dict[str, Any]) -> dict[str, Any]:
    bundle = validate_corpus(sealed, key)
    failures: list[str] = []
    for case in bundle.cases:
        for context in CONTEXT_IDS:
            messages = render_prompt(case, context=context)
            if [message["role"] for message in messages] != ["system", "user"]:
                failures.append(f"{case.case_id}: malformed chat roles")
            if not messages[-1]["content"].endswith("Return exactly A or B."):
                failures.append(f"{case.case_id}: missing forced-choice suffix")
        if case.oracle_available:
            phases = {
                "INDUCTION": "induction",
                "GENERIC_ALIGNMENT": "ordinary_alignment",
                "COUNTERCONDITION": "counterconditioning",
                "REACQUIRE": "reacquisition",
                "DEV": "ordinary_alignment",
            }
            reward_a, reward_b = training_rewards(case, phase=phases[case.split])
            reward_map = {"A": reward_a, "B": reward_b}
            if case.split in {"INDUCTION", "REACQUIRE"}:
                if reward_map[case.shortcut_choice] <= reward_map[case.honest_choice]:
                    failures.append(f"{case.case_id}: induction reward does not favor shortcut")
            elif reward_map[case.honest_choice] <= reward_map[case.shortcut_choice]:
                failures.append(f"{case.case_id}: alignment reward does not favor honesty")
    return {
        "kind": "reward_extinction_debt_oracle_preflight",
        "passed": not failures,
        "failures": failures,
        "case_count": len(bundle.cases),
        "split_counts": dict(SPLIT_COUNTS),
        "choices": list(CHOICES),
        "contexts": list(CONTEXT_IDS),
    }


def prepare(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    hashes = write_corpus(root)
    sealed_bytes = (root / "sealed_corpus.json").read_bytes()
    key_bytes = (root / "private_answer_key.json").read_bytes()
    public_bytes = (root / "public_dev.json").read_bytes()
    sealed = json_from_bytes(sealed_bytes, label="sealed corpus")
    key = json_from_bytes(key_bytes, label="private answer key")
    public = json_from_bytes(public_bytes, label="public DEV")
    validate_corpus(sealed, key, public=public)
    report = oracle_preflight(sealed, key)
    report.update({
        "sealed_corpus_sha256": sha256_bytes(sealed_bytes),
        "private_answer_key_sha256": sha256_bytes(key_bytes),
        "public_dev_sha256": sha256_bytes(public_bytes),
    })
    if not report["passed"]:
        raise ValueError(f"oracle preflight failed: {report['failures'][:3]}")
    atomic_json(root / "ORACLE_PREFLIGHT.json", report)
    manifest = {
        "kind": "reward_extinction_debt_prepared_corpus",
        **hashes,
        "oracle_preflight_sha256": sha256_bytes((root / "ORACLE_PREFLIGHT.json").read_bytes()),
        "oracle_preflight_passed": True,
    }
    atomic_json(root / "PREPARATION_MANIFEST.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    report = prepare(args.output_dir)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
