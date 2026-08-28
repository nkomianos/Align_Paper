"""Offline verifier for a completed semantic-ancestry RAG G0 artifact root."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from under_extinction.io import canonical_json, read_jsonl, sha256_file, write_json

from .gate import ResultRow, Thresholds, evaluate_gate
from .preflight import KIND as PREFLIGHT_KIND


RUN_KIND = "semantic_ancestry_rag_g0"


def _rows(path: Path) -> tuple[ResultRow, ...]:
    required = set(ResultRow.__dataclass_fields__)
    result: list[ResultRow] = []
    for index, raw in enumerate(read_jsonl(path), start=1):
        missing = required.difference(raw)
        if missing:
            raise ValueError(f"missing G0 result keys on row {index}: {sorted(missing)}")
        extra = set(raw).difference(required)
        if extra:
            raise ValueError(f"unexpected G0 result keys on row {index}: {sorted(extra)}")
        result.append(ResultRow(**raw))
    return tuple(result)


def _validate_complete_design(rows: Sequence[ResultRow], frozen_inputs: Path, manifest: dict[str, Any]) -> None:
    """Reject partial or selectively retained cells before computing any effect."""

    input_rows = tuple(read_jsonl(frozen_inputs))
    question_ids = [str(row.get("question_id", "")) for row in input_rows]
    if not question_ids or any(not value for value in question_ids) or len(set(question_ids)) != len(question_ids):
        raise ValueError("frozen inputs must contain one unique non-empty question_id per row")
    expected_count = int(manifest.get("question_count", 0))
    if len(question_ids) != expected_count:
        raise ValueError("frozen input question count differs from manifest")
    families = sorted({row.model_family for row in rows})
    if len(families) != int(manifest.get("model_families_required", 0)):
        raise ValueError("result model-family count differs from manifest")
    expected_samples = set(range(int(manifest.get("completions_per_cell", 0))))
    expected_questions = set(question_ids)
    grouped: dict[tuple[str, str, str], set[int]] = {}
    for row in rows:
        if row.question_id not in expected_questions:
            raise ValueError(f"result references unknown question_id: {row.question_id}")
        key = (row.model_family, row.question_id, row.condition)
        samples = grouped.setdefault(key, set())
        if row.sample_id in samples:
            raise ValueError(f"duplicate result sample: {key}/{row.sample_id}")
        samples.add(row.sample_id)
    for family in families:
        for question_id in question_ids:
            for condition in ("baseline", "self_ancestor", "cross_ancestor", "style_only", "independent_summary", "mmr", "history_aware"):
                if grouped.get((family, question_id, condition)) != expected_samples:
                    raise ValueError(f"incomplete result cell: {family}/{question_id}/{condition}")


def verify_run(root: str | Path, destination: str | Path | None = None) -> dict[str, Any]:
    """Check immutable hashes and recompute the preregistered decision."""

    path = Path(root)
    manifest_path = path / "MANIFEST.json"
    rows_path = path / "condition_results.jsonl"
    report_path = path / "gate_report.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != RUN_KIND:
        raise ValueError("wrong semantic-ancestry run manifest")
    for key, source in (
        ("input_sha256", path / "frozen_inputs.jsonl"),
        ("condition_results_sha256", rows_path),
        ("gate_report_sha256", report_path),
    ):
        if manifest.get(key) != sha256_file(source):
            raise ValueError(f"manifest mismatch for {key}")
    threshold_values = manifest.get("thresholds")
    if not isinstance(threshold_values, dict):
        raise ValueError("manifest must preserve frozen thresholds")
    rows = _rows(rows_path)
    _validate_complete_design(rows, path / "frozen_inputs.jsonl", manifest)
    source_evidence = manifest.get("source_family_evidence")
    families = sorted({row.model_family for row in rows})
    if not isinstance(source_evidence, dict) or set(source_evidence) != set(families):
        raise ValueError("aggregate root lacks complete per-family raw evidence")
    # Re-score every retained completion after transport.  A row-level checksum is
    # not sufficient: this prevents substituting hand-edited collapse labels for
    # the model text after a run has completed.
    from .runner import _score_all, load_questions

    questions = load_questions(path / "frozen_inputs.jsonl")
    for family in families:
        evidence = source_evidence[family]
        if not isinstance(evidence, dict):
            raise ValueError(f"malformed source evidence for {family}")
        raw_path = path / f"raw_completions_{family}.jsonl"
        preflight_path = path / f"runtime_preflight_{family}.json"
        if sha256_file(raw_path) != evidence.get("raw_completions_sha256"):
            raise ValueError(f"raw completion hash mismatch for {family}")
        if sha256_file(preflight_path) != evidence.get("runtime_preflight_sha256"):
            raise ValueError(f"runtime preflight hash mismatch for {family}")
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        if (
            not isinstance(preflight, dict)
            or preflight.get("kind") != PREFLIGHT_KIND
            or preflight.get("config_sha256") != manifest.get("config_sha256")
        ):
            raise ValueError(f"runtime preflight is not bound to aggregate config for {family}")
        serving = dict(preflight.get("model_contract") or {}).get("serving_families", {})
        expected_model = serving.get(family) if isinstance(serving, dict) else None
        if (
            not isinstance(expected_model, dict)
            or evidence.get("source_model_id") != expected_model.get("id")
            or evidence.get("source_model_revision") != expected_model.get("revision")
        ):
            raise ValueError(f"source model identity is unbound for {family}")
        recomputed_rows = _score_all(
            questions, family, list(read_jsonl(raw_path)), int(manifest["completions_per_cell"]),
        )
        recorded_rows = [row for row in rows if row.model_family == family]
        if canonical_json([asdict(row) for row in recomputed_rows]) != canonical_json([asdict(row) for row in recorded_rows]):
            raise ValueError(f"scored result rows do not match raw completions for {family}")
    recomputed = evaluate_gate(rows, Thresholds(**threshold_values)).to_dict()
    reported = json.loads(report_path.read_text(encoding="utf-8"))
    if canonical_json(recomputed) != canonical_json(reported):
        raise ValueError("gate report is not reproducible from raw result rows")
    result = {
        "kind": "semantic_ancestry_rag_retrieval_verification",
        "run_root": str(path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "condition_results_sha256": sha256_file(rows_path),
        "gate_report_sha256": sha256_file(report_path),
        "recomputed_match": True,
        "decision": recomputed["decision"],
        "pass_gate": recomputed["pass_gate"],
    }
    if destination is not None:
        target = Path(destination)
        if target.exists():
            raise FileExistsError("refusing to overwrite ancestry-RAG verification")
        write_json(target, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify a retrieved semantic-ancestry RAG G0 run")
    parser.add_argument("--root", required=True)
    parser.add_argument("--destination")
    args = parser.parse_args(argv)
    print(canonical_json(verify_run(args.root, args.destination)))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
